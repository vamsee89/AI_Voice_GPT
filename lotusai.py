import json
import re
import sqlparse
import requests
import pandas as pd
import gradio as gr
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import date
from dateutil.relativedelta import relativedelta
from lotus_ai import Tool

# ----------------------------
# CONFIG
# ----------------------------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "mydb"
PG_USER = "readonly_user"
PG_PASSWORD = "readonly_password"
PG_SCHEMA = "public"

# Your inference API
INFERENCE_API_URL = "https://your-inference-api-endpoint/generate"
YOUR_API_KEY = "xxx"   # replace with your key

# ----------------------------
# DB Utilities
# ----------------------------
def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    )

def fetch_schema_and_samples(limit_rows: int = 8):
    tables = ["member_survey_info", "member_info"]
    schemas = {}
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        for tbl in tables:
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position;
            """, (PG_SCHEMA, tbl))
            cols = cur.fetchall()
            column_names = [c["column_name"] for c in cols]

            cur.execute(f'SELECT * FROM "{PG_SCHEMA}"."{tbl}" LIMIT {limit_rows};')
            rows = cur.fetchall()
            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=column_names)

            schemas[tbl] = {"cols": cols, "sample": df}
    return schemas

def run_sql(sql: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn)

# ----------------------------
# Lotus-AI Tool
# ----------------------------
class InferenceTool(Tool):
    name = "inference_tool"
    description = "Call inference API for SQL, insights, or text analysis"

    inputs = {
        "task": {"type": "text", "description": "sql, insight, or text_analysis"},
        "prompt": {"type": "text", "description": "Prompt to send to inference API"}
    }
    output_type = "text"

    def forward(self, task: str, prompt: str) -> str:
        resp = requests.post(
            INFERENCE_API_URL,
            json={"task": task, "prompt": prompt},
            headers={"Authorization": f"Bearer {YOUR_API_KEY}"}
        )
        resp.raise_for_status()
        return resp.json().get("output", "")

inference_tool = InferenceTool()

# ----------------------------
# Prompt Engineering
# ----------------------------
def build_sql_prompt(user_question: str, schemas: dict) -> str:
    # column definitions
    defs = []
    for tbl, data in schemas.items():
        col_defs = [f"- {c['column_name']} ({c['data_type']})" for c in data["cols"]]
        defs.append(f"Table {tbl}:\n" + "\n".join(col_defs))
    defs_text = "\n\n".join(defs)

    today = date.today()
    last_month_start = (today.replace(day=1) - relativedelta(months=1))
    last_month_end = today.replace(day=1) - relativedelta(days=1)

    return f"""
You are an expert SQL analyst.
We have two tables in schema "{PG_SCHEMA}".

Column Definitions:
{defs_text}

Relationships:
- member_survey_info.member_id = member_info.member_id

RULES:
- Use ONLY existing columns.
- If column type is text/varchar → use single quotes around values.
- If column type is numeric → do not use quotes.
- If column type is date → use 'YYYY-MM-DD'.
- Use table aliases: ms = {PG_SCHEMA}.member_survey_info, mi = {PG_SCHEMA}.member_info.
- Always qualify table names with schema.
- If user says "last month", interpret as {last_month_start.isoformat()} to {last_month_end.isoformat()}.
- Return ONLY a valid PostgreSQL SELECT query inside <sql>...</sql>.

User Question:
{user_question}
""".strip()

def build_answer_prompt(user_question: str, sql_text: str, result_df: pd.DataFrame) -> str:
    preview = result_df.head(5).to_dict(orient="records")
    meta = {"rows": len(result_df), "columns": list(result_df.columns)}
    return f"""
You are a helpful data analyst.

User question: {user_question}
SQL used: {sql_text}
Result preview: {json.dumps(preview, default=str)}
Meta: {json.dumps(meta)}

Write a clear, professional explanation of the answer in 2-5 sentences.
""".strip()

def build_insights_prompt(user_question: str, sql_text: str, result_df: pd.DataFrame) -> str:
    preview = result_df.head(15).to_dict(orient="records")
    desc = result_df.describe(include="all").to_dict() if not result_df.empty else {}

    return f"""
You are a senior data analyst.
The user asked: {user_question}
The SQL used: {sql_text}
Preview of results: {json.dumps(preview, default=str)}
Basic stats: {json.dumps(desc, default=str)}

Write a short section called 'Notable Findings' in bullet points.
""".strip()

def extract_sql_from_text(txt: str) -> str:
    m = re.search(r"<sql>(.+?)</sql>", txt, flags=re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
    else:
        parsed = sqlparse.parse(txt)
        sql = ""
        for stmt in parsed or []:
            s = str(stmt).strip()
            if s.upper().startswith("SELECT"):
                sql = s
                break
    if not sql:
        return "SELECT 1;"
    if not sql.endswith(";"):
        sql += ";"
    return sql

def qualify_table_names(sql: str) -> str:
    sql = re.sub(r'\bmember_survey_info\b', f'{PG_SCHEMA}.member_survey_info', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bmember_info\b', f'{PG_SCHEMA}.member_info', sql, flags=re.IGNORECASE)
    return sql

def enforce_types_and_ops(sql: str, schemas: dict) -> str:
    for tbl, data in schemas.items():
        for c in data["cols"]:
            col = c["column_name"]
            dtype = c["data_type"].lower()
            if "char" in dtype or "text" in dtype:
                sql = re.sub(rf"({col}\s*(=|<|>|<=|>=|!=|<>)\s*)(\d+)(\b)", r"\1'\3'\4", sql, flags=re.IGNORECASE)
            elif "int" in dtype or "numeric" in dtype or "double" in dtype:
                sql = re.sub(rf"({col}\s*(=|<|>|<=|>=|!=|<>)\s*)'(\d+)'", r"\1\3", sql, flags=re.IGNORECASE)
    return sql

# ----------------------------
# Helpers
# ----------------------------
def is_comment_question(user_question: str) -> bool:
    keywords = ["comment", "feedback", "text", "suggestion", "negative", "positive", "sentiment", "topic"]
    return any(k in user_question.lower() for k in keywords)

def render_sql_error(sql, e):
    return f"<div style='background:#ffe6e6;padding:10px;border-radius:8px'>❌ SQL Error: {e}<br><br><code>{sql}</code></div>"

def render_final(sql_text, explanation, insights):
    return f"""
    <div style="background:#f9f9f9;padding:15px;border-radius:12px;box-shadow:0 2px 5px rgba(0,0,0,0.1)">
        <div style="font-size:14px;color:#555;margin-bottom:8px">🧾 <b>Generated SQL</b></div>
        <pre style="background:#272822;color:#f8f8f2;padding:10px;border-radius:8px;overflow-x:auto;font-size:13px">{sql_text}</pre>
        <div style="font-size:14px;color:#555;margin-top:12px">💡 <b>Answer</b></div>
        <div style="background:#e8f4ff;padding:10px;border-radius:8px;font-size:14px;line-height:1.5;color:#333">{explanation}</div>
        <div style="font-size:14px;color:#555;margin-top:12px">✨ <b>Notable Findings</b></div>
        <div style="background:#fff7e6;padding:10px;border-radius:8px;font-size:14px;line-height:1.5;color:#333">{insights}</div>
    </div>
    """

def render_text_analysis(analysis):
    return f"""
    <div style="background:#f9f9f9;padding:15px;border-radius:12px;box-shadow:0 2px 5px rgba(0,0,0,0.1)">
        <div style="font-size:14px;color:#555;margin-top:12px">📝 <b>Comment Insights</b></div>
        <div style="background:#fff7e6;padding:10px;border-radius:8px;font-size:14px;line-height:1.5;color:#333">{analysis}</div>
    </div>
    """

# ----------------------------
# Pipeline
# ----------------------------
schemas = fetch_schema_and_samples()

def answer_question(user_question: str):
    if not user_question.strip():
        return "<div style='color:red'>⚠️ Please ask a question.</div>"

    # Case 1: comment/text analysis
    if is_comment_question(user_question):
        try:
            df = run_sql(f'SELECT comments FROM "{PG_SCHEMA}"."member_survey_info" WHERE comments IS NOT NULL LIMIT 200;')
            comments = df["comments"].dropna().tolist()
        except Exception as e:
            return render_sql_error("SELECT comments ...", e)

        prompt = f"""
User asked: {user_question}
Analyze the following comments and provide sentiment/topics.
Comments: {comments[:50]}
"""
        analysis = inference_tool(task="text_analysis", prompt=prompt)
        return render_text_analysis(analysis)

    # Case 2: SQL + structured insights
    sql_prompt = build_sql_prompt(user_question, schemas)
    raw = inference_tool(task="sql", prompt=sql_prompt)
    sql_text = extract_sql_from_text(raw)
    sql_text = qualify_table_names(sql_text)
    sql_text = enforce_types_and_ops(sql_text, schemas)

    try:
        df = run_sql(sql_text)
    except Exception as e:
        return render_sql_error(sql_text, e)

    ans_prompt = build_answer_prompt(user_question, sql_text, df)
    explanation = inference_tool(task="insight", prompt=ans_prompt)

    insights_prompt = build_insights_prompt(user_question, sql_text, df)
    insights = inference_tool(task="insight", prompt=insights_prompt)

    return render_final(sql_text, explanation, insights)

# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(css="""
    body {background:#f0f2f5}
    .gr-row {gap: 20px; display:flex}
    .gr-column {flex:1; padding:20px}
    .gr-column:first-child {
        background:#fff;border-radius:12px;
        box-shadow:0 2px 6px rgba(0,0,0,0.1);flex:1
    }
    .gr-column:last-child {
        flex:2
    }
""") as demo:
    with gr.Row():
        with gr.Column():
            gr.Image(
                value="brainy.png",  # local brain icon
                show_label=False,
                height=100
            )
        with gr.Column():
            gr.Markdown("## 🧠 **Brainy Data Analyst Chatbot**\nAsk questions → I’ll generate SQL or analyze comments with insights.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Ask a Question 💬")
            question = gr.Textbox(
                label="", 
                placeholder="e.g. Which plan has the lowest NPS? or How many negative comments last month?",
                lines=3
            )
            with gr.Row():
                run_btn = gr.Button("▶️ Run", variant="primary")
                clear_btn = gr.Button("🧹 Clear")
        with gr.Column():
            output = gr.HTML(label="Output")

    run_btn.click(fn=answer_question, inputs=question, outputs=output)
    clear_btn.click(lambda: "", None, output)
    clear_btn.click(lambda: "", None, question)

if __name__ == "__main__":
    demo.launch()
