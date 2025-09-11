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

# ----------------------------
# CONFIG (edit these directly)
# ----------------------------
PG_HOST = "localhost"
PG_PORT = 5432
PG_DB = "mydb"
PG_USER = "readonly_user"
PG_PASSWORD = "readonly_password"
PG_SCHEMA = "public"
PG_TABLE = "member_survey"

HF_API_TOKEN = "hf_xxx"   # <-- put your HuggingFace API token here
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # or another text-gen model
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# ----------------------------
# DB Utilities
# ----------------------------
def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    )

def fetch_schema_and_samples(limit_rows: int = 8):
    with get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """, (PG_SCHEMA, PG_TABLE))
        cols = cur.fetchall()
        column_names = [c["column_name"] for c in cols]

        cur.execute(f'SELECT * FROM "{PG_SCHEMA}"."{PG_TABLE}" LIMIT {limit_rows};')
        rows = cur.fetchall()
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=column_names)
        return cols, df

def run_sql(sql: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn)

# ----------------------------
# Hugging Face Inference API
# ----------------------------
def hf_generate(prompt: str, max_new_tokens=300, temperature=0.2) -> str:
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
    }
    resp = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    else:
        return str(data)


# ----------------------------
# Prompt Engineering
# ----------------------------
def build_sql_prompt(user_question: str, columns_meta, samples_df: pd.DataFrame) -> str:
    col_lines = [f'- {c["column_name"]} ({c["data_type"]})' for c in columns_meta]
    sample_json = samples_df.head(5).to_dict(orient="records")
    today = date.today()
    last_month_start = (today.replace(day=1) - relativedelta(months=1))
    last_month_end = today.replace(day=1) - relativedelta(days=1)

    return f"""
You are an expert SQL analyst. 
Write a single valid PostgreSQL SELECT query for the table "{PG_SCHEMA}"."{PG_TABLE}".
- Use ONLY existing columns.
- No schema changes, no DML, only SELECT.
- If user says "last month", interpret as {last_month_start.isoformat()} to {last_month_end.isoformat()}.

Schema:
{chr(10).join(col_lines)}

Sample data:
{json.dumps(sample_json, indent=2)}

Question: {user_question}

Return ONLY the SQL inside <sql>...</sql>.
""".strip()

def build_answer_prompt(user_question: str, sql_text: str, result_df: pd.DataFrame) -> str:
    preview = result_df.head(5).to_dict(orient="records")
    meta = {"rows": len(result_df), "columns": list(result_df.columns)}
    return f"""
You are a helpful data analyst.

User question: {user_question}
SQL used: {sql_text}
Result preview: {json.dumps(preview)}
Meta: {json.dumps(meta)}

Write a clear, professional explanation of the answer in 2-5 sentences.
""".strip()

def extract_sql_from_text(txt: str) -> str:
    m = re.search(r"<sql>(.+?)</sql>", txt, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(";") + ";"
    parsed = sqlparse.parse(txt)
    if parsed:
        for stmt in parsed:
            s = stmt.to_unicode().strip()
            if s.upper().startswith("SELECT"):
                return s if s.endswith(";") else s + ";"
    return txt.strip()

# ----------------------------
# Pipeline
# ----------------------------
columns_meta, samples_df = fetch_schema_and_samples()

def answer_question(user_question: str):
    if not user_question.strip():
        return "<div style='color:red'>⚠️ Please ask a question.</div>"

    sql_prompt = build_sql_prompt(user_question, columns_meta, samples_df)
    raw = hf_generate(sql_prompt)
    sql_text = extract_sql_from_text(raw)

    try:
        df = run_sql(sql_text)
    except Exception as e:
        return f"""
        <div style="background:#ffe6e6;padding:10px;border-radius:8px">
        ❌ <b>SQL Error</b>: {e}<br><br>
        <code>{sql_text}</code>
        </div>
        """

    ans_prompt = build_answer_prompt(user_question, sql_text, df)
    try:
        explanation = hf_generate(ans_prompt, max_new_tokens=250, temperature=0.3)
    except Exception as e:
        explanation = f"Query executed but explanation failed: {e}"

    # Format as chat-like card
    return f"""
    <div style="background:#f9f9f9;padding:15px;border-radius:12px;
                box-shadow:0 2px 5px rgba(0,0,0,0.1)">
        <div style="font-size:14px;color:#555;margin-bottom:8px">🧾 <b>Generated SQL</b></div>
        <pre style="background:#272822;color:#f8f8f2;padding:10px;border-radius:8px;
                    overflow-x:auto;font-size:13px">{sql_text}</pre>
        <div style="font-size:14px;color:#555;margin-top:12px">💡 <b>Answer</b></div>
        <div style="background:#e8f4ff;padding:10px;border-radius:8px;
                    font-size:14px;line-height:1.5;color:#333">{explanation}</div>
    </div>
    """

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
                value="https://cdn-icons-png.flaticon.com/512/4712/4712028.png",  # 🔵 Blue brain icon
                show_label=False,
                height=120
            )
        with gr.Column():
            gr.Markdown("## 🤖 **Brainy Data Analyst Chatbot**\nAsk me questions in natural language → I’ll generate SQL + explain insights.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Ask a Question 💬")
            question = gr.Textbox(
                label="", 
                placeholder="e.g. How many survey responses were there last month?",
                lines=3
            )
            # 🟦 Put Run + Clear on the same row
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
