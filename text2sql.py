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
def build_sql_prompt(user_question: str, schemas: dict) -> str:
    table_descriptions = []
    for tbl, data in schemas.items():
        cols = [f'- {c["column_name"]} ({c["data_type"]})' for c in data["cols"]]
        table_descriptions.append(f"Table {tbl}:\n" + "\n".join(cols))

    # use small sample preview for context
    sample_preview = {
        tbl: data["sample"].head(3).to_dict(orient="records")
        for tbl, data in schemas.items()
    }

    return f"""
You are an expert SQL analyst.
We have two tables in schema "{PG_SCHEMA}":

{chr(10).join(table_descriptions)}

Relationships:
- member_survey_info.member_id = member_info.member_id

RULES:
- Use ONLY existing columns.
- If question references survey metrics (nps_score, csat_score, survey_date, survey_id, comments), those are in member_survey_info (alias ms).
- If question references member attributes (plan, county, business_segment, etc.), those are in member_info (alias mi).
- Always JOIN ms with mi on member_id when question needs both survey metrics and member attributes.
- Use table aliases: ms for member_survey_info, mi for member_info.
- Return only the SQL inside <sql>...</sql>.

Schema Samples:
{json.dumps(sample_preview, indent=2)}

### Examples:

User: Which plans have the lowest average NPS?
SQL:
<sql>
SELECT mi.plan, AVG(ms.nps_score) AS avg_nps, COUNT(*) AS total_surveys
FROM {PG_SCHEMA}.member_survey_info ms
JOIN {PG_SCHEMA}.member_info mi ON ms.member_id = mi.member_id
GROUP BY mi.plan
ORDER BY avg_nps ASC;
</sql>

User: Which counties had the most survey responses last month?
SQL:
<sql>
SELECT mi.county, COUNT(*) AS total_surveys
FROM {PG_SCHEMA}.member_survey_info ms
JOIN {PG_SCHEMA}.member_info mi ON ms.member_id = mi.member_id
WHERE ms.survey_date BETWEEN '2024-07-01' AND '2024-07-31'
GROUP BY mi.county
ORDER BY total_surveys DESC;
</sql>

User: How many promoters, passives, and detractors are there based on NPS score?
SQL:
<sql>
SELECT 
  CASE 
    WHEN ms.nps_score > 6 THEN 'Promoter'
    WHEN ms.nps_score < 5 THEN 'Detractor'
    ELSE 'Neutral'
  END AS nps_category,
  COUNT(*) AS total
FROM {PG_SCHEMA}.member_survey_info ms
GROUP BY nps_category;
</sql>

User: Which business segment has the most detractors?
SQL:
<sql>
SELECT mi.business_segment, COUNT(*) AS detractors
FROM {PG_SCHEMA}.member_survey_info ms
JOIN {PG_SCHEMA}.member_info mi ON ms.member_id = mi.member_id
WHERE ms.nps_score < 5
GROUP BY mi.business_segment
ORDER BY detractors DESC;
</sql>


User Question:
{user_question}
""".strip()

def enforce_types(sql: str, schemas: dict) -> str:
    for tbl, data in schemas.items():
        for c in data["cols"]:
            col = c["column_name"]
            dtype = c["data_type"].lower()
            # enforce quoting for text
            if "char" in dtype or "text" in dtype:
                sql = re.sub(rf"({col}\s*[=<>]\s*)(\d+)(\b)", r"\1'\2'\3", sql, flags=re.IGNORECASE)
            # enforce no quotes for integer/numeric
            elif "int" in dtype or "numeric" in dtype or "double" in dtype:
                sql = re.sub(rf"({col}\s*[=<>]\s*)'(\d+)'", r"\1\2", sql, flags=re.IGNORECASE)
    return sql


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
Focus on:
- biggest differences (e.g., Plan A has 25% lower CSAT than average),
- anomalies or outliers,
- any obvious trend (if a date column is present).

Keep it professional and concise.
""".strip()


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
        return f"<div style='background:#ffe6e6;padding:10px;border-radius:8px'>❌ SQL Error: {e}<br><br><code>{sql_text}</code></div>"

    # Explanation
    ans_prompt = build_answer_prompt(user_question, sql_text, df)
    explanation = hf_generate(ans_prompt, max_new_tokens=250, temperature=0.3)

    # Insights
    insights_prompt = build_insights_prompt(user_question, sql_text, df)
    insights = hf_generate(insights_prompt, max_new_tokens=250, temperature=0.4)

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
            gr.Markdown("## 🤖 🧠 **Brainy Data Analyst Chatbot**\nAsk me questions in natural language → I’ll generate SQL + explain insights.")
    
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
