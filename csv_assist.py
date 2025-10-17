import gradio as gr
import pandas as pd
import plotly.express as px
import ollama
import re, json, os, logging

# ----------------------------
# CONFIG
# ----------------------------
RULES_FILE = "rules.json"
MODEL_NAME = "mistral"  # your Ollama model
MAX_HISTORY = 5
LOG_FILE = "chatbot.log"

# ----------------------------
# LOGGING SETUP
# ----------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# GLOBAL STATE
# ----------------------------
df = None
conversation_history = []
custom_rules = {}

# ----------------------------
# PERSISTENCE HELPERS
# ----------------------------
def load_rules():
    global custom_rules
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE) as f:
                custom_rules = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load rules: {e}")
            custom_rules = {}

def save_rules():
    try:
        with open(RULES_FILE, "w") as f:
            json.dump(custom_rules, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save rules: {e}")

load_rules()

# ----------------------------
# DATA LOADING + CLEANING
# ----------------------------
def load_csv(file):
    global df, conversation_history
    conversation_history = []

    try:
        df = pd.read_csv(file.name, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file.name, encoding="latin1", low_memory=False)
    except Exception as e:
        logging.error(f"CSV Load Error: {e}")
        return f"❌ Failed to load CSV: {e}"

    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    mixed_columns = []
    for col in df.columns:
        sample = df[col].dropna()
        if len(sample) == 0:
            continue
        unique_types = set(type(v) for v in sample.sample(min(50, len(sample))))
        if len(unique_types) > 1:
            mixed_columns.append(col)
            df[col] = pd.to_numeric(df[col].replace('[^0-9\.-]', '', regex=True), errors="coerce")

    summary = f"""
✅ CSV Loaded and Cleaned  
📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns  
⚙️ Mixed-Type Columns Fixed: {mixed_columns if mixed_columns else 'None'}  
"""
    logging.info(f"CSV Loaded: {file.name} | Columns: {list(df.columns)}")
    return summary

# ----------------------------
# MODEL CALL
# ----------------------------
def ask_llm(prompt):
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "❌ LLM call failed."

# ----------------------------
# SAFE EXECUTION SANDBOX
# ----------------------------
def safe_eval(code, local_vars):
    try:
        # Limit builtins for safety
        safe_globals = {"__builtins__": {"len": len, "min": min, "max": max, "sum": sum}}
        return eval(code, safe_globals, local_vars)
    except Exception as e:
        logging.warning(f"Code execution error: {e}")
        raise

# ----------------------------
# PLOTLY CHART
# ----------------------------
def generate_plot(result):
    try:
        if isinstance(result, pd.Series):
            r = result.reset_index()
            r.columns = ["x", "y"]
            return px.bar(r, x="x", y="y", title="Series Result")

        elif isinstance(result, pd.DataFrame):
            num_cols = result.select_dtypes(include="number").columns
            if len(num_cols) >= 2:
                return px.line(result, x=num_cols[0], y=num_cols[1:], title="Line Plot")
            elif len(num_cols) == 1:
                return px.bar(result, x=result.index, y=num_cols[0], title="Bar Plot")
            else:
                return px.bar(result.head(10))
        else:
            return None
    except Exception as e:
        logging.error(f"Plot Error: {e}")
        return None

# ----------------------------
# RULE PARSER
# ----------------------------
def handle_rule_definition(question):
    q_lower = question.lower()
    if "if i say" in q_lower and "use" in q_lower:
        try:
            key = question.split("if i say", 1)[1].split(",")[0].strip().lower()
            rule = question.split("use", 1)[1].strip()
            custom_rules[key] = rule
            save_rules()
            logging.info(f"Rule added: {key} -> {rule}")
            return f"✅ Remembered this rule for '{key}': {rule}", True
        except Exception as e:
            logging.error(f"Rule parse error: {e}")
            return f"⚠️ Couldn't parse rule: {e}", True
    return None, False

# ----------------------------
# MAIN ANALYSIS LOGIC
# ----------------------------
def analyze_question(question):
    global df, conversation_history, custom_rules

    if df is None:
        return "⚠️ Upload a CSV first.", "", None

    # Check if it's a rule definition
    rule_resp, is_rule = handle_rule_definition(question)
    if is_rule:
        return rule_resp, "", None

    # Build custom rules context
    rule_context = "\n".join([f"If user asks '{k}', use this logic: {v}" for k, v in custom_rules.items()])
    context = "\n".join([f"User: {q}\nBot: {a}" for q, a in conversation_history[-MAX_HISTORY:]])

    prompt = f"""
You are a data analyst using pandas DataFrame `df` with columns {list(df.columns)}.
Custom analytical rules:
{rule_context}

Recent context:
{context}

Generate a concise, safe pandas expression using df to answer:
"{question}"

Do not use imports or print statements. Return only the code.
    """

    code = ask_llm(prompt)
    code_line = re.findall(r"(?s)`([^`]*)`", code)
    if code_line:
        code = code_line[0]
    code = code.strip()

    try:
        result = safe_eval(code, {"df": df, "pd": pd})
        if isinstance(result, (pd.Series, pd.DataFrame)):
            text_result = result.head(10).to_string()
        else:
            text_result = str(result)

        fig = generate_plot(result)
        conversation_history.append((question, text_result))
        logging.info(f"Query: {question} | Code: {code}")
        return text_result, code, fig
    except Exception as e:
        logging.warning(f"Execution failed: {e}")
        return f"❌ Error executing code: {e}", code, None

# ----------------------------
# GRADIO UI (v4.19.2 Compatible)
# ----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🧠 Enterprise CSV Data Analyst Chatbot — Secure, Context-Aware, Local LLM")

    with gr.Row():
        csv_file = gr.File(label="📂 Upload CSV")
        load_btn = gr.Button("Load CSV")
    load_status = gr.Textbox(label="Dataset Info")

    with gr.Row():
        question = gr.Textbox(
            label="Ask a question or define a rule",
            placeholder="e.g., Calculate average sales OR If I say calculate NPS, use promoters=9–10, detractors=0–6"
        )
        analyze_btn = gr.Button("Analyze")

    result_box = gr.Textbox(label="🧠 Analytical Answer", lines=10)
    code_box = gr.Code(label="💡 Generated Pandas Code", language="python")
    chart_output = gr.Plot(label="📊 Interactive Visualization")

    load_btn.click(load_csv, inputs=csv_file, outputs=load_status)
    analyze_btn.click(analyze_question, inputs=question, outputs=[result_box, code_box, chart_output])

app.launch()
