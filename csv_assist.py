import gradio as gr
import pandas as pd
import plotly.express as px
import ollama, re, json, os, logging, numpy as np

MODEL_NAME   = "mistral"
RULES_FILE   = "rules.json"
LOG_FILE     = "chatbot.log"
MAX_HISTORY  = 5

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

df = None
conversation_history = []
custom_rules = {}

# ---------- Utilities ----------
def load_rules():
    global custom_rules
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE) as f: custom_rules = json.load(f)
    else:
        custom_rules = {}
def save_rules(): 
    with open(RULES_FILE, "w") as f: json.dump(custom_rules, f, indent=2)
load_rules()

# ---------- CSV Cleaning ----------
def clean_csv(file):
    global df, conversation_history
    conversation_history = []
    na_vals = ["NA","N/A","null","missing","None",""]
    df_local = pd.read_csv(file.name, low_memory=False, na_values=na_vals)
    df_local.dropna(axis=0, how="all", inplace=True)
    df_local.dropna(axis=1, how="all", inplace=True)
    df_local = df_local.apply(lambda x: x.str.strip() if x.dtype=="object" else x)
    for c in df_local.columns:
        if df_local[c].dtype=="object":
            s=df_local[c].dropna().astype(str)
            if s.empty: continue
            num_like=s.str.replace(r"[^0-9\.-]","",regex=True)
            if num_like.apply(lambda v:v.replace('-','',1).replace('.','',1).isdigit()).mean()>0.5:
                df_local[c]=pd.to_numeric(num_like,errors="coerce")
    for c in df_local.columns:
        if any(k in c.lower() for k in["date","time","timestamp"]):
            try: df_local[c]=pd.to_datetime(df_local[c],errors="coerce")
            except: pass
    before=len(df_local); df_local.drop_duplicates(inplace=True)
    globals()["df"]=df_local
    msg=f"✅ Loaded {df_local.shape[0]} rows × {df_local.shape[1]} columns\nColumns: {list(df_local.columns)}"
    logging.info(msg)
    return msg

# ---------- Data Quality ----------
def data_quality_report():
    if df is None: return pd.DataFrame({"info":["Upload CSV first."]})
    rows=[]
    for c in df.columns:
        s=df[c]; rows.append({
            "column":c,"dtype":str(s.dtype),
            "missing_%":round(s.isna().mean()*100,2),
            "unique":int(s.nunique(dropna=True))
        })
    return pd.DataFrame(rows)

# ---------- LLM helpers ----------
def ask_llm(prompt):
    try:
        r=ollama.chat(model=MODEL_NAME,messages=[{"role":"user","content":prompt}])
        return r["message"]["content"].strip()
    except Exception as e:
        logging.error(e); return "(LLM unavailable)"

def safe_eval(code, local_vars):
    safe_globals={"__builtins__":{"len":len,"min":min,"max":max,"sum":sum,"abs":abs}}
    return eval(code,safe_globals,local_vars)

# ---------- Plot Auto ----------
def generate_plot_auto(df_res):
    if df_res is None or not isinstance(df_res,pd.DataFrame) or df_res.empty: return None
    num=df_res.select_dtypes(include=np.number).columns.tolist()
    cat=df_res.select_dtypes(exclude=np.number).columns.tolist()
    try:
        if len(cat)>=1 and len(num)>=1:
            fig=px.bar(df_res,x=cat[0],y=num[0])
        elif len(num)>=2:
            fig=px.scatter(df_res,x=num[0],y=num[1])
        elif len(cat)>=2:
            if "count" in df_res.columns:
                fig=px.bar(df_res,x=cat[0],y="count",color=cat[1])
            else:
                fig=px.histogram(df_res,x=cat[0],color=cat[1])
        else:
            fig=None
        if fig: fig.update_layout(template="plotly_white",height=400)
        return fig
    except: return None

# ---------- Rule handler ----------
def handle_rule_definition(q):
    if "if i say" in q.lower() and "use" in q.lower():
        key=q.split("if i say",1)[1].split(",")[0].strip().lower()
        rule=q.split("use",1)[1].strip()
        custom_rules[key]=rule; save_rules()
        return f"✅ Remembered rule for '{key}': {rule}",True
    return None,False

# ---------- Main logic ----------
def chat_respond(message, history):
    global df,conversation_history
    if message.strip()=="":
        return "Please enter a question."

    # handle upload reference (if no df yet)
    if df is None:
        return "⚠️ Please upload a CSV first using ➕ above."

    rule_resp,is_rule=handle_rule_definition(message)
    if is_rule: return rule_resp

    rules_txt="\n".join([f"If user asks '{k}', use: {v}" for k,v in custom_rules.items()])
    context="\n".join([f"User:{q}\nBot:{a}" for q,a in conversation_history[-MAX_HISTORY:]])

    prompt=f"""
DataFrame df columns: {list(df.columns)}.
{rules_txt}

Context:
{context}

Generate ONE concise pandas expression using df (and pd if needed)
that returns a DataFrame or Series to answer:
"{message}"
Return only the code, no prose.
"""
    code=ask_llm(prompt)
    code=re.sub(r"`+","",code).strip()

    try:
        result=safe_eval(code,{"df":df,"pd":pd})
        if isinstance(result,pd.Series):
            result=result.reset_index(); result.columns=[f"col_{i}" for i in range(result.shape[1])]
        if isinstance(result,(list,tuple)): result=pd.DataFrame(result,columns=["result"])
        if not isinstance(result,pd.DataFrame): result=pd.DataFrame({"result":[str(result)]})
        fig=generate_plot_auto(result.head(50))
        text=result.head(10).to_string(index=False)

        summary_prompt=f"""You are a business analyst.
Here are results:
{text}
Summarize key insight(s) in one short paragraph."""
        summary=ask_llm(summary_prompt)
        conversation_history.append((message,summary))

        # display table preview under chart
        if fig:
            return (gr.update(value=summary, visible=True),
                    gr.update(value=fig, visible=True),
                    gr.update(value=result.head(20), visible=True))
        else:
            return (summary,None,result.head(20))
    except Exception as e:
        return f"❌ Error executing code: {e}"

# ---------- UI ----------
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🤖 Enterprise CSV Data Analyst — Chat Style")

    with gr.Row():
        upload = gr.UploadButton("➕ Upload CSV", file_types=[".csv"], label="Upload")
        dq_btn = gr.Button("View Data Quality Report")

    dq_table = gr.Dataframe(visible=False, label="Data Quality Report")

    with gr.Tab("Chat"):
        chatbot = gr.ChatInterface(
            fn=chat_respond,
            chatbot=gr.Chatbot(label="Conversation"),
            textbox=gr.Textbox(placeholder="Ask a question or define a rule..."),
            title="CSV Analyst Assistant",
            description="Upload CSV with ➕ above, then ask analytical questions. Click 📤 to send.",
            theme="soft",
            submit_btn="📤 Send",
            stop_btn="⏹ Stop"
        )

    upload.upload(clean_csv, upload, chatbot.chatbot)
    dq_btn.click(data_quality_report, outputs=dq_table)

app.launch()
