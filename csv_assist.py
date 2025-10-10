import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests, json, re, numpy as np, socket, traceback
from typing import Optional, Tuple

# ============================================================
# ⚙️  API Configuration
# ============================================================
API_ENDPOINT = "https://your-domain.com/llama-4-scout-17B-16E-Instruct/v1/chat/completions"
API_HEADERS = {"Authorization": "Basic cccccccccc", "Content-Type": "application/json"}

# ============================================================
# 🧠  CSV Chatbot
# ============================================================
class CSVChatbot:
    def __init__(self):
        self.df = None
        self.df_info = None
        self.conversation_history = []

    # --------------------------------------------------------
    def load_csv(self, file) -> Tuple[str, str]:
        """Load and summarize CSV"""
        try:
            self.df = pd.read_csv(file)
            self.df_info = {
                "columns": list(self.df.columns),
                "shape": self.df.shape,
                "dtypes": self.df.dtypes.astype(str).to_dict(),
                "head": self.df.head(5).to_string(),
                "missing": self.df.isnull().sum().to_dict()
            }

            # use escaped backticks so it copies cleanly
            summary = f"""✅ CSV Loaded!

**Rows:** {self.df.shape[0]}  
**Columns:** {self.df.shape[1]}  
**Names:** {', '.join(self.df.columns)}

**Missing Values:**
{chr(10).join([f"- {k}: {v}" for k, v in self.df_info['missing'].items() if v>0]) or "No missing values"}

**Preview:**
\\`\\`\\`text
{self.df_info['head']}
\\`\\`\\`
"""
            return summary, "CSV file loaded successfully!"
        except Exception as e:
            return f"❌ Error loading CSV: {e}", "Error loading file"

    # --------------------------------------------------------
    def is_safe_query(self, query: str) -> Tuple[bool, str]:
        """Check for malicious code or shell access"""
        patterns = [
            r'import\\s+os', r'subprocess', r'eval\\s*\\(', r'exec\\s*\\(',
            r'socket', r'open\\s*\\(', r'pickle', r'system', r';', r'&&', r'`'
        ]
        for p in patterns:
            if re.search(p, query, re.I):
                return False, "⚠️ Potentially unsafe operation detected."
        return True, "Safe"

    # --------------------------------------------------------
    def _compress_context(self) -> str:
        """Generate concise dataset context for LLM"""
        if not self.df_info:
            return ""
        head = self.df_info['head']
        sample_rows = "\\n".join(head.split("\\n")[:8])
        return f"Columns: {', '.join(self.df_info['columns'])}\\n\\nSample data:\\n{sample_rows}"

    # --------------------------------------------------------
    def generate_llm_response(self, query: str) -> str:
        """Send contextual question to LLM"""
        if self.df is None:
            return "Please upload a CSV first."

        context = f"""
You are a data analyst assistant. The dataset has {self.df_info['shape'][0]} rows and {self.df_info['shape'][1]} columns.
{self._compress_context()}

User Query: {query}
"""

        # Keep short memory
        recent = self.conversation_history[-4:]
        messages = [{"role": "system", "content": "You analyze CSV data and explain insights clearly."}]
        for turn in recent:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": context})

        payload = {"messages": messages, "max_tokens": 800, "temperature": 0.7}

        try:
            r = requests.post(API_ENDPOINT, headers=API_HEADERS, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]["content"]
                self.conversation_history.append({"user": query, "assistant": msg})
                return msg
            return f"API Error {r.status_code}: {r.text}"
        except Exception as e:
            traceback.print_exc()
            return f"Error contacting API: {e}"

    # --------------------------------------------------------
    def execute_analysis(self, query: str) -> Tuple[str, Optional[go.Figure]]:
        """Generate LLM response and optional visualization"""
        if self.df is None:
            return "Please upload a CSV first.", None

        ok, msg = self.is_safe_query(query)
        if not ok:
            return msg, None

        llm_resp = self.generate_llm_response(query)
        viz_keywords = ['plot','chart','graph','visual','hist','scatter','bar',
                        'line','box','pie','heatmap','correlation','3d','violin','sunburst']

        if any(k in query.lower() for k in viz_keywords):
            fig = self.auto_generate_plot(query)
            return llm_resp, fig
        return llm_resp, None

    # --------------------------------------------------------
    def auto_generate_plot(self, query: str) -> Optional[go.Figure]:
        """Heuristic visualization generator"""
        try:
            q = query.lower()
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            cat_cols = self.df.select_dtypes(include=['object']).columns
            fig = None

            if 'hist' in q and len(num_cols):
                fig = px.histogram(self.df, x=num_cols[0], nbins=30, title=f"Distribution of {num_cols[0]}")
            elif 'scatter' in q and len(num_cols) >= 2:
                color = cat_cols[0] if len(cat_cols) else None
                fig = px.scatter(self.df, x=num_cols[0], y=num_cols[1], color=color)
            elif 'bar' in q and len(cat_cols):
                vc = self.df[cat_cols[0]].value_counts().head(10)
                fig = px.bar(x=vc.index, y=vc.values, title=f"Count of {cat_cols[0]}")
            elif 'line' in q and len(num_cols):
                fig = px.line(self.df, y=num_cols[0], title=f"Trend of {num_cols[0]}")
            elif 'box' in q and len(num_cols):
                fig = go.Figure([go.Box(y=self.df[c], name=c) for c in num_cols[:5]])
            elif 'correlation' in q or 'heatmap' in q and len(num_cols) > 1:
                corr = self.df[num_cols].corr()
                fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
            elif 'pie' in q and len(cat_cols):
                vc = self.df[cat_cols[0]].value_counts().head(10)
                fig = px.pie(values=vc.values, names=vc.index, hole=0.3)
            elif '3d' in q and len(num_cols) >= 3:
                fig = px.scatter_3d(self.df, x=num_cols[0], y=num_cols[1], z=num_cols[2])
            elif 'violin' in q and len(num_cols):
                fig = go.Figure([go.Violin(y=self.df[c], name=c) for c in num_cols[:5]])
            elif 'sunburst' in q and len(cat_cols) >= 2:
                fig = px.sunburst(self.df, path=[cat_cols[0], cat_cols[1]])

            if fig:
                fig.update_layout(template="plotly_white", height=500, margin=dict(l=40, r=40, t=50, b=40))
            return fig
        except Exception as e:
            print("Plot error:", e)
            return None


# ============================================================
# 🖥️  Gradio Interface
# ============================================================
chatbot = CSVChatbot()

def upload_file(file):
    if file is None:
        return "Please upload a file", "No file uploaded", None, None
    summary, status = chatbot.load_csv(file)
    return summary, status, None, None

def chat(message, history):
    if chatbot.df is None:
        return history + [[message, "⚠️ Please upload a CSV file first."]], None
    reply, fig = chatbot.execute_analysis(message)
    return history + [[message, reply]], fig


# ============================================================
# 💡  UI Layout
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(), title="CSV Data Chatbot") as demo:
    gr.Markdown("""
    # 📊 CSV Data Analysis Chatbot  
    ### 🤖 Powered by Llama-4 Scout (Internal API)
    Upload your CSV and chat with your dataset — ask questions or request charts!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="📁 Upload CSV File", file_types=[".csv"])
            upload_btn = gr.Button("Load CSV", variant="primary")

            dataset_info = gr.Textbox(value="", label="📄 Dataset Summary",
                                      lines=15, max_lines=20, interactive=False)
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="Chat with your Data", height=400)
            plot_output = gr.Plot(label="Visualization")

            msg_input = gr.Textbox(
                label="💬 Ask a question",
                placeholder="e.g. 'Show me a histogram of revenue' or 'What’s the average age?'",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                clear_btn = gr.Button("🧹 Clear Chat")

    status_text = gr.Textbox(value="", label="Status", interactive=False)

    gr.Markdown("""
    ### 💡 Example Questions
    - "Show the first 10 rows"  
    - "Create a histogram of revenue"  
    - "Show correlation heatmap"  
    - "Plot a scatter of Age vs Salary"
    """)

    # Event wiring
    upload_btn.click(upload_file, inputs=[file_upload],
                     outputs=[dataset_info, status_text, chatbot_ui, plot_output])
    submit_btn.click(chat, inputs=[msg_input, chatbot_ui],
                     outputs=[chatbot_ui, plot_output]).then(lambda: "", outputs=[msg_input])
    msg_input.submit(chat, inputs=[msg_input, chatbot_ui],
                     outputs=[chatbot_ui, plot_output]).then(lambda: "", outputs=[msg_input])
    clear_btn.click(lambda: (None, None), outputs=[chatbot_ui, plot_output])

# ============================================================
# 🚀  Launch Helper
# ============================================================
def find_free_port(start=7860, end=7890):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    raise RuntimeError("No free ports found")

if __name__ == "__main__":
    port = find_free_port()
    print(f"🔍 Launching on port {port}...")
    try:
        demo.launch(server_name="0.0.0.0", server_port=port, share=False)
    except Exception as e:
        print(f"⚠️ Localhost failed: {e}\\nRetrying with share=True...")
        demo.launch(share=True)
