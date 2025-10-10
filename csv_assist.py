import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests, json, re, numpy as np, traceback
from typing import Optional, Tuple


# ============================================================
# ⚙️  API CONFIGURATION
# ============================================================
API_ENDPOINT = "https://your-domain.com/llama-4-scout-17B-16E-Instruct/v1/chat/completions"
API_HEADERS = {"Authorization": "Basic cccccccccc", "Content-Type": "application/json"}


# ============================================================
# 🧠  CSV CHATBOT CORE
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

            summary = (
                f"✅ CSV Loaded!\n\n"
                f"**Rows:** {self.df.shape[0]}  \n"
                f"**Columns:** {self.df.shape[1]}  \n"
                f"**Names:** {', '.join(self.df.columns)}\n\n"
                f"**Missing Values:**\n"
                f"{chr(10).join([f'- {k}: {v}' for k, v in self.df_info['missing'].items() if v>0]) or 'No missing values'}\n\n"
                f"**Preview:**\n```text\n{self.df_info['head']}\n```"
            )
            return summary, "CSV file loaded successfully!"
        except Exception as e:
            return f"❌ Error loading CSV: {e}", "Error loading file"

    # --------------------------------------------------------
    def is_safe_query(self, query: str) -> Tuple[bool, str]:
        """Detect dangerous operations"""
        patterns = [
            r'import\s+os', r'subprocess', r'eval\s*\(', r'exec\s*\(',
            r'socket', r'open\s*\(', r'pickle', r'system', r';', r'&&', r'`'
        ]
        for p in patterns:
            if re.search(p, query, re.I):
                return False, "⚠️ Potentially unsafe operation detected."
        return True, "Safe"

    # --------------------------------------------------------
    def _compress_context(self) -> str:
        """Compact summary for LLM"""
        if not self.df_info:
            return ""
        head = self.df_info['head']
        sample_rows = "\n".join(head.split("\n")[:8])
        return "Columns: %s\n\nSample data:\n%s" % (
            ", ".join(self.df_info['columns']),
            sample_rows,
        )

    # --------------------------------------------------------
    def generate_llm_response(self, query: str) -> str:
        if self.df is None:
            return "Please upload a CSV first."

        context = (
            "You are a data analyst assistant. The dataset has %d rows and %d columns.\n"
            "%s\n\nUser Query: %s"
            % (self.df_info['shape'][0], self.df_info['shape'][1], self._compress_context(), query)
        )

        history = self.conversation_history[-3:]
        messages = [{"role": "system", "content": "You analyze CSV data and explain insights clearly."}]
        for turn in history:
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
            return "API Error %d: %s" % (r.status_code, r.text)
        except Exception as e:
            traceback.print_exc()
            return "Error contacting API: %s" % str(e)

    # --------------------------------------------------------
    def execute_analysis(self, query: str) -> Tuple[str, Optional[go.Figure]]:
        """Run query + optional plot"""
        if self.df is None:
            return "Please upload a CSV first.", None

        ok, msg = self.is_safe_query(query)
        if not ok:
            return msg, None

        llm_resp = self.generate_llm_response(query)
        viz_keywords = [
            'plot','chart','graph','visual','hist','scatter','bar',
            'line','box','pie','heatmap','correlation','3d','violin','sunburst'
        ]
        if any(k in query.lower() for k in viz_keywords):
            fig = self.auto_generate_plot(query)
            return llm_resp, fig
        return llm_resp, None

    # --------------------------------------------------------
    def auto_generate_plot(self, query: str) -> Optional[go.Figure]:
        """Heuristic auto-plot generator"""
        try:
            q = query.lower()
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            cat_cols = self.df.select_dtypes(include=['object']).columns
            fig = None

            if 'hist' in q and len(num_cols):
                fig = px.histogram(self.df, x=num_cols[0], nbins=30, title="Distribution of %s" % num_cols[0])
            elif 'scatter' in q and len(num_cols) >= 2:
                color = cat_cols[0] if len(cat_cols) else None
                fig = px.scatter(self.df, x=num_cols[0], y=num_cols[1], color=color)
            elif ('bar' in q or 'count' in q) and len(cat_cols):
                vc = self.df[cat_cols[0]].value_counts().head(10)
                fig = px.bar(x=vc.index, y=vc.values, title="Count of %s" % cat_cols[0])
            elif 'line' in q and len(num_cols):
                fig = px.line(self.df, y=num_cols[0], title="Trend of %s" % num_cols[0])
            elif 'box' in q and len(num_cols):
                traces = [go.Box(y=self.df[c], name=c) for c in num_cols[:5]]
                fig = go.Figure(data=traces)
            elif ('correlation' in q or 'heatmap' in q) and len(num_cols) > 1:
                corr = self.df[num_cols].corr()
                fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
            elif 'pie' in q and len(cat_cols):
                vc = self.df[cat_cols[0]].value_counts().head(10)
                fig = px.pie(values=vc.values, names=vc.index, hole=0.3)
            elif '3d' in q and len(num_cols) >= 3:
                fig = px.scatter_3d(self.df, x=num_cols[0], y=num_cols[1], z=num_cols[2])
            elif 'violin' in q and len(num_cols):
                traces = [go.Violin(y=self.df[c], name=c) for c in num_cols[:5]]
                fig = go.Figure(data=traces)
            elif 'sunburst' in q and len(cat_cols) >= 2:
                fig = px.sunburst(self.df, path=[cat_cols[0], cat_cols[1]])

            if fig:
                fig.update_layout(template="plotly_white", height=500, margin=dict(l=40, r=40, t=50, b=40))
            return fig
        except Exception as e:
            print("Plot error:", e)
            return None


# ============================================================
# 🖥️  GRADIO UI + CALLBACKS
# ============================================================
chatbot = CSVChatbot()


def upload_file(file):
    if file is None:
        return "Please upload a file", "No file uploaded", [], gr.update(value=None)
    summary, status = chatbot.load_csv(file)
    return summary, status, [], gr.update(value=None)


def chat(message, history):
    if chatbot.df is None:
        return history + [[message, "⚠️ Please upload a CSV file first."]], gr.update(value=None)
    reply, fig = chatbot.execute_analysis(message)
    return history + [[message, reply]], fig


def clear_all():
    return [], gr.update(value=None)


with gr.Blocks(theme=gr.themes.Soft(), title="CSV Data Chatbot") as demo:
    gr.Markdown("# 📊 CSV Data Analysis Chatbot\n### 🤖 Powered by Llama-4 Scout API")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="📁 Upload CSV File", file_types=[".csv"])
            upload_btn = gr.Button("Load CSV", variant="primary")

            dataset_info = gr.Textbox(
                value="", label="📄 Dataset Summary",
                lines=15, max_lines=20, interactive=True
            )
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="Chat with your Data", height=400, value=[])
            plot_output = gr.Plot(label="Visualization", value=None)
            msg_input = gr.Textbox(
                value="", label="💬 Ask a question",
                placeholder="e.g. 'Show histogram of sales'", lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                clear_btn = gr.Button("🧹 Clear Chat")

    status_text = gr.Textbox(value="", label="Status", interactive=True)

    upload_btn.click(upload_file, inputs=[file_upload],
                     outputs=[dataset_info, status_text, chatbot_ui, plot_output])
    submit_btn.click(chat, inputs=[msg_input, chatbot_ui],
                     outputs=[chatbot_ui, plot_output]).then(lambda: "", outputs=[msg_input])
    msg_input.submit(chat, inputs=[msg_input, chatbot_ui],
                     outputs=[chatbot_ui, plot_output]).then(lambda: "", outputs=[msg_input])
    clear_btn.click(clear_all, outputs=[chatbot_ui, plot_output])


# ============================================================
# 🚀  LAUNCH (LOCAL ONLY)
# ============================================================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)
