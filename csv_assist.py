import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json, re, requests, traceback
from typing import Optional, Tuple


# ============================================================
# ⚙️  API CONFIG
# ============================================================
API_ENDPOINT = "https://your-domain.com/llama-4-scout-17B-16E-Instruct/v1/chat/completions"
API_HEADERS = {"Authorization": "Basic cccccccccc", "Content-Type": "application/json"}

# ✅ Always provide a valid empty figure to avoid bool/None schema issues
EMPTY_FIG = go.Figure()
EMPTY_FIG.update_layout(template="plotly_white")


# ============================================================
# 🧠  CHATBOT CORE
# ============================================================
class CSVChatbot:
    def __init__(self):
        self.df = None
        self.df_info = None
        self.conversation_history = []

    def load_csv(self, file) -> Tuple[str, str]:
        try:
            path = file.name if hasattr(file, "name") else file
            self.df = pd.read_csv(path)

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
            traceback.print_exc()
            return f"❌ Error loading CSV: {e}", "Error loading file"

    def is_safe_query(self, query: str) -> Tuple[bool, str]:
        patterns = [
            r'import\s+os', r'subprocess', r'eval\s*\(', r'exec\s*\(',
            r'socket', r'open\s*\(', r'pickle', r'system', r';', r'&&', r'`'
        ]
        for p in patterns:
            if re.search(p, query, re.I):
                return False, "⚠️ Potentially unsafe operation detected."
        return True, "Safe"

    def generate_llm_response(self, query: str) -> str:
        if self.df is None:
            return "Please upload a CSV first."

        try:
            head = self.df.head(5).to_string()
            dtypes = json.dumps(self.df.dtypes.astype(str).to_dict(), indent=2)
            context = (
                f"You are a helpful data analyst. Dataset has shape {self.df.shape}.\n\n"
                f"Columns: {', '.join(self.df.columns)}\n\n"
                f"Sample:\n{head}\n\nData types:\n{dtypes}\n\n"
                f"User question: {query}"
            )

            payload = {
                "messages": [
                    {"role": "system", "content": "You analyze CSV data and explain insights clearly."},
                    {"role": "user", "content": context}
                ],
                "max_tokens": 800,
                "temperature": 0.7,
            }

            r = requests.post(API_ENDPOINT, headers=API_HEADERS, json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]["content"]
                self.conversation_history.append({"user": query, "assistant": msg})
                return msg
            else:
                return f"API Error {r.status_code}: {r.text}"

        except Exception as e:
            traceback.print_exc()
            return f"Error contacting API: {str(e)}"

    def execute_analysis(self, query: str) -> Tuple[str, go.Figure]:
        if self.df is None:
            return "Please upload a CSV first.", EMPTY_FIG

        safe, msg = self.is_safe_query(query)
        if not safe:
            return msg, EMPTY_FIG

        answer = self.generate_llm_response(query)
        viz_words = ['plot','chart','graph','visual','hist','scatter','bar','line',
                     'box','pie','heatmap','correlation','3d','violin','sunburst']
        fig = self.auto_plot(query) if any(w in query.lower() for w in viz_words) else EMPTY_FIG
        return answer, fig

    def auto_plot(self, query: str) -> go.Figure:
        try:
            q = query.lower()
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            cat_cols = self.df.select_dtypes(include=['object']).columns
            fig = EMPTY_FIG

            if 'hist' in q and len(num_cols):
                fig = px.histogram(self.df, x=num_cols[0], nbins=30)
            elif 'scatter' in q and len(num_cols) >= 2:
                fig = px.scatter(self.df, x=num_cols[0], y=num_cols[1])
            elif ('bar' in q or 'count' in q) and len(cat_cols):
                vc = self.df[cat_cols[0]].value_counts().head(10)
                fig = px.bar(x=vc.index, y=vc.values)
            elif 'line' in q and len(num_cols):
                fig = px.line(self.df, y=num_cols[0])
            elif ('correlation' in q or 'heatmap' in q) and len(num_cols) > 1:
                corr = self.df[num_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r")
            elif 'pie' in q and len(cat_cols):
                vc = self.df[cat_cols[0]].value_counts().head(10)
                fig = px.pie(values=vc.values, names=vc.index)
            elif '3d' in q and len(num_cols) >= 3:
                fig = px.scatter_3d(self.df, x=num_cols[0], y=num_cols[1], z=num_cols[2])

            fig.update_layout(template="plotly_white", height=480)
            return fig
        except Exception as e:
            print("Plot error:", e)
            traceback.print_exc()
            return EMPTY_FIG


# ============================================================
# 🖥️  GRADIO UI + CALLBACKS
# ============================================================
chatbot = CSVChatbot()

def upload_file(file):
    if file is None:
        return "Please upload a file", "No file uploaded", [], gr.update(value=EMPTY_FIG)
    summary, status = chatbot.load_csv(file)
    return str(summary), str(status), [], gr.update(value=EMPTY_FIG)

def chat(message, history):
    try:
        if chatbot.df is None:
            return history + [[message, "⚠️ Please upload a CSV file first."]], gr.update(value=EMPTY_FIG)
        reply, fig = chatbot.execute_analysis(message)
        return history + [[message, reply]], fig
    except Exception as e:
        traceback.print_exc()
        return history + [[message, f"❌ Error: {str(e)}"]], gr.update(value=EMPTY_FIG)

def clear_all():
    return [], gr.update(value=EMPTY_FIG)


# ============================================================
# 🌈  BUILD UI
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(), title="CSV Data Chatbot") as demo:
    gr.Markdown("# 📊 CSV Data Analysis Chatbot\n### 🤖 Powered by Llama-4 Scout API")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="📁 Upload CSV", file_types=[".csv"])
            upload_btn = gr.Button("Load CSV", variant="primary")
            dataset_info = gr.Textbox(value="", label="📄 Dataset Summary",
                                      lines=15, max_lines=20, interactive=True)
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="Chat with your Data", height=400, value=[])
            plot_output = gr.Plot(label="Visualization", value=EMPTY_FIG)
            msg_input = gr.Textbox(value="", label="💬 Ask a question",
                                   placeholder="e.g. 'Show histogram of sales'", lines=2)
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                clear_btn = gr.Button("🧹 Clear")

    status_text = gr.Textbox(value="", label="Status", interactive=True)

    upload_btn.click(upload_file, inputs=[file_upload],
                     outputs=[dataset_info, status_text, chatbot_ui, plot_output])
    submit_btn.click(chat, inputs=[msg_input, chatbot_ui],
                     outputs=[chatbot_ui, plot_output]).then(lambda: "", outputs=[msg_input])
    msg_input.submit(chat, inputs=[msg_input, chatbot_ui],
                     outputs=[chatbot_ui, plot_output]).then(lambda: "", outputs=[msg_input])
    clear_btn.click(clear_all, outputs=[chatbot_ui, plot_output])


# ============================================================
# 🚀  LOCAL-ONLY LAUNCH
# ============================================================
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)
