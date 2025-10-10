import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests, json, re, numpy as np, socket
from typing import Optional, Tuple

# ============================================================
# 🧠 API Configuration
# ============================================================
API_ENDPOINT = "https://your-domain.com/llama-4-scout-17B-16E-Instruct/v1/chat/completions"
API_HEADERS = {"Authorization": "Basic cccccccccc", "Content-Type": "application/json"}

# ============================================================
# 🧩 CSV Chatbot Class
# ============================================================
class CSVChatbot:
    def __init__(self):
        self.df = None
        self.df_info = None

    # --------------------------------------------------------
    def load_csv(self, file) -> Tuple[str, str]:
        """Load and summarize CSV"""
        try:
            self.df = pd.read_csv(file)
            self.df_info = {
                "columns": list(self.df.columns),
                "shape": self.df.shape,
                "dtypes": self.df.dtypes.to_dict(),
                "head": self.df.head().to_string(),
                "describe": self.df.describe().to_string(),
                "missing": self.df.isnull().sum().to_dict()
            }
            summary = f"""✅ CSV File Loaded Successfully!

**Dataset Overview:**
- Rows: {self.df.shape[0]}
- Columns: {self.df.shape[1]}
- Columns: {', '.join(self.df.columns)}

**Data Types:**
{chr(10).join([f"- {c}: {t}" for c,t in self.df.dtypes.items()])}

**Missing Values:**
{chr(10).join([f"- {c}: {v}" for c,v in self.df.isnull().sum().items() if v>0]) or "No missing values"}

**Preview:**
