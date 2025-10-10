import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from huggingface_hub import InferenceClient
import json
import re
from typing import Optional, Tuple
import numpy as np

# Initialize Hugging Face Inference Client
# You'll need to set your HF token as an environment variable or pass it directly
client = InferenceClient(token="YOUR_HF_TOKEN_HERE")

class CSVChatbot:
    def __init__(self):
        self.df = None
        self.df_info = None
        self.conversation_history = []
        
    def load_csv(self, file) -> Tuple[str, str]:
        """Load and analyze CSV file"""
        try:
            self.df = pd.read_csv(file.name)
            
            # Generate dataset summary
            self.df_info = {
                'columns': list(self.df.columns),
                'shape': self.df.shape,
                'dtypes': self.df.dtypes.to_dict(),
                'head': self.df.head().to_string(),
                'describe': self.df.describe().to_string(),
                'missing': self.df.isnull().sum().to_dict()
            }
            
            summary = f"""✅ CSV File Loaded Successfully!

**Dataset Overview:**
- Rows: {self.df.shape[0]}
- Columns: {self.df.shape[1]}
- Column Names: {', '.join(self.df.columns)}

**Data Types:**
{chr(10).join([f"- {col}: {dtype}" for col, dtype in self.df.dtypes.items()])}

**Missing Values:**
{chr(10).join([f"- {col}: {count}" for col, count in self.df.isnull().sum().items() if count > 0]) or "No missing values"}

**First Few Rows:**
```
{self.df.head().to_string()}
```

You can now ask questions about the data or request visualizations!
"""
            return summary, "CSV loaded successfully! Ask me anything about your data."
            
        except Exception as e:
            return f"❌ Error loading CSV: {str(e)}", "Error loading file"
    
    def is_safe_query(self, query: str) -> Tuple[bool, str]:
        """Check if the query is safe and doesn't contain malicious code"""
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+sys',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'subprocess',
            r'socket',
            r'requests',
            r'urllib',
            r'pickle',
            r'shelve',
            r'\.system\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Security alert: Query contains potentially unsafe operations"
        
        return True, "Safe"
    
    def generate_llm_response(self, user_query: str) -> str:
        """Generate response using Llama 4 via HuggingFace Inference API"""
        
        if self.df is None:
            return "Please upload a CSV file first before asking questions."
        
        # Create context about the dataset
        context = f"""You are a helpful data analyst assistant. You have access to a CSV dataset with the following information:

Dataset Shape: {self.df_info['shape']}
Columns: {', '.join(self.df_info['columns'])}
Data Types: {json.dumps(self.df_info['dtypes'], indent=2)}

First few rows:
{self.df_info['head']}

Statistical Summary:
{self.df_info['describe']}

User Query: {user_query}

Guidelines:
1. Provide clear, concise answers about the data
2. If the user asks for visualization, suggest appropriate chart types
3. For data analysis questions, be specific and reference actual data
4. If you need to suggest code, use pandas and matplotlib/seaborn
5. Always explain your reasoning
6. If the query is unclear, ask for clarification

Respond to the user's query in a helpful and informative way."""

        try:
            # Call Llama 4 via HuggingFace Inference API
            response = ""
            for message in client.chat_completion(
                model="meta-llama/Llama-3.3-70B-Instruct",  # Using Llama 3.3 as Llama 4 may not be released yet
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant specialized in CSV data analysis."},
                    {"role": "user", "content": context}
                ],
                max_tokens=1000,
                stream=True,
            ):
                response += message.choices[0].delta.content or ""
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}\n\nPlease check your HuggingFace token and API access."
    
    def execute_analysis(self, query: str) -> Tuple[str, Optional[go.Figure]]:
        """Execute data analysis based on user query"""
        
        if self.df is None:
            return "Please upload a CSV file first.", None
        
        # Check for safety
        is_safe, message = self.is_safe_query(query)
        if not is_safe:
            return message, None
        
        try:
            # Get LLM response
            llm_response = self.generate_llm_response(query)
            
            # Check if user wants visualization
            viz_keywords = ['plot', 'chart', 'graph', 'visualiz', 'show', 'display', 'histogram', 
                           'scatter', 'bar', 'line', 'box', 'pie', 'heatmap', 'correlation', '3d', 
                           'violin', 'sunburst']
            if any(keyword in query.lower() for keyword in viz_keywords):
                # Try to generate appropriate visualization
                plot_fig = self.auto_generate_plot(query)
                return llm_response, plot_fig
            
            return llm_response, None
            
        except Exception as e:
            return f"Error during analysis: {str(e)}", None
    
    def auto_generate_plot(self, query: str) -> Optional[go.Figure]:
        """Automatically generate interactive Plotly plot based on query"""
        
        try:
            # Detect plot type from query
            query_lower = query.lower()
            fig = None
            
            if 'histogram' in query_lower or 'distribution' in query_lower:
                # Interactive histogram of first numeric column
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.histogram(
                        self.df, 
                        x=numeric_cols[0],
                        nbins=30,
                        title=f'Distribution of {numeric_cols[0]}',
                        labels={numeric_cols[0]: numeric_cols[0]},
                        color_discrete_sequence=['#636EFA']
                    )
                    fig.update_layout(
                        showlegend=False,
                        hovermode='x unified'
                    )
            
            elif 'scatter' in query_lower:
                # Interactive scatter plot of first two numeric columns
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    # Try to find a categorical column for color coding
                    cat_cols = self.df.select_dtypes(include=['object']).columns
                    color_col = cat_cols[0] if len(cat_cols) > 0 else None
                    
                    fig = px.scatter(
                        self.df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        color=color_col,
                        title=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                        hover_data=self.df.columns[:5].tolist(),  # Show first 5 columns on hover
                        opacity=0.7
                    )
                    fig.update_traces(marker=dict(size=8))
            
            elif 'bar' in query_lower or 'count' in query_lower:
                # Interactive bar plot of first categorical column
                cat_cols = self.df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    value_counts = self.df[cat_cols[0]].value_counts().head(10)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f'Count of {cat_cols[0]}',
                        labels={'x': cat_cols[0], 'y': 'Count'},
                        color=value_counts.values,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(showlegend=False)
            
            elif 'line' in query_lower or 'trend' in query_lower:
                # Interactive line plot of first numeric column
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.line(
                        self.df,
                        y=numeric_cols[0],
                        title=f'Trend of {numeric_cols[0]}',
                        labels={'index': 'Index', numeric_cols[0]: numeric_cols[0]}
                    )
                    fig.update_traces(line_color='#636EFA', line_width=2)
            
            elif 'box' in query_lower or 'boxplot' in query_lower:
                # Interactive box plot of numeric columns
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = go.Figure()
                    for col in numeric_cols[:5]:  # Limit to 5 columns for readability
                        fig.add_trace(go.Box(y=self.df[col], name=col))
                    fig.update_layout(
                        title='Box Plot of Numeric Columns',
                        yaxis_title='Value',
                        showlegend=True
                    )
            
            elif 'correlation' in query_lower or 'heatmap' in query_lower:
                # Interactive correlation heatmap
                numeric_df = self.df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        aspect='auto',
                        color_continuous_scale='RdBu_r',
                        title='Correlation Heatmap',
                        labels=dict(color="Correlation")
                    )
                    fig.update_layout(
                        width=700,
                        height=600
                    )
            
            elif 'pie' in query_lower:
                # Pie chart of first categorical column
                cat_cols = self.df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    value_counts = self.df[cat_cols[0]].value_counts().head(10)
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f'Distribution of {cat_cols[0]}',
                        hole=0.3  # Makes it a donut chart
                    )
            
            elif '3d' in query_lower or 'three dimensional' in query_lower:
                # 3D scatter plot
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 3:
                    fig = px.scatter_3d(
                        self.df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        z=numeric_cols[2],
                        title=f'3D Scatter: {numeric_cols[0]}, {numeric_cols[1]}, {numeric_cols[2]}',
                        opacity=0.7
                    )
            
            elif 'violin' in query_lower:
                # Violin plot
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = go.Figure()
                    for col in numeric_cols[:5]:
                        fig.add_trace(go.Violin(y=self.df[col], name=col, box_visible=True))
                    fig.update_layout(
                        title='Violin Plot of Numeric Columns',
                        yaxis_title='Value'
                    )
            
            elif 'sunburst' in query_lower:
                # Sunburst chart for hierarchical data
                cat_cols = self.df.select_dtypes(include=['object']).columns
                if len(cat_cols) >= 2:
                    fig = px.sunburst(
                        self.df,
                        path=[cat_cols[0], cat_cols[1]] if len(cat_cols) >= 2 else [cat_cols[0]],
                        title='Hierarchical Sunburst Chart'
                    )
            
            else:
                # Default: interactive line plot of first numeric column
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.line(
                        self.df,
                        y=numeric_cols[0],
                        title=f'{numeric_cols[0]} Overview',
                        labels={'index': 'Index', numeric_cols[0]: numeric_cols[0]}
                    )
            
            if fig:
                # Apply consistent styling
                fig.update_layout(
                    template='plotly_white',
                    hovermode='closest',
                    height=500,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
            
            return fig
            
        except Exception as e:
            print(f"Error generating plot: {str(e)}")
            return None

# Initialize chatbot
chatbot = CSVChatbot()

# Gradio Interface
def upload_file(file):
    if file is None:
        return "Please upload a file", "No file uploaded", None, None
    summary, status = chatbot.load_csv(file)
    return summary, status, None, None

def chat(message, history):
    if chatbot.df is None:
        return history + [[message, "⚠️ Please upload a CSV file first before asking questions."]], None
    
    response, plot_fig = chatbot.execute_analysis(message)
    
    return history + [[message, response]], plot_fig

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="CSV Data Chatbot") as demo:
    gr.Markdown("""
    # 📊 CSV Data Analysis Chatbot
    ### Powered by Llama 4 & Hugging Face
    
    Upload your CSV file and interact with your data using natural language!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                type="filepath"
            )
            upload_btn = gr.Button("📁 Load CSV", variant="primary")
            
            gr.Markdown("### Dataset Info")
            dataset_info = gr.Textbox(
                label="Dataset Summary",
                lines=15,
                max_lines=20,
                interactive=False
            )
        
        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(
                label="Chat with your Data",
                height=400,
                type="messages"
            )
            
            plot_output = gr.Plot(
                label="Visualization",
                visible=True
            )
            
            msg_input = gr.Textbox(
                label="Ask a question about your data",
                placeholder="e.g., 'What are the top 5 values in column X?' or 'Show me a histogram'",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat")
    
    status_text = gr.Textbox(label="Status", interactive=False)
    
    gr.Markdown("""
    ### 💡 Example Questions:
    - "Show me the first 10 rows"
    - "What are the column names and data types?"
    - "Create a histogram of [column name]"
    - "Show correlation heatmap"
    - "What's the average value of [column]?"
    - "Plot a scatter chart of [col1] vs [col2]"
    - "Show me a 3D scatter plot"
    - "Create a violin plot"
    - "Display a pie chart"
    - "Generate a sunburst chart"
    """)
    
    # Event handlers
    upload_btn.click(
        fn=upload_file,
        inputs=[file_upload],
        outputs=[dataset_info, status_text, chatbot_ui, plot_output]
    )
    
    submit_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot_ui],
        outputs=[chatbot_ui, plot_output]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot_ui],
        outputs=[chatbot_ui, plot_output]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    clear_btn.click(
        lambda: (None, None),
        outputs=[chatbot_ui, plot_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
