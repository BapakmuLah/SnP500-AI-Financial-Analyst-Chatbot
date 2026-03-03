import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import plotly.express as px
import plotly.graph_objects as go

from langchain_google_genai import ChatGoogleGenerativeAI

from load_data import clean_output

## GET API KEY & DEFINE MODEL
load_dotenv()
MODEL_NAME = "models/gemma-3-27b-it"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)



VIZ_SYSTEM = """You are a data visualisation expert. Given a DataFrame description and user question, decide the best chart type.
                Respond ONLY with a JSON object: {"chart_type": "bar|horizontal_bar|scatter|line|pie|histogram|box|heatmap|treemap|bubble", 
                "x": "col", "y": "col", "color": "col", "size": "col", "title": "title", "x_label": "label", "y_label": "label"}"""


# ==== BUILD AGENT FOR VISUALIZATION =====


def _decide_chart(question: str, df: pd.DataFrame) -> dict:

    # DEFINE COLUMNS, NUMERIC COLUMNS, CATEGORICAL COLUMNS
    cols     = df.columns.tolist()
    num_cols = df.select_dtypes(include = 'number').columns.tolist()
    cat_cols = df.select_dtypes(include = ['object', 'category']).columns.tolist()

    # VISUALIZATION PROMPT
    prompt = f"""{VIZ_SYSTEM}
              User question: "{question}"
              DataFrame shape: {df.shape[0]} rows × {df.shape[1]} cols
              Columns: {cols}
              Numeric columns: {num_cols}
              Categorical columns: {cat_cols}
              Sample rows: {json.dumps(df.head(3).to_dict(orient="records"), default=str)}
              TASK : Decide the best visualisation."""
    
    # RUN LLM
    agent_response = llm.invoke(input = prompt, config = {'temperature' : 0.1, "max_output_tokens" : 512})

    # CLEAN OUTPUT
    result = clean_output(text = agent_response.content)

    return json.loads(result)


def visualization_agent(question: str, df: pd.DataFrame) -> go.Figure:
    
    # 
    plot_chosen = _decide_chart(question, df)

    chart_type = plot_chosen.get("chart_type", None)
    x, y, color, size = [plot_chosen.get(k) if plot_chosen.get(k) in df.columns else None for k in ["x", "y", "color", "size"]]
    title, x_label, y_label = plot_chosen.get("title", ""), plot_chosen.get("x_label", x or ""), plot_chosen.get("y_label", y or "")

    # INITIALIZE PLOT LAYOUT
    LAYOUT = dict(template = "plotly_white", 
                  font = dict(family = "Inter, sans-serif", size = 12), 
                  title = dict(text = title, font = dict(size = 18)), 
                  margin = dict(l = 60, r = 40, t = 60, b = 60), 
                  colorway = px.colors.qualitative.Set2)
    
    # VISUALIZE PLOT BASED ON CHART_TYPE
    try:
        if chart_type == "bar" and x and y : fig = px.bar(df.sort_values(y, ascending=False).head(25), x = x, y = y, color = color, title = title, labels = {x:x_label, y:y_label})
        elif chart_type == "horizontal_bar" and x and y : fig = px.bar(df.sort_values(y, ascending=True).tail(25), x=y, y=x, orientation="h", color=color, title=title, labels={x:x_label, y:y_label})
        elif chart_type == "scatter" and x and y: fig = px.scatter(df, x=x, y=y, color=color, size=size, hover_data=[c for c in ["Symbol","Name","Sector"] if c in df.columns], title=title, labels={x:x_label, y:y_label}, trendline="ols" if len(df)>5 else None)
        elif chart_type == "pie" and y: fig = px.pie(df.head(8), names=x, values=y, title=title)
        elif chart_type == "histogram" and x: fig = px.histogram(df, x=x, color=color, title=title, labels={x:x_label}, nbins=30)
        elif chart_type == "box" and y: fig = px.box(df, x=x, y=y, color=color, title=title, labels={x:x_label, y:y_label})
        elif chart_type == "heatmap": 
            corr = df.select_dtypes(include=[np.number]).corr().round(3)
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(), colorscale="RdBu", zmid=0, text=corr.values.round(2), texttemplate="%{text}"))
        elif chart_type == "treemap" and y: fig = px.treemap(df, path=[c for c in ["Sector","Symbol"] if c in df.columns] or [x], values=y, title=title)
        elif chart_type == "bubble" and x and y and size: fig = px.scatter(df, x=x, y=y, size=size, color=color, hover_data=[c for c in ["Symbol","Name"] if c in df.columns], title=title, labels={x:x_label, y:y_label})
        else: fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns), fill_color="#4A90D9", font=dict(color="white", size=12)), cells=dict(values=[df[c] for c in df.columns]))])

    except Exception as error:
        fig = go.Figure(data = [go.Table(header = dict(values=list(df.columns), fill_color="#4A90D9", font=dict(color="white")), cells=dict(values=[df[c].tolist() for c in df.columns]))]).update_layout(title="Results Table")

    return fig.update_layout(**LAYOUT)