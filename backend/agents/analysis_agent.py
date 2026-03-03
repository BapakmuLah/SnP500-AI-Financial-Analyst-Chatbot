
import json
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

## GET API KEY & DEFINE MODEL
load_dotenv()
MODEL_NAME = "models/gemma-3-27b-it"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


# BUILD AGENT FOR ANALYSIS WORKFLOW

ANALYSIS_SYSTEM = """You are a senior financial analyst. Given structured financial data from the S&P 500, provide sharp, insightful analysis.
                    - Highlight actionable findings
                    - Flag outliers
                    - Be concise (3-5 key points max)"""


# FUNCTION TO DESCRIBE STATISTICS DATA
def compute_stats(df: pd.DataFrame) -> dict:

    # GET ALL NUMBER COLUMNS
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty: return {}

    # STATISTIC DESCRIPTIVE
    result = {"descriptive": numeric.describe().round(3).to_dict()}

    # TOP FEATURE CORRELATION
    if len(numeric.columns) >= 2 and len(df) >= 5:

        # CALCULATE PEARSON CORRELATION FOR EACH NUMERIC COLUMN
        corr = numeric.corr().round(3)
        corr_vals = []

        # GET ALL COMBINATION PAIRS 
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if j > i:   # ---> THIS CODE IS TO REMOVE DUPLICATION
                    val = corr.loc[c1, c2]   # --> GET CORRELATION BETWEEN PAIRS

                    # IF CORRELATION IS NOT NULL
                    if not np.isnan(val):
                        corr_vals.append((abs(val), val, c1, c2))

        # SORT BY DESCENDING
        corr_vals.sort(reverse=True)
        result["top_correlations"] = [{"col1": c1, "col2": c2, "r": round(r, 3)} for _, r, c1, c2 in corr_vals[:5]]


        # OUTLIER DETECTION (INTERQUARTILE RANGE)
        outliers = {}
        for col in numeric.columns:

            num_col = numeric[col].dropna()  # DROP NULL VALUES

            # INTERQUARTILE RANGE (REMOVE OUTLIER)
            q1, q3 = num_col.quantile(0.25), num_col.quantile(0.75)
            iqr = q3 - q1
            out = df[(numeric[col] < q1 - 1.5*iqr) | (numeric[col] > q3 + 1.5*iqr)]

            # SAVE THE OUTLIER VALUE (IF ANY)
            if not out.empty and "Symbol" in df.columns:
                outliers[col] = out["Symbol"].tolist()[:5]
        result["outliers"] = outliers

        # AGGREGATION STATISTICS PER SECTOR
        if "Sector" in df.columns and len(numeric.columns) > 0:
            first_metric = numeric.columns[0]
            result["sector_summary"] = df.groupby("Sector")[first_metric].agg(["mean", "count"]).round(2).to_dict(orient="index")
            
        return result
    

def analysis_agent(question : str, df : pd.DataFrame, sql_explanation: str = "") -> str:

    if df.empty: return "No data returned — cannot perform analysis."

    # CALCULATE STATISTICS DATA
    stats = compute_stats(df)
    preview_rows = min(30, len(df))

    # DISPLAY DATAFRAME
    try             : df_preview = df.head(preview_rows).to_markdown(index=False)
    except Exception: df_preview = df.head(preview_rows).to_string(index=False)

    ANALYSIS_PROMPT = f"""{ANALYSIS_SYSTEM} 
                       User question: "{question}"
                       Data context: {sql_explanation}
                       Rows returned: {len(df)}
                       DATA PREVIEW (first {preview_rows} rows):  {df_preview}
                       COMPUTED STATISTICS: {json.dumps(stats, indent=2, default=str)}
                       Provide concise financial analysis of this data."""
    
    # RUN LLM
    agent_response = llm.invoke(input = ANALYSIS_PROMPT, config = {'temperature' : 0.3, "max_output_tokens" : 2000})
    return agent_response.text