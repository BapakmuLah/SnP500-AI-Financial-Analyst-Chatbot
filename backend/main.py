import os
import re
import json
import time
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from google import genai
from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.sql_database.query import create_sql_query_chain


# IMPORT PROJECT FOLDER
import schema
import load_data
from load_data import clean_output, clean_data, custom_table_description
from agents.sql_agent import generate_sql
from agents.analysis_agent import analysis_agent
from agents.visualization_agent import visualization_agent
from agents.evaluation import Evaluator

# 1. SETUP & CONFIGURATION BACKEND

## DATASET PATH 
os.makedirs("data", exist_ok=True)
DATABASE_NAME = "financial.db"
TABLE_NAME = "financials_data"
DB_PATH = Path(f"data/{DATABASE_NAME}")
CSV_PATH = Path("data/financials.csv")
META_PATH = Path("data/db_metadata.json")

## DEFINE FAST API
app = FastAPI(title="S&P 500 Financial Analyst API")
app.add_middleware(CORSMiddleware, 
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


## GET API KEY & DEFINE MODEL
load_dotenv()
MODEL_NAME = "models/gemma-3-27b-it"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


# 2. BUILD AGENT ORCHESTRATION

## BUILD PROMPT FOR ORCHESTRATION
INTENT_SYSTEM = """You are an intent classifier for a financial data chatbot. Classify the user query into ONE of these intents: 
                - DATA_LOOKUP, RANKING, COMPARISON, AGGREGATION, CORRELATION, DISTRIBUTION, FULL_ANALYSIS, VISUALIZATION, GREETING, OUT_OF_SCOPE
                Respond ONLY with JSON:{{"intent": "<INTENT>", 
                                         "needs_sql": true/false, 
                                         "needs_analysis": true/false, 
                                         "needs_viz": true/false, 
                                         "entities": ["names"], 
                                         "metrics": ["metrics"], 
                                         "reasoning": "reason"}}

                QUESTION : {user_query}
                """
SYNTHESIS_SYSTEM = """You are a helpful financial data assistant for S&P 500 analysis. You are synthesising results from multiple specialised agents into a clear, helpful response.
                    - Lead with the direct answer to the user's question
                    - Reference specific numbers from the data when available
                    - Use markdown formatting (bold, tables) for readability
                    - Do NOT repeat raw SQL unless user asks
                    - if the question is out_of_scope, reject it politely and do not answer the question"""


## CREATE ORCHESTRATION
class Orchestrator:
    def __init__(self):
        self.schema_context = schema.schema_data

    # INTENT CLASSIFICATION
    def classify_intent(self, question: str) -> dict:

        # RUN LLM
        INTENT_PROMPT = INTENT_SYSTEM.format(user_query = question)
        intent_response = llm.invoke(input = INTENT_PROMPT, config={"temperature": 0.1, "max_output_tokens": 256})
        
        # CLEAN OUTPUT (REMOVE MARKDOWN)
        intent_result = clean_output(intent_response.content)

        # RETURN AS JSON FILE
        return json.loads(intent_result)

    
    # UNTUK MENGHASILKAN RESPONSE JAWABAN YG SESUAI DGN INPUT USER
    def synthesise(self, question: str, intent: dict, sql_res: dict, analysis: str, has_viz: bool) -> str:

        # GET LABEL
        intent_label = intent.get("intent", "")

        # DISPLAY OUTPUT BASED ON INTENT LABEL
        #if intent_label == "GREETING": return "Hello! 👋 I'm your S&P 500 Financial Analyst powered by Gemini."
        #if intent_label == "OUT_OF_SCOPE": return "I'm specialised in S&P 500 financial data. Ask me about stock prices, P/E ratios, etc."
        #if sql_res and sql_res.get("error"): return f"⚠️ Couldn't retrieve data: `{sql_res['error']}`"

        # GET SQL QUERY RESULT
        df        = sql_res.get("dataframe", pd.DataFrame()) if sql_res else pd.DataFrame()
        sql_query = sql_res.get("sql", "") if sql_res else ""
        sql_expl  = sql_res.get("explanation", "") if sql_res else ""

        # TABLE THAT WILL BE SHOWN IN OUTPUT (IF AVAILABLE)
        data_summary = ""
        if not df.empty:
            try: data_summary = df.head(20).to_markdown(index=False)
            except Exception: data_summary = df.head(20).to_string(index=False)

        #
        context_parts = [f'User question: "{question}"', f"Intent: {intent_label}", f"SQL: {sql_query}", f"What it fetched: {sql_expl}", f"Rows returned: {len(df)}"]
        if data_summary: context_parts.append(f"\nDATA:\n{data_summary}")
        if analysis: context_parts.append(f"\nANALYSIS:\n{analysis}")
        if has_viz: context_parts.append("\nA chart has been generated.")

        # CREATE FINAL PROMPT TO ANSWER AND GIVE RESPONSE FEEDBACK TO USER
        context_parts.append("\nSynthesise a clear, helpful response to the user's question.")
        FINAL_PROMPT = f"{SYNTHESIS_SYSTEM}\n" + "\n".join(context_parts)

        # MODEL RESPONSE TO USER INPUT
        model_response = llm.invoke(input = FINAL_PROMPT, config={"temperature": 0.4, "max_output_tokens": 2000})
        
        return model_response.text


    
    def run(self, question: str, show_token_usage : bool = None) -> dict:
        
        # 1. CLASSIFY USER_INPUT TO DETERMINE USER INTENT USING LLM
        intent = self.classify_intent(question)
        intent_label = intent.get("intent", "DATA_LOOKUP")    # --> LABEL INTENT

        sql_result, analysis, figure = None, None, None

        # IF USER INPUT NEED QUERY SQL
        if intent.get("needs_sql"):
            sql_result = generate_sql(question)

        # DATAFRAME FROM EXECUTED SQL
        df = sql_result.get("dataframe", pd.DataFrame()) if sql_result else pd.DataFrame()

        # IF USER INPUT NEED ANALYSIS
        if not df.empty and (intent.get("needs_analysis") or intent_label == "FULL_ANALYSIS"):
            analysis = analysis_agent(question, df, sql_result.get("explanation",""))

        # IF USER INPUT NEED VISUALIZATION
        if not df.empty and (intent.get("needs_viz") or intent_label in ("DISTRIBUTION","CORRELATION","VISUALIZATION","RANKING","COMPARISON","AGGREGATION")):
            figure = visualization_agent(question, df)

        response = self.synthesise(question = question, 
                                   intent = intent, 
                                   sql_res = sql_result, 
                                   analysis = analysis or "", 
                                   has_viz = figure is not None)
        

        return {"response" : response, 
                "figure"   : figure, 
                "sql"      : sql_result.get("sql","") if sql_result else "", 
                "dataframe": df, 
                "intent"   : intent['intent']}
    

## DEFINE ORCHESTRATOR
analyst = Orchestrator()


# 3. ============ CREATE ENDPOINT API =================

## HEALTH CHECKING
@app.get("/")
def read_root():
    return {"message": "Full Documentation API here : https:"}

## CREATE DATABASE ENDPONT
@app.get(path = "/create-database")
def create_database():
    
    # SAVE DATABASE
    load_data.save_data(CSV_PATH = CSV_PATH, DB_PATH = DB_PATH, TABLE_NAME = TABLE_NAME)

    return {"info" : f"Database saved on {DB_PATH}"}

## CHATBOT ENDPOINT
@app.post("/api/chat", response_model= schema.ChatResponse)
async def chat_endpoint(request: schema.ChatRequest):
    try:

        # RUN ORCHESTRATOR PIPELINE
        result = analyst.run(question = request.question)
        
        # SERIALIZE PLOTLY FIGURE INTO JSON (CAUSE FASTAPI CAN'T SHOW PLOTLY CHART)
        fig_dict = None
        if result.get("figure"):
            # to_json() outputs a JSON string, we load it back to a python Dict for FastAPI to return as JSON
            fig_dict = json.loads(result["figure"].to_json())

        return schema.ChatResponse(
            intent = result["intent"],
            response = result["response"],
            sql = result["sql"],
            figure = fig_dict
        )
    
    # IF CHATBOT FAIL
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



## OVERVIEW ENDPOINT 
@app.get("/api/overview", response_model = schema.OverviewResponse)
async def get_dataset_overview():

    try:
        # IF DATASET IS EMPTY
        if load_data.preview_df.empty:
            raise ValueError("Dataset is empty or not loaded properly.")

        # 1. COUNT NUMBER OF COMPANY AND SECTOR
        total_companies = len(load_data.preview_df)
        total_sectors = load_data.preview_df['Sector'].nunique()

        # 2. GET THE HIGHEST MARKET CAP COMPANY
        top_idx = load_data.preview_df['Market_Cap'].idxmax()
        top_company = load_data.preview_df.loc[top_idx, 'Name']
        top_mc_billions = load_data.preview_df.loc[top_idx, 'Market_Cap'] / 1e9

        # 3. AVERAGE P/E RATIO
        avg_pe = load_data.preview_df['Price_Earnings'].mean()

        # 4. FIND SECTOR DOMINANT 
        largest_sector = load_data.preview_df.groupby('Sector')['Market_Cap'].sum().idxmax()

        return schema.OverviewResponse(total_companies=total_companies,
                                        total_sectors=total_sectors,
                                        top_company=top_company,
                                        top_market_cap_b=round(top_mc_billions, 2),
                                        avg_pe_ratio=round(avg_pe, 2),
                                        largest_sector=largest_sector)
        
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Failed to fetch overview: {str(e)}")



## EVALUATION ENDPOINT
@app.post("/api/evaluate")
async def trigger_evaluation(req: schema.EvalRequest):
    try:
        evaluator = Evaluator(iteration_name=req.iteration_name)
        df_result = evaluator.run_full_eval(analyst)
        
        # Pydantic requires standard python dicts/lists, so we convert the DataFrame
        results_list = df_result.to_dict(orient="records")
        mean_score = df_result["overall"].mean()
        
        return {
            "message": f"Evaluation '{req.iteration_name}' completed.",
            "mean_overall_score": mean_score,
            "details": results_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluate/history")
async def get_evaluation_history():
    try:
        df_history = Evaluator.compare_iterations()
        return {"history": df_history.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
