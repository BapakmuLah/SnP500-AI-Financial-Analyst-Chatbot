import os
import json
import sqlite3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.sql_database.query import create_sql_query_chain

# IMPORT NECESSARY PROJECT FOLDER
from load_data import db, clean_output
from schema import schema_data

## DATASET PATH 
os.makedirs("data", exist_ok=True)
DATABASE_NAME = "financial.db"
TABLE_NAME = "financials_data"
DB_PATH = Path(f"data/{DATABASE_NAME}")
CSV_PATH = Path("data/financials.csv")
META_PATH = Path("data/db_metadata.json")


## GET API KEY & DEFINE MODEL
load_dotenv()
MODEL_NAME = "models/gemma-3-27b-it"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)


# =========== BUILD SQL AGENT WORKFLOW  ======================
# SQL AGENT PROMPT
SQL_SYSTEM_PROMPT = PromptTemplate(template = """You are an expert SQLite analyst for the S&P 500 financial database. 
                                                Table name: financials_data
                                                Your ONLY job is to convert the user question into a single, valid SQLite SELECT statement.
                                                Database schema: {table_info}
                                                User question: {input}
                                                Table Schema: {schema}
                                                
                                                OUTPUT FORMAT — respond with ONLY a JSON object, 
                                                nothing else:{{"sql": "<your SQL query>",
                                                                "explanation": "<one sentence: what the query does>",
                                                                "columns_used": ["col1", "col2"]}}
                                                                RULES: - Output valid SQLite syntax ONLY. No markdown, no commentary outside the JSON.
                                                                        - NEVER use DROP, INSERT, UPDATE, DELETE, CREATE.
                                                                        - Default LIMIT is {top_k} unless user says "all" or you're doing aggregation.
                                                                        - For "Top N per category" questions, YOU MUST use a CTE with ROW_NUMBER() OVER(PARTITION BY category ORDER BY value DESC). Do NOT just use a global LIMIT.
                                                                        - For ranking queries use ORDER BY + LIMIT.""", 
                                   input_variables=["input", "top_k", "table_info"], 
                                   partial_variables = {'schema' : schema_data})

# CREATE SQL AGENT
query_chain = create_sql_query_chain(llm = llm, db = db, prompt = SQL_SYSTEM_PROMPT)



def generate_sql(question: str) -> dict:
    try:
        # RUN LLM TO GENERATE SQL QUERY & CONVERT IT TO JSON FILE
        sql_response = query_chain.invoke(input={"question": question}, config={'temperature': 0, 'max_output_tokens': 1024})
        sql_result = json.loads(clean_output(sql_response))

        # EXECUTE SQL QUERY TO DATAFRAME
        sql_conn = sqlite3.connect(database=DB_PATH)
        df_res = pd.read_sql_query(sql=sql_result['sql'], con=sql_conn)

        sql_conn.close()

        return {"sql": sql_result['sql'], "explanation": sql_result.get("explanation", ""), "dataframe": df_res, "error": None}
    
    except Exception as e:
        return {"sql": "", "explanation": "", "dataframe": pd.DataFrame(), "error": str(e)}
    

