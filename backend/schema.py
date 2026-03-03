import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional, Dict, Any

import load_data

# DATASET PATH
os.makedirs("data", exist_ok=True)
DATABASE_NAME = "financial.db"
TABLE_NAME = "financials_data"

DB_PATH = Path(f"data/{DATABASE_NAME}")
CSV_PATH = Path("data/financials.csv") 
META_PATH = Path("data/db_metadata.json")


# ============= CREATE PYDANTIC SCHEMA ===============

# SCHEMA FOR USER INPUT
class ChatRequest(BaseModel):
    question: str

# SCHEMA FOR OUTPUT MODEL RESPONSE
class ChatResponse(BaseModel):
    intent: str
    response: str
    sql: str
    figure: Optional[Dict[str, Any]] = None  # Dict to handle JSON serializable Plotly figure

# SCHEMA FOR DATASET OVERVIEW
class OverviewResponse(BaseModel):
    total_companies: int
    total_sectors: int
    top_company: str
    top_market_cap_b: float
    avg_pe_ratio: float
    largest_sector: str

# SCHEMA FOR EVALUATION ENDPOINT
class EvalRequest(BaseModel):
    iteration_name: str = "baseline"





# ============= CREATE DATASET SCHEMA =======================

# COLUMN DATASET
COLUMN_DOCS = {"Symbol":          "NYSE/NASDAQ ticker symbol (e.g. AAPL, MSFT). Primary key.",
               "Name":            "Full company name.",
               "Sector":          "GICS sector (e.g. Information Technology, Health Care, Financials).",
                "Price":           "Current stock price in USD.",
                "Price/Earnings":  "P/E ratio – stock price divided by earnings per share. High P/E can indicate growth expectations or overvaluation.",
                "Dividend Yield":  "Annual dividend as a percentage of stock price. Higher = more income-generating.",
                "Earnings/Share":  "EPS – net profit divided by number of shares. Profitability per share.",
                "52 Week High":    "Highest stock price in the past 52 weeks.",
                "52 Week Low":     "Lowest stock price in the past 52 weeks.",
                "Market Cap":      "Total market capitalisation in USD (shares × price). Size indicator.",
                "EBITDA":          "Earnings Before Interest, Taxes, Depreciation & Amortisation in USD. Core operational profitability.",
                "Price/Sales":     "Market cap divided by annual revenue. Useful for companies with no earnings yet.",
                "Price/Book":      "Stock price divided by book value per share. Values < 1 may indicate undervaluation.",
                "SEC Filings":     "URL to the company's SEC EDGAR filings page."}

# FUNCTION TO BUILD A STRUCTURED SUMMARY OF A DATASET
def build_metadata(df: pd.DataFrame) -> dict:

    # GET ALL NUMERIC COLUMNS
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # INFORMATION STATISTICS ALL NUMERRIC COLUMNS
    stats = {}
    for col in numeric_cols:
        s = df[col].dropna()
        stats[col] = {"min": round(float(s.min()), 4),
                      "max": round(float(s.max()), 4),
                      "mean": round(float(s.mean()), 4),
                      "median": round(float(s.median()), 4),
                      "null_count": int(df[col].isna().sum())}
        
    # INFORMATION ABOUT DATASET
    meta = {"table_name"  : "sp500",
            "row_count"   : len(df),
            "description" : f"S&P 500 companies financial snapshot. Contains data for {len(df)} companies across {df.get('Sector').nunique()} GICS sectors.",
            "columns"     : {col: {"description": COLUMN_DOCS.get(col, ""), "dtype": str(df[col].dtype), **({"stats": stats[col]} if col in stats else {})} for col in df.columns},
            "sector_distribution": df.get("Sector").value_counts().to_dict() if "Sector" in df.columns else {},
            "sample_rows" : df.head(3).to_dict(orient="records")}
    
    # SAVE METADATA
    with open(META_PATH, "w") as file:
        json.dump(meta, file, indent = 2)

    print(f"[META] Metadata written → {META_PATH}")

    return meta


# BUILD METADATA
meta = build_metadata(load_data.df)


# CREATE SCHEMA DATASET FROM METADATA
col_lines = [f"  - {col} ({info.get('dtype', '')}): {info.get('description', '')}" for col, info in meta["columns"].items()]  # --> COLUMN DESCRIPTION
sectors_str = ', '.join(meta.get('sector_distribution', {}).keys())

# CREATE SCHEMA DATASET
schema_data = f"""DATABASE: SQLite table `{meta['table_name']}` 
               ROWS: {meta['row_count']} S&P 500 companies
               DESCRIPTION: {meta['description']}
               COLUMNS: {chr(10).join(col_lines)}
               SECTORS present: {sectors_str}
               IMPORTANT SQL RULES: - Use Market_Cap_Billions and EBITDA_Billions (derived, in $B) for readable output.
                                    - Always LIMIT results unless aggregating or user explicitly wants all.
                                    - For case-insensitive sector/name search use: LOWER(Sector) LIKE LOWER('%tech%')"""


