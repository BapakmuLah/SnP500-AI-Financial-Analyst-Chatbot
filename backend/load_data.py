import os
import re
import sqlite3
import pandas as pd

from pathlib import Path
from langchain_community.utilities import SQLDatabase



# DATASET PATH 
os.makedirs("data", exist_ok=True)
DATABASE_NAME = "financial.db"
TABLE_NAME = "financials_data"
DB_PATH = Path(f"data/{DATABASE_NAME}")
CSV_PATH = Path("data/financials.csv")
META_PATH = Path("data/db_metadata.json")

# DATASET SCHEMA
custom_table_description = {"financials_data": """Table containing financial metrics of companies. 
                            1. "Symbol" TEXT, --> TICKER STOCK 
                            2. "Name" TEXT, ---> NAME OF COMPANY
                            3. "Sector" TEXT, --> INDUSTRY SECTOR 
                            4. "Price" REAL, --> current price, 
                            5. "Price_Earnings" REAL, --> Rasio P/E (Price to Earnings)
                            6. "Dividend_Yield" REAL --> stored as actual percentage number,
                            7. "Earnings_Share" REAL, --> EPS (Earnings per Share)
                            8. "52_Week_Low" REAL --> the lowest price in the last 1 year, 
                            9. "52_Week_High" REAL --> the highest price in the last 1 year, 
                            10. "Market_Cap" REAL, --> Kapitalisasi pasar 
                            11. "EBITDA" REAL  --> Earnings Before Interest, Taxes, Depreciation, and Amortization., 
                            12. "Price_Sales" REAL, 
                            13. "Price_Book" REAL, 
                            14. "SEC_Filings" TEXT  --> Link to documents"""}


def clean_data(df : pd.DataFrame):
                                                                                       #   before      --->  after               before   ---> after
    # PREPROCESSING : REFACTORING COLUMN NAME TO IMPROVE LLM UNDERSTANDING . FOR EXAMPLE : Price/Sales ---> Price_Sales,   Dividend Yield ---> Dividend_Yield
    df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')

    # TRANSFORM NUMERIC COLUMN INTO NUMERIC DATA TYPE
    numeric_cols = ["Price", "Price_Earnings", "Dividend_Yield", "Earnings_Share", "52_Week_High", "52_Week_Low", "Market_Cap", "EBITDA", "Price_Sales", "Price_Book"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # CREATE SOME NEW FEATURE
    df["52_Week_Range_Pct"] = ((df["Price"] - df["52_Week_Low"]) / (df["52_Week_High"] - df["52_Week_Low"]) * 100).round(2)
    df["Market_Cap_Billions"] = (df["Market_Cap"] / 1e9).round(3)
    df["EBITDA_Billions"] = (df["EBITDA"] / 1e9).round(3)
    
    return df


# SAVE NEW DATABASE
def save_data(CSV_PATH, DB_PATH, TABLE_NAME):

    # LOAD DATA
    df = pd.read_csv(CSV_PATH)

    # PREPROCESSING & CLEAN DATA
    cleaned_df = clean_data(df)

    # CONNECT DATABASE & SAVE CLEANED DATA INTO DATABASE
    sql_conn = sqlite3.connect(DB_PATH)
    cleaned_df.to_sql(name = TABLE_NAME, con = sql_conn, if_exists = 'replace', index = False)

    sql_conn.close()
    print('Database Saved on ', DB_PATH)


# LOAD DATABASE
def load_data(DB_PATH, CSV_PATH, save_database = False):

    # IF THERE IS NO DATABASE YET
    if save_database:
        save_data(CSV_PATH, DB_PATH, TABLE_NAME)

    # LOAD SQLITE DATABASE
    db = SQLDatabase.from_uri(database_uri = f"sqlite:///{DB_PATH}", sample_rows_in_table_info = 2, custom_table_info = custom_table_description)

    # LOAD DATAFRAME
    df = pd.read_csv(CSV_PATH)

    return {'db' : db, "df" : df}



## REMOVE MARKDOWN CHARACTER
def clean_output(text: str):
    
    # REMOVE BACKTICKS (MARKDOWN)
    text = re.sub(r"```.*?\n", "", text)
    text = text.replace("```", "")

    return text.strip()


# LOAD DATAFRAME & DATABASE
# LOAD DATABASE
data = load_data(DB_PATH, CSV_PATH)
db = data['db']
df = data['df']



## LOAD EVAL DATA
def preview_data(DB_PATH, CSV_PATH, save_database=False):
    df = pd.read_csv(CSV_PATH)
    cleaned_df = clean_data(df.copy()) # Clean it first
    
    if save_database or not os.path.exists(DB_PATH):
        sql_conn = sqlite3.connect(DB_PATH)
        cleaned_df.to_sql(name=TABLE_NAME, con=sql_conn, if_exists='replace', index=False)
        sql_conn.close()
        
    db = SQLDatabase.from_uri(database_uri=f"sqlite:///{DB_PATH}", 
                              sample_rows_in_table_info=2, 
                              custom_table_info = custom_table_description)
    
    # Return the CLEANED dataframe so the columns match your code
    return {'db': db, "df": cleaned_df} 

# LOAD DATA
data_bundle = preview_data(DB_PATH, CSV_PATH)
preview_db = data_bundle['db']
preview_df = data_bundle['df'] 