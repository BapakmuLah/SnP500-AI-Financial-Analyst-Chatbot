
import time
import json
import sqlite3
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


from load_data import clean_output



## GET API KEY & DEFINE MODEL
load_dotenv()
MODEL_NAME = "models/gemma-3-27b-it"
llm = ChatGoogleGenerativeAI(model = MODEL_NAME, temperature=0)


# EVALUATION PROMPT
JUDGE_SYSTEM = """You are an evaluation judge for a financial data chatbot.
                Score the chatbot response on 3 axes (0.0 – 1.0 each):
                1. relevance:     Does the response directly address the user's question?
                2. accuracy:      Are the facts/numbers consistent with the data provided?
                3. completeness:  Does it cover all aspects the user asked about?

                Respond ONLY with JSON:
                {"relevance": 0.0-1.0, "accuracy": 0.0-1.0, "completeness": 0.0-1.0, "reasoning": "brief"}"""

# GROUND TRUTH TEST
GOLDEN_TEST_SET = [{"id":"T01","question":"What are the top 5 companies by market cap?",
                    "expected_intent":"RANKING","must_contain":["Apple","Microsoft","AAPL","MSFT"]}]

# EVAL DATABASE PATH
EVAL_DB = r"data/eval_results.db"

def _llm_judge(question: str, response: str, data_preview: str) -> dict:
    evaluation_prompt = f"""{JUDGE_SYSTEM}
                        Question: {question}
                        Data available: {data_preview[:800] if data_preview else "None"}
                        Chatbot response: {response[:1200]}"""
    
    # SCORING WITH LLM
    result = llm.invoke(input=evaluation_prompt, config={'temperature': 0, "max_output_tokens": 300})
    result_clean = clean_output(result.content)
    return json.loads(result_clean)

class Evaluator:
    def __init__(self, iteration_name: str = "baseline", EVAL_DB=EVAL_DB):
        self.iteration_name = iteration_name
        self.SAVE_EVAL = EVAL_DB
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.SAVE_EVAL)
        conn.execute("""CREATE TABLE IF NOT EXISTS eval_results (
            id TEXT, iteration TEXT, question TEXT, timestamp REAL, latency REAL,
            sql_score REAL, intent_score REAL, relevance REAL, accuracy REAL,
            completeness REAL, must_contain REAL, overall REAL,
            response_preview TEXT, sql_used TEXT, predicted_intent TEXT, judge_reasoning TEXT,
            PRIMARY KEY (id, iteration))""")
        conn.commit()
        conn.close()

    def evaluate_single(self, test_case: dict, agent_output: dict, latency: float) -> dict:
        response = agent_output.get("response","")
        df = agent_output.get("dataframe", pd.DataFrame())
        sql = agent_output.get("sql","")
        predicted_intent = agent_output.get("intent","")
        sql_error = agent_output.get("sql_error", False)

        true_intent = test_case.get("expected_intent","")
        true_mc     = test_case.get("must_contain",[])

        sql_score = 0.0 if sql_error else (0.5 if df.empty else 1.0)
        intent_score = 1.0 if predicted_intent == true_intent else 0.0
        mc_score = 1.0 if not true_mc else (1.0 if any(w.lower() in response.lower() for w in true_mc) else 0.0)

        data_preview = df.head(5).to_string() if not df.empty else ""
        judge = _llm_judge(test_case["question"], response, data_preview)

        overall = (sql_score * 0.20 + intent_score * 0.15 + judge["relevance"] * 0.25 + 
                   judge["accuracy"] * 0.25 + judge["completeness"]*0.10 + mc_score*0.05)

        result = {
            "id": test_case["id"], "iteration": self.iteration_name,
            "question": test_case["question"], "timestamp": time.time(),
            "latency": round(latency,2), "sql_score": sql_score,
            "intent_score": intent_score, "relevance": judge["relevance"],
            "accuracy": judge["accuracy"], "completeness": judge["completeness"],
            "must_contain": mc_score, "overall": round(overall,4),
            "response_preview": response[:200], "sql_used": sql[:200],
            "predicted_intent": predicted_intent,
            "judge_reasoning": judge.get("reasoning",""),
        }

        conn = sqlite3.connect(self.SAVE_EVAL)
        conn.execute("""INSERT OR REPLACE INTO eval_results VALUES
                        (:id,:iteration,:question,:timestamp,:latency,:sql_score,:intent_score,
                        :relevance,:accuracy,:completeness,:must_contain,:overall,
                        :response_preview,:sql_used,:predicted_intent,:judge_reasoning)""", result)
        conn.commit()
        conn.close()
        return result
    
    def run_full_eval(self, orchestrator) -> pd.DataFrame:
        results = []
        for test_case in GOLDEN_TEST_SET:
            start = time.time()
            output = orchestrator.run(test_case["question"])
            latency = time.time() - start
            score = self.evaluate_single(test_case, output, latency)
            results.append(score)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compare_iterations() -> pd.DataFrame:
        conn = sqlite3.connect(EVAL_DB)
        df = pd.read_sql_query("SELECT iteration, AVG(overall) as mean_overall, AVG(latency) as mean_latency,"
                                " AVG(relevance) as mean_relevance, AVG(sql_score) as mean_sql, COUNT(*) as n"
                                " FROM eval_results GROUP BY iteration ORDER BY mean_overall DESC", conn)
        conn.close()
        return df.round(3)