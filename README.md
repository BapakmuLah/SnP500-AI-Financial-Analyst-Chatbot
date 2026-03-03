# 📈 S&P 500 AI Financial Automated Data Analyst

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

An interactive Natural Language to SQL (NL2SQL) chatbot designed to translate conversational queries into actionable financial insights. Built to explore and visualize S&P 500 financial data instantly, this project leverages a sophisticated multi-agent AI architecture to go beyond simple query generation.


# 📺 Live Demo

**🔗 Frontend Demo (UI) : [Live Demo on Vercel](snp500-ai-financial-analyst-chatbot.vercel.app)** <br>
**🔗 Backend Demo (API) : [Live Demo on Hugging Face Spaces](https://sandking-snp-500-ai-financial-chatbot.hf.space/docs)**


# ✨ Key Features

* **🗣️ Conversational NL2SQL Data Exploration**
    Instantly translate natural language queries into accurate SQLite commands to extract market caps, P/E ratios, and stock prices from the S&P 500 dataset.
* **🧠 Specialized Multi-Agent Orchestration**
    Overcomes the limitations of standard single-agent systems. The workflow dynamically routes tasks among specialized agents for:
    * *Intent Classification*
    * *SQL Generation*
    * *Deep Statistical Analysis*
    * *Dynamic Chart Visualization*
* **🛡️ Enterprise-Grade Security & Guardrails**
    Features a strict frontline validation layer to secure the database. It successfully mitigates prompt injection attacks and aggressively blocks unauthorized non-SELECT queries (e.g., `DROP`, `DELETE`, `ALTER`, `UPDATE`).
* **📊 Automated Performance Benchmarking**
    Integrates an MLOps pipeline using an **LLM-as-a-Judge** approach against a Golden Test Set. Continuously tracks and logs SQL generation accuracy, relevance, and completeness scores.
* **💻 Seamless UI/UX**
    A highly responsive, custom glassmorphism frontend providing a smooth cross-device experience, featuring interactive data cards and native Plotly rendering.


# 🔄 Architecture Pipeline

The system utilizes a sophisticated LangChain-based multi-agent routing architecture:
1. **User Input Layer:** Captures natural language queries via the frontend interface.
2. **Guardrail & Intent Agent:** Acts as the first line of defense. It validates input, neutralizes prompt injections, and classifies the intent (e.g., Data Lookup, Ranking, Visualization) to route the query appropriately.
3. **SQL Agent:** Translates the validated natural language query into an optimized SQLite query and executes it against the financial database.
4. **Analysis Agent:** Ingests the raw SQL output to perform descriptive statistical analysis, flag outliers, and generate concise financial insights.
5. **Visualization Agent:** Intelligently determines the optimal chart type based on the data structure and generates a rendering-ready Plotly JSON schema.
6. **Synthesis Layer:** Aggregates findings from all specialized agents into a cohesive, user-friendly markdown response.

# 💼 Business Model

Currently positioned as a **B2B Enterprise SaaS / Internal BI Tool**, this platform is designed to drastically reduce the time financial analysts spend writing repetitive SQL queries and building preliminary reports. 
* **Target Audience:** Investment firms, hedge funds, fintech startups, and data-driven retail investors.
* **Value Proposition:** Reduces data-to-insight latency from hours to seconds, lowering operational overhead and democratizing complex financial data access for non-technical stakeholders.


# 🛠️ Tech Stack

* **AI & LLM:** LangChain, Google Gemini API (`gemma-3-27b-it`)
* **Backend:** FastAPI, Python, SQLite, Pandas
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Tailwind CSS (CDN), Plotly.js
* **Deployment:** Hugging Face Spaces

# 🔌 API Endpoints

The FastAPI backend exposes the following RESTful endpoints to power the frontend and evaluation tools:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root endpoint. Returns a welcome message and a link to the full API documentation. |
| `GET` | `/create-database` | Initializes the system by cleaning the CSV data and saving it into the SQLite database. |
| `POST` | `/api/chat` | Main NL2SQL endpoint. Accepts a user query and returns the classified intent, generated SQL, text response, and Plotly figure JSON. |
| `GET` | `/api/overview` | Retrieves a high-level dataset summary (total companies, market leader, avg P/E, etc.) for the frontend interactive dashboard cards. |
| `POST` | `/api/evaluate` | Triggers the LLM-as-a-Judge evaluation pipeline against the Golden Test Set and logs the iteration results. |
| `GET` | `/api/evaluate/history` | Fetches historical evaluation benchmark scores to track model performance across different prompt/model iterations. |


# 🗺️ Future Roadmap & Monetization

To scale this MVP into a fully production-ready commercial product, the following milestones are planned:
* **Real-Time Data Ingestion:** Transition from a static CSV to live, streaming data feeds using external financial APIs (e.g., Alpha Vantage, Bloomberg API).
* **Predictive Analytics Agent:** Introduce a dedicated machine learning agent capable of forecasting basic stock trends or clustering companies based on financial health indicators.
* **Multi-Tenant Architecture:** Implement robust user authentication (OAuth2), chat history retention, and personalized workspaces for different enterprise analyst teams.
* **API-as-a-Service (Monetization):** Offer tiered API subscription plans, allowing other fintech platforms to seamlessly embed this proprietary NL2SQL conversational engine into their own applications.

# Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/sp500-ai-analyst.git](https://github.com/yourusername/sp500-ai-analyst.git)
   cd sp500-ai-analyst
