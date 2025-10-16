# AskDB 🧠

AskDB is an intelligent **Natural Language to SQL** Streamlit app powered by **LangChain** and **Groq’s Mistral LLM**.  
It allows you to ask questions about your MySQL database in plain English and automatically generates and executes the corresponding SQL queries.

---

## 🚀 Features
- 🗣️ Convert natural language queries to SQL automatically
- 💾 Connect dynamically to any MySQL database
- 🔍 Uses Groq’s Mistral-saba-24b model for SQL generation
- ⚖️ Calculates a hallucination score to assess reliability
- 💬 Interactive Streamlit chat interface with history

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** – UI framework
- **LangChain** – chaining & prompt orchestration
- **Groq (Mistral-saba-24b)** – LLM backend
- **MySQL Connector** – database access
- **dotenv** – environment variable management

---

## 📦 Installation

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/AskDB.git
cd AskDB
