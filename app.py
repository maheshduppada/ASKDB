from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
import os
import urllib.parse

# Load environment variables
load_dotenv()

def init_database(user: str, password: str, host: str, port: int, database: str) -> SQLDatabase:
    user_encoded = urllib.parse.quote_plus(user)
    password_encoded = urllib.parse.quote_plus(password)
    db_uri = f"mysql+mysqlconnector://{user_encoded}:{password_encoded}@{host}:{port}/{database}"
    try:
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {str(e)}")

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database. 
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    Important Rules for SQL:
    1. When using GROUP BY, include all non-aggregated columns from SELECT in the GROUP BY clause
    2. Use GROUP_CONCAT() instead of multiple rows when showing grouped data
    3. For counting distinct values, always use COUNT(DISTINCT column)
    
    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text.
    
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    llm = ChatGroq(model="Mistral-saba-24b", temperature=0, api_key=groq_api_key)

    def get_schema(_):
        return db.get_table_info()

    return (RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser())

def calculate_hallucination_score(response: str, query: str, db_response: str) -> float:
    """
    Calculate a hallucination score (0-1) for the response.
    0 = completely factual, 1 = completely hallucinated.
    """
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("❌ ERROR: GROQ_API_KEY is not set!")
            return 0.5  # Default score

        llm = ChatGroq(model="Mistral-saba-24b", temperature=0, api_key=groq_api_key)
        prompt = """
        Analyze the following database response and determine how factual it is. Consider if the response contains any information not supported by the data or makes unsupported assumptions.
        
        User Query: {query}
        Database Response: {db_response}
        AI Response: {response}
        
        Rate the hallucination level on a scale from 0 to 1 where:
        0 = Completely factual and supported by data
        0.3 = Minor interpretation needed but generally factual
        0.7 = Significant interpretation or some unsupported claims
        1 = Completely hallucinated or unsupported by data
        
        Return ONLY the score as a float between 0 and 1, nothing else.
        """
        print("🔹 Sending hallucination check to Groq API...")
        evaluation = llm.invoke(prompt.format(query=query, db_response=db_response, response=response))
        print("✅ Groq API Response (Raw):", evaluation.content.strip())
        score = float(evaluation.content.strip())
        return min(max(score, 0), 1)  # Ensure score is between 0 and 1

    except Exception as e:
        print("❌ ERROR in hallucination score calculation:", e)
        return 0.5  # Default score if error occurs

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    """
    Generate a response based on user queries using SQL chain and database interactions.
    """
    sql_chain = get_sql_chain(db)

    print("🔹 Generating SQL Query...")
    sql_query = sql_chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    print("✅ SQL Query Generated:", sql_query)

    print("🔹 Running SQL Query...")
    db_response = db.run(sql_query)
    print("✅ Database Response:", db_response)

    print("🔹 Calculating Hallucination Score...")
    hallucination_score = calculate_hallucination_score(sql_query, user_query, db_response)
    print("✅ Hallucination Score:", hallucination_score)

    formatted_response = f"{db_response}\n\n---\n\n*Hallucination Score*: {hallucination_score:.2f}/1.0 "

    if hallucination_score < 0.3:
        formatted_response += "(Low - Highly Reliable)"
    elif hallucination_score < 0.7:
        formatted_response += "(Medium - Some Interpretation)"
    else:
        formatted_response += "(High - Potential Inaccuracies)"

    return formatted_response

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# Streamlit UI configuration
st.set_page_config(page_title="AskDB", page_icon=":mortar_board:")
st.title("AskDB")

# Sidebar for database connection
with st.sidebar:
    st.subheader("Database Connection")
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="", key="Password")
    st.text_input("Database", value="student_db", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    int(st.session_state["Port"]),
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("Connected to database!")
            except Exception as e:
                st.error(f"Failed to connect to the database: {e}")

# Display chat messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Handle user input
user_query = st.chat_input("Ask about student data...")
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        try:
            if "db" not in st.session_state:
                st.error("Please connect to the database first!")
            else:
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
        except Exception as e:
            response = f"An error occurred: {e}"
            st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))
