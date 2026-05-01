import os
import sqlite3
import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px

from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
 # -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="AI SQL Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI SQL Data Analyst Agent")
st.write("Upload CSV → Ask Questions in English → Get SQL + Insights + Visualization")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read CSV
        df = pd.read_csv(uploaded_file, encoding="latin1")

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        # -----------------------------
        # Save to Temporary SQLite DB
        # -----------------------------
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = temp_db.name

        engine = create_engine(f"sqlite:///{db_path}")

        table_name = "sales_data"

        df.to_sql(
            table_name,
            engine,
            index=False,
            if_exists="replace"
        )

        # -----------------------------
        # Connect LangChain SQL DB
        # -----------------------------
        db = SQLDatabase(engine)

        llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            groq_api_key=groq_api_key
        )

        chain = create_sql_query_chain(llm, db)

        # -----------------------------
        # Ask Question
        # -----------------------------
        st.subheader("💬 Ask Questions About Your Data")

        question = st.text_input(
            "Example: Show top 5 products by sales"
        )

        if question:

            with st.spinner("Generating SQL Query..."):

                # Generate SQL
                sql_query = chain.invoke({
                    "question": question
                })

                # Clean SQL Output
                sql_query = (
                    sql_query
                    .replace("```sql", "")
                    .replace("```", "")
                    .strip()
                )

                st.subheader("🧠 Generated SQL Query")
                st.code(sql_query, language="sql")

                # -----------------------------
                # Execute SQL
                # -----------------------------
                result_df = pd.read_sql_query(sql_query, engine)

                st.subheader("📌 Query Result")
                st.dataframe(result_df)

                # -----------------------------
                # Natural Language Summary
                # -----------------------------
                st.subheader("📖 Insight Summary")

                summary_prompt = f"""
                Based on the SQL result below, summarize the insight in simple business language:

                Question: {question}

                Result:
                {result_df.head(20).to_string()}
                """

                summary = llm.invoke(summary_prompt)

                st.success(summary.content)

                # -----------------------------
                # Visualization
                # -----------------------------
                if len(result_df.columns) >= 2 and len(result_df) > 0:

                    st.subheader("📈 Visualization")

                    x_col = result_df.columns[0]
                    y_col = result_df.columns[1]

                    fig = px.bar(
                        result_df,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col} by {x_col}"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

    except Exception as e:
        st.error(f"Error: {str(e)}")