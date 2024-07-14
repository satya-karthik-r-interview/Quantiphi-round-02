import streamlit as st
from utils import generate_answer, milvus_client, COLLECTION_NAME


# Streamlit app
st.title("Biology Q&A")
user_question = st.text_input("Enter your question:")
if user_question:
    answer = generate_answer(milvus_client, COLLECTION_NAME, user_question)
    st.text_area("Answer:", value=answer, height=500)