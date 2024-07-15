import streamlit as st
import requests
import json


# Streamlit app
st.title("Biology Q&A")
user_question = st.text_input("Enter your question:")
if user_question:
    # send a post request to localhost:8000/ask
    data = {
        'question': user_question,
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://localhost:8000/ask',
                             data=json.dumps(data), headers=headers)
    response_data = response.json()
    answer = response_data['answer']
    st.text_area("Answer:", value=answer, height=500)
