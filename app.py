from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the assistant. Please provide a helpful response to the user query."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("Offline Llama Chatbot 🦙")
input_text = st.text_input("Ask me anything:")

# Initialize the LLM (Offline model)
llm = Ollama(model="llama3.2:1b")
output_parser = StrOutputParser()

# Build the chain
chain = prompt | llm | output_parser

# Run when user gives input
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
