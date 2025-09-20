import streamlit as st
import os
# from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")

st.title("📁 ObjectBox Vector Store with llama3")

llm = ChatOllama(model="llama3:8b")
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
"""
)

# Vector Embedding and ObjectBox Vector Store db

def vector_embedding():
    if "vectors" not in st.session_state:
        # st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census") ##Data ingestion
        st.session_state.docs = st.session_state.loader.load() #Documents onboarding
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #Text splitting
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=1024) #Vector Store creation

input_prompt = st.text_input("Enter your query here: ")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("ObjectBox Database is ready. Documents embedded successfully!")

import time

if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start=time.process_time()

    response = retrieval_chain.invoke({'input': input_prompt})

    print("Time taken: ", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Source Documents"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------")


