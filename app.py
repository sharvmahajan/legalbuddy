import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("LegalBuddy")

# Create two different prompts for summary and Q&A
qa_prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions:{input}
"""
)

summary_prompt = ChatPromptTemplate.from_template(
"""
Please provide a comprehensive summary of the following document context.
Focus on the key points and main ideas.
<context>
{context}
</context>
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Add option selection at the top
option = st.radio(
    "Choose functionality",
    ["Document Summary", "Ask Questions"]
)

if st.button("Process Documents"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if option == "Document Summary":
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, summary_prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': 'Summarize the document'})
        st.write("Summary Generation Time:", time.process_time() - start)
        
        st.subheader("Document Summary")
        st.write(response['answer'])
        
        with st.expander("View Source Sections"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

else:  # Ask Questions option
    prompt1 = st.text_input("Enter Your Question From Documents")
    
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response Time:", time.process_time() - start)
        
        st.subheader("Answer")
        st.write(response['answer'])
        
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")