import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- 1. Load API Keys ---
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("🚨 CRITICAL ERROR: GROQ_API_KEY not found in .env file.")
    st.stop()

# --- 2. Imports ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# IMPORT GROQ (LLAMA 3) INSTEAD OF GOOGLE
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 3. Configuration ---
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Using Llama 3 (70 Billion parameter version - Very Smart)
LLM_MODEL = "llama-3.3-70b-versatile"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Still using Local CPU for memory (No rate limits)
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    # --- CHANGED TO GROQ (LLAMA 3) ---
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=LLM_MODEL,
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Insurance Policy Advisor.
    Answer the question based ONLY on the context provided below.
    If the answer is not in the context, say "I cannot find that information in the document."
    
    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)
    
    return retrieval_chain

def main():
    st.set_page_config(page_title="Insurance AI", page_icon="🛡️")
    st.title("🛡️ Policy Document Q&A Assistant")
    st.write("Engine: **Llama 3 (via Groq)** + **HuggingFace** (Local Memory)")

    # Sidebar
    with st.sidebar:
        st.header("Upload Policy")
        pdf_docs = st.file_uploader("Choose a PDF", accept_multiple_files=True)
        
        if st.button("Analyze Policy"):
            with st.spinner("Processing locally..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    
                    st.session_state.vector_store = vector_store
                    st.success("✅ Policy Processed!")
                else:
                    st.warning("Upload a PDF first.")

    # Chat Area
    user_question = st.text_input("Ask a question regarding your policy:")

    if user_question:
        if "vector_store" not in st.session_state:
            st.error("Please upload and analyze a document first.")
        else:
            chain = get_conversational_chain(st.session_state.vector_store)
            response = chain.invoke({"input": user_question})
            st.markdown("### Answer:")
            st.write(response["answer"])

if __name__ == "__main__":
    main()