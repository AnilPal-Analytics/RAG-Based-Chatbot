# ======================================================
# ENTERPRISE-STYLE RAG BASED PDF CHATBOT (SINGLE FILE)
# ======================================================

import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------
# ENV & MODEL CONFIGURATION
# ------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel("gemini-2.5-flash-lite")

# ------------------------------------------------------
# PDF TEXT EXTRACTION
# ------------------------------------------------------
def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# ------------------------------------------------------
# VECTOR STORE CREATION
# ------------------------------------------------------
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=250
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)

# ------------------------------------------------------
# RAG RESPONSE GENERATION
# ------------------------------------------------------
def rag_answer(query, retriever):
    docs = retriever.invoke(query)
    context = " ".join(doc.page_content for doc in docs)

    prompt = f"""
You are a professional AI assistant.

Answer the user's question strictly using the information provided
in the context below. If the answer is not present, say:
"I could not find this information in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.generate_content(prompt)
    return response.text.strip()

# ======================================================
# STREAMLIT UI
# ======================================================

st.set_page_config(
    page_title="RAG PDF Chatbot",
    layout="wide"
)

# ------------------ HEADER ----------------------------
st.markdown(
    """
    <h1 style="text-align:center; color:#1F4FD8;">🤖 RAG-Based Document Chatbot</h1>
    <p style="text-align:center; color:gray;">
    Ask accurate questions from your PDF using Retrieval-Augmented Generation
    </p>
    """,
    unsafe_allow_html=True
)

# ------------------ SIDEBAR ---------------------------
st.sidebar.markdown("## 📄 Document Upload")
st.sidebar.markdown(
    "Upload a **PDF document** to activate document-aware chat."
)

uploaded_file = st.sidebar.file_uploader(
    "Select PDF File",
    type=["pdf"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ System Status")

# ------------------ SESSION STATE ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ------------------ DOCUMENT PROCESSING ---------------
if uploaded_file and st.session_state.retriever is None:
    with st.spinner("Analyzing document and building knowledge base..."):
        pdf_text = extract_pdf_text(uploaded_file)
        vector_db = build_vector_store(pdf_text)
        st.session_state.retriever = vector_db.as_retriever(
            search_kwargs={"k": 3}
        )

    st.sidebar.success("Document processed ✔")

elif st.session_state.retriever:
    st.sidebar.success("Ready for questions ✔")
else:
    st.sidebar.warning("Waiting for PDF upload")

# ------------------ CHAT AREA -------------------------
st.markdown("### 💬 Chat Interface")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div style='background:#e8f5e9; padding:10px; border-radius:8px;'><b>You:</b> {msg['text']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background:#f1f1f1; padding:10px; border-radius:8px;'><b>Assistant:</b> {msg['text']}</div>",
            unsafe_allow_html=True
        )

# ------------------ USER INPUT ------------------------
if st.session_state.retriever:
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input(
            "Ask a question based on the uploaded document"
        )
        send = st.form_submit_button("Send")

    if send and user_query:
        st.session_state.messages.append(
            {"role": "user", "text": user_query}
        )

        with st.spinner("Generating answer..."):
            answer = rag_answer(
                user_query,
                st.session_state.retriever
            )

        st.session_state.messages.append(
            {"role": "assistant", "text": answer}
        )

        st.rerun()
else:
    st.info("Upload a PDF from the sidebar to begin.")

