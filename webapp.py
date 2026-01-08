import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Vector database wrapper
from langchain_community.embeddings import HuggingFaceEmbeddings  # Word embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Chunking

from pypdf import PdfReader
# import faiss  # Not needed, FAISS wrapper from LangChain is enough
import streamlit as st
from pdfextractor import text_extractor_pdf  # Custom PDF text extractor

# ✅ Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Create the main page
st.title(':green[RAG Based CHATBOT]')
tips = """Follow the steps to use this application:
* Upload your PDF document in the sidebar.
* Write your query and start chatting with the bot."""
st.subheader(tips)

# Load PDF in Sidebar
st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF Only)]')
file_uploaded = st.sidebar.file_uploader('Upload File', type=['pdf'])  # restrict to PDF only

if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)  # works with BytesIO object

    # Step 1: Configure the models
    key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # Step 2: Configure Embedding Model
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Step 3: Chunking (Create Chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Step 4: Create FAISS Vector Store
    vector_store = FAISS.from_texts(chunks, embedding_model)

    # Step 5: Configure retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Function to generate response
    def generate_response(query: str) -> str:
        # Retrieval
        retrieved_docs = retriever.invoke(query)  # instead of get_relevant_documents
        context = ' '.join([doc.page_content for doc in retrieved_docs])

        # Augmented prompt
        prompt = f"""
        You are a helpful assistant using RAG.
        Context: {context}
        User Query: {query}
        """

        # Generation
        content = llm_model.generate_content(prompt)
        return content.text if hasattr(content, "text") else content.candidates[0].content.parts[0].text

    # Initialize chat if there is no history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display the History
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f':green[User:] :blue[{msg["text"]}]')
        else:
            st.write(f':orange[Chatbot:] {msg["text"]}')

    # Input from the user (Using Streamlit Form)
    with st.form('Chat Form', clear_on_submit=True):
        user_input = st.text_input('Enter Your Text Here:')
        send = st.form_submit_button('Send')

    # Start the conversation and append the output and query in history
    if user_input and send:
        st.session_state.history.append({"role": 'user', "text": user_input})
        model_output = generate_response(user_input)
        st.session_state.history.append({'role': 'chatbot', 'text': model_output})
        st.rerun()