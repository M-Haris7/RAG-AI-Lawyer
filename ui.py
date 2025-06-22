import os, tempfile, re
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline import answer_query, llm_model

# Step 1: Setup Streamlit app
st.set_page_config(page_title="Lawgic AI", page_icon="⚖️", layout="centered")
st.title("Lawgic AI")
st.caption("Hey, this is your AI Lawyer. Ask questions about the PDFs you upload.")

# Step 2: Upload files
uploaded_files = st.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

# Helper: Build vectorstore from uploaded PDFs
def build_vectorstore_from_uploaded_files(files):
    documents = []

    for up_file in files:
        # 1) Write Streamlit UploadedFile → temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(up_file.getbuffer())
            temp_path = tmp.name

        # 2) Load text from that temp PDF
        loader = PDFPlumberLoader(temp_path)
        documents.extend(loader.load())

        # 3) Clean up temp file
        os.remove(temp_path)

    # 4) Split docs to chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = splitter.split_documents(documents)

    # 5) Embed & index in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.from_documents(chunks, embeddings)


# Step 3: Set up chat interface
user_query = st.text_area(
    "Enter your query", placeholder="Type your question here…", height=100
)

if st.button("Ask Lawyer"):
    if not uploaded_files:
        st.error("Please upload at least one PDF file to ask a question.")
        st.stop()

    st.chat_message("user").write(user_query)

    # Build a fresh vector store for the current uploads
    with st.spinner("Processing documents…"):
        vectorstore = build_vectorstore_from_uploaded_files(uploaded_files)

    # Retrieve relevant chunks & call the LLM
    retrieved_docs = vectorstore.similarity_search(user_query)
    response      = answer_query(retrieved_docs, llm_model, user_query)

    # Strip <think> block
    full_response = response.content
    think_match   = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
    reasoning     = think_match.group(1).strip() if think_match else None
    clean_answer  = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

    # Display answer
    st.chat_message("AI Lawyer").write(clean_answer)

    # Optional detailed reasoning
    if reasoning:
        with st.expander("🔍 Show detailed reasoning (from AI)"):
            st.markdown(f"```\n{reasoning}\n```")
