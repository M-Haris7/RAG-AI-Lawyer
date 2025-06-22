from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

#Step 1: Upload & Load raw PDF(s)

pdfs_directory = "pdfs/" # Directory where PDFs are stored

def upload_pdf(file):
    """
    Function to upload a PDF file and save it to the specified directory.
    """
    with open(f"{pdfs_directory}{file.name}", "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    """
    Function to load all PDF files from the specified directory.
    Returns a list of loaded documents.
    """
    pdf_files = [f for f in os.listdir(pdfs_directory) if f.endswith('.pdf')]
    docs = []
    
    for pdf_file in pdf_files:
        loader = PDFPlumberLoader(f"{pdfs_directory}{pdf_file}")
        docs.extend(loader.load())
    
    return docs

file_path = "universal_declaration_of_human_rights.pdf"
documents = load_pdf(file_path)
# print(f"Loaded {len(document)} pages from {file_path}")

#Step 2: Create Chunks

def create_chunks(documents):
    """
    Function to split documents into smaller chunks for processing.
    Returns a list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
        )
    text_chunks = text_splitter.split_documents(documents)
    
    return text_chunks

text_chunks = create_chunks(documents)
# print(f"Created {len(text_chunks)} text chunks from the documents.")

#Step 3: Setup Embeddings Model (Use BAAI embedding model with Hugging Face)

def setup_embeddings():
    """
    Function to set up the Hugging Face embeddings model.
    Returns the embeddings model.
    """
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    return embeddings


#Step 4: Index Documents **Store embeddings in FAISS (vector store)

FAISS_DB_PATH="vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, setup_embeddings())
faiss_db.save_local(FAISS_DB_PATH)