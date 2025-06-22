# Lawgic AI ⚖️

An intelligent legal assistant powered by Retrieval-Augmented Generation (RAG) that helps users get answers from legal documents. Upload PDF documents and ask questions to get precise, context-based legal information.

## 🌟 Features

- **PDF Document Processing**: Upload multiple PDF files containing legal documents
- **Intelligent Question Answering**: Ask questions in natural language and get accurate responses
- **Context-Aware Responses**: Answers are generated only from the uploaded documents, ensuring reliability
- **Advanced Reasoning**: Uses DeepSeek R1 model with detailed reasoning capabilities
- **User-Friendly Interface**: Clean Streamlit web interface for easy interaction
- **Vector Search**: Efficient similarity search using FAISS for relevant document retrieval

## 🏗️ Architecture

The system consists of four main components:

1. **Document Processing** (`vector_database.py`): Handles PDF loading, text chunking, and vector storage
2. **RAG Pipeline** (`rag_pipeline.py`): Core logic for document retrieval and answer generation
3. **User Interface** (`ui.py`): Streamlit web application for user interaction
4. **Requirements** (`requirements.txt`): All necessary dependencies

## 🛠️ Technology Stack

- **LLM**: DeepSeek R1 Distill LLaMA 70B via Groq
- **Embeddings**: BAAI/bge-small-en-v1.5 (HuggingFace)
- **Vector Database**: FAISS
- **PDF Processing**: PDFPlumber
- **Framework**: LangChain
- **UI**: Streamlit
- **Text Splitting**: RecursiveCharacterTextSplitter

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key (for DeepSeek R1 model access)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd lawgic-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Create necessary directories**
   ```bash
   mkdir pdfs vectorstore
   ```

## 💻 Usage

### Running the Streamlit App

1. **Start the application**
   ```bash
   streamlit run ui.py
   ```

2. **Access the web interface**
   Open your browser and navigate to `http://localhost:8501`

3. **Upload PDF documents**
   - Click on "Upload PDF(s)" button
   - Select one or more PDF files containing legal documents

4. **Ask questions**
   - Type your legal question in the text area
   - Click "Ask Lawyer" to get an answer
   - View detailed AI reasoning in the expandable section

### Using Individual Components

#### Building Vector Database
```python
from vector_database import load_pdf, create_chunks, setup_embeddings
from langchain_community.vectorstores import FAISS

# Load documents
documents = load_pdf("path/to/your/pdf")

# Create chunks
text_chunks = create_chunks(documents)

# Build vector database
embeddings = setup_embeddings()
faiss_db = FAISS.from_documents(text_chunks, embeddings)
```

#### Direct RAG Pipeline Usage
```python
from rag_pipeline import answer_query, llm_model

# Retrieve relevant documents
retrieved_docs = faiss_db.similarity_search("your question")

# Get answer
response = answer_query(retrieved_docs, llm_model, "your question")
print(response.content)
```

## 🔧 Configuration

### Model Settings
- **Chunk Size**: 1000 characters (configurable in `vector_database.py`)
- **Chunk Overlap**: 200 characters
- **Similarity Search**: Top-k retrieval (default: 4 documents)

### Prompt Engineering
The system uses a carefully crafted prompt template that:
- Ensures responses are based only on provided context
- Maintains legal accuracy and clarity
- Handles insufficient information gracefully

## 📁 Project Structure

```
lawgic-ai/
├── rag_pipeline.py      # Core RAG logic and LLM integration
├── ui.py               # Streamlit web interface
├── vector_database.py  # Document processing and vector storage
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── pdfs/              # Directory for PDF storage (create this)
└── vectorstore/       # FAISS database storage (create this)
```

## 🔒 Important Notes

### Legal Disclaimer
- This tool is for informational purposes only
- Responses are based solely on uploaded documents
- Always consult qualified legal professionals for legal advice
- Verify all information independently

### Data Privacy
- Documents are processed locally
- No data is stored permanently unless explicitly configured
- API calls to Groq are made for LLM inference only


## 🚀 Future Enhancements

- [ ] Support for multiple document formats (Word, TXT)
- [ ] Conversation memory for follow-up questions
- [ ] Citation tracking and source referencing
- [ ] Advanced filtering and search capabilities
- [ ] Integration with legal databases
- [ ] Multi-language support

