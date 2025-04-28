from fastapi import FastAPI, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import logging
import shutil
import json
from pathlib import Path
from datetime import datetime

# PDF processing
from PyPDF2 import PdfReader

# LangChain components
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-chatbot")

# Check for Google API key
if not os.getenv("GOOGLE_API_KEY"):
    logger.warning("GOOGLE_API_KEY not found in environment variables!")

# Constants
UPLOAD_DIR = Path("uploads")
VECTOR_STORE_DIR = Path("vector_stores")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="PDF Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store document metadata
document_store = {}
if os.path.exists("document_metadata.json"):
    try:
        with open("document_metadata.json", "r") as f:
            document_store = json.load(f)
    except Exception as e:
        logger.error(f"Error loading document metadata: {e}")

def save_document_metadata():
    """Save document metadata to JSON file"""
    try:
        with open("document_metadata.json", "w") as f:
            json.dump(document_store, f)
    except Exception as e:
        logger.error(f"Error saving document metadata: {e}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        raise

def create_vector_store(text, doc_id):
    """Create and save a vector store from text"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            logger.warning("No text chunks created")
            return False
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create vector store
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
        # Save vector store
        vector_store_path = VECTOR_STORE_DIR / f"{doc_id}"
        vector_store.save_local(str(vector_store_path))
        
        return str(vector_store_path)
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

def get_qa_chain(doc_id):
    """Create a retrieval chain for QA"""
    try:
        # Check if vector store exists
        vector_store_path = document_store[doc_id]["vector_store_path"]
        if not os.path.exists(vector_store_path):
            logger.error(f"Vector store not found at {vector_store_path}")
            return None
        
        # Load vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        # Create a template
        template = """
        You are an AI assistant answering questions based on provided document content.
        Answer the question strictly based on the document context provided.
        If the answer cannot be found in the context, respond with "I don't know" or "I cannot find the answer in the document."
        Do not make up or infer information that's not explicitly in the context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        # Create a prompt template using the new structure
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create a retrieval chain using the modern API
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        raise

# API Endpoints
@app.post("/api/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Create document directory
        doc_dir = UPLOAD_DIR / doc_id
        doc_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = doc_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract text
        text = extract_text_from_pdf(file_path)
        if not text or len(text) < 50:  # Basic validation
            raise HTTPException(status_code=400, detail="Could not extract text from PDF or PDF is too short")
        
        # Create vector store
        vector_store_path = create_vector_store(text, doc_id)
        if not vector_store_path:
            raise HTTPException(status_code=500, detail="Failed to create vector store")
        
        # Store document metadata
        document_store[doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "upload_date": datetime.now().isoformat(),
            "vector_store_path": vector_store_path
        }
        
        # Save metadata
        save_document_metadata()
        
        return {"id": doc_id, "filename": file.filename, "message": "Document processed successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/api/documents")
async def get_documents():
    """Get list of uploaded documents"""
    documents = list(document_store.values())
    return {"documents": documents}

@app.post("/api/documents/{doc_id}/ask")
async def ask_question(doc_id: str, question: str = Form(...)):
    """Ask a question about a specific document"""
    try:
        # Check if document exists
        if doc_id not in document_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Validate question
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Get QA chain
        chain = get_qa_chain(doc_id)
        if not chain:
            raise HTTPException(status_code=500, detail="Could not create QA chain")
        
        # Get answer using the new invoke method
        try:
            answer = chain.invoke(question)
            answer = answer.strip()
        except Exception as e:
            logger.error(f"Error invoking chain: {e}")
            answer = "Error processing your question. Please try again."
        
        # Fallback if answer is empty
        if not answer:
            answer = "I don't know the answer based on the document content."
        
        return {"answer": answer}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    try:
        # Check if document exists
        if doc_id not in document_store:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete document directory
        doc_dir = UPLOAD_DIR / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        
        # Delete vector store
        vector_store_path = Path(document_store[doc_id]["vector_store_path"])
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
        
        # Remove from document store
        del document_store[doc_id]
        
        # Save metadata
        save_document_metadata()
        
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting PDF Chatbot API")
    # Validate directories
    UPLOAD_DIR.mkdir(exist_ok=True)
    VECTOR_STORE_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)