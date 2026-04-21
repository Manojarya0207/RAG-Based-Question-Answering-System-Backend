import os
import uuid
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from sqlalchemy.orm import Session

load_dotenv()

from models import QueryRequest, QueryResponse, DocumentResponse, SourceChunk, DocumentMetadata
from services.ingestion_service import IngestionService
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.llm_service import LLMService
from database import get_db, DocumentModel, ChatMessageModel

# Initialize app and dependencies
app = FastAPI(title="Pro RAG-Based Question Answering System")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
ingestion_service = IngestionService()
embedding_service = EmbeddingService()
vector_store = VectorStore()
llm_service = LLMService()

# Constants
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_document_background(doc_id: str, file_path: str, filename: str):
    """
    Background task to process uploaded documents and generate summary.
    """
    db = next(get_db())
    try:
        doc_record = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        if not doc_record:
            return
            
        doc_record.status = "processing"
        db.commit()
        
        # 1. Parse and Chunk
        chunks_metadata = ingestion_service.process_document(file_path, doc_id, filename)
        
        # 2. Embed
        texts = [c['text'] for c in chunks_metadata]
        embeddings = embedding_service.encode(texts)
        
        # 3. Add to Vector Store
        vector_store.add_documents(embeddings, chunks_metadata)
        
        # 4. Generate Summary (IMPROVED)
        # Sample more strategically: Start, Middle, and End
        if len(texts) > 5:
            indices = [0, len(texts)//4, len(texts)//2, 3*len(texts)//4, len(texts)-1]
            summary_context = " ".join([texts[i] for i in sorted(list(set(indices)))])
        else:
            summary_context = " ".join(texts)
            
        summary = llm_service.summarize_document(summary_context)
        doc_record.summary = summary
        
        doc_record.status = "ready"
        db.commit()
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing document {doc_id}: {error_msg}")
        doc_record = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
        if doc_record:
            doc_record.status = "failed"
            doc_record.error = error_msg
            db.commit()
    finally:
        db.close()

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Save to SQLite
    new_doc = DocumentModel(
        id=doc_id,
        filename=file.filename,
        status="pending",
        created_at=time.time()
    )
    db.add(new_doc)
    db.commit()
    
    background_tasks.add_task(process_document_background, doc_id, file_path, file.filename)
    
    return DocumentResponse(document_id=doc_id, status="pending", filename=file.filename)

@app.get("/status/{doc_id}")
async def get_status(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "status": doc.status,
        "error": doc.error if doc.status == "failed" else None,
        "summary": doc.summary
    }

@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents(db: Session = Depends(get_db)):
    docs = db.query(DocumentModel).all()
    return [DocumentMetadata(
        id=d.id,
        filename=d.filename,
        status=d.status,
        error=d.error,
        created_at=d.created_at
    ) for d in docs]

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from vector store
    vector_store.delete_document(doc_id)
    
    # Remove from SQLite
    db.delete(doc)
    db.commit()
    
    return {"status": "deleted"}

@app.get("/documents/{doc_id}/file")
async def get_document_file(doc_id: str, db: Session = Depends(get_db)):
    """
    Serves the original document file (PDF or TXT).
    """
    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Reconstruct filename used in upload
    # Search for files starting with doc_id in UPLOAD_DIR
    files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(doc_id)]
    if not files:
        raise HTTPException(status_code=404, detail="Original file not found on server")
        
    file_path = os.path.join(UPLOAD_DIR, files[0])
    
    # Determine media type
    media_type = "application/pdf" if doc.filename.lower().endswith(".pdf") else "text/plain"
    
    return FileResponse(file_path, media_type=media_type, filename=doc.filename)

@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_documents(query_data: QueryRequest, request: Request, db: Session = Depends(get_db)): 
    # 1. Fetch Chat History (NEW FEATURE)
    history_records = db.query(ChatMessageModel).order_by(ChatMessageModel.timestamp.desc()).limit(10).all()
    history = [{"role": h.role, "content": h.content} for h in reversed(history_records)]
    
    # 2. Embed query
    query_vector = embedding_service.encode_query(query_data.question)
    
    # 3. Search vector store
    search_results = vector_store.search(query_vector, top_k=query_data.top_k, doc_id=query_data.document_id)
    
    if not search_results:
        return QueryResponse(
            answer="I couldn't find any relevant information in the uploaded documents.",
            sources=[]
        )
    
    # 4. Generate answer with history
    context_chunks = [r['text'] for r in search_results]
    answer = llm_service.generate_answer(query_data.question, context_chunks, history=history)
    
    # 5. Save interaction to History (NEW FEATURE)
    db.add(ChatMessageModel(role="user", content=query_data.question))
    db.add(ChatMessageModel(role="assistant", content=answer))
    db.commit()
    
    # 6. Prepare sources
    sources = [
        SourceChunk(
            chunk_index=r['chunk_index'],
            score=r['score'],
            text=r['text'],
            filename=r['filename']
        ) for r in search_results
    ]
    
    return QueryResponse(answer=answer, sources=sources)

@app.get("/documents/{doc_id}/summary")
async def get_document_summary(doc_id: str, db: Session = Depends(get_db)):
    """
    Fetches the automatically generated summary for a document.
    """
    doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc.status != "ready":
        return {"summary": "Document is still processing..."}
        
    return {"summary": doc.summary or "Summary not available for this document."}

@app.post("/history/clear")
async def clear_history(db: Session = Depends(get_db)):
    """
    Clears the chat history.
    """
    db.query(ChatMessageModel).delete()
    db.commit()
    return {"status": "history cleared"}

@app.get("/history")
async def get_history(db: Session = Depends(get_db)):
    """
    Fetches the full chat history.
    """
    history_records = db.query(ChatMessageModel).order_by(ChatMessageModel.timestamp.asc()).all()
    return [{"role": h.role, "content": h.content, "timestamp": h.timestamp} for h in history_records]

@app.get("/debug/config")
async def debug_config():
    """
    Exposes non-sensitive configuration for debugging.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "not set"
    mock_status = os.getenv("MOCK_MODE", "false").lower() == "true"
    
    return {
        "openai_api_key_status": "set" if api_key else "missing",
        "openai_api_key_preview": masked_key,
        "upload_dir_exists": os.path.exists(UPLOAD_DIR),
        "mock_mode": mock_status,
        "db_path": "./data/rag_system.db",
        "current_time": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
