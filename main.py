import os
import uuid
import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from models import QueryRequest, QueryResponse, DocumentResponse, SourceChunk, DocumentMetadata
from services.ingestion_service import IngestionService
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.llm_service import LLMService

# Initialize app and dependencies
app = FastAPI(title="RAG-Based Question Answering System")
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

# In-memory document status store (shared across requests in this single process)
# In production, this would be a database.
documents_db: Dict[str, DocumentMetadata] = {}

# Constants
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_document_background(doc_id: str, file_path: str, filename: str):
    """
    Background task to process uploaded documents.
    """
    try:
        documents_db[doc_id].status = "processing"
        
        # 1. Parse and Chunk
        chunks_metadata = ingestion_service.process_document(file_path, doc_id, filename)
        
        # 2. Embed
        texts = [c['text'] for c in chunks_metadata]
        embeddings = embedding_service.encode(texts)
        
        # 3. Add to Vector Store
        vector_store.add_documents(embeddings, chunks_metadata)
        
        documents_db[doc_id].status = "ready"
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing document {doc_id}: {error_msg}")
        if doc_id in documents_db:
            documents_db[doc_id].status = "failed"
            documents_db[doc_id].error = error_msg

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
    
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    doc_meta = DocumentMetadata(
        id=doc_id,
        filename=file.filename,
        status="pending",
        created_at=time.time()
    )
    documents_db[doc_id] = doc_meta
    
    background_tasks.add_task(process_document_background, doc_id, file_path, file.filename)
    
    return DocumentResponse(document_id=doc_id, status="pending", filename=file.filename)

@app.get("/status/{doc_id}")
async def get_status(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents_db[doc_id]
    return {
        "status": doc.status,
        "error": doc.error if doc.status == "failed" else None
    }

@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    return list(documents_db.values())

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from vector store
    vector_store.delete_document(doc_id)
    
    # Remove from in-memory db
    del documents_db[doc_id]
    
    return {"status": "deleted"}

@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_documents(request: QueryRequest, req: Any): # req needed for limiter
    # 1. Embed query
    query_vector = embedding_service.encode_query(request.question)
    
    # 2. Search vector store
    search_results = vector_store.search(query_vector, top_k=request.top_k, doc_id=request.document_id)
    
    if not search_results:
        return QueryResponse(
            answer="I couldn't find any relevant information in the uploaded documents.",
            sources=[]
        )
    
    # 3. Generate answer
    context_chunks = [r['text'] for r in search_results]
    answer = llm_service.generate_answer(request.question, context_chunks)
    
    # 4. Prepare sources
    sources = [
        SourceChunk(
            chunk_index=r['chunk_index'],
            score=r['score'],
            text=r['text'],
            filename=r['filename']
        ) for r in search_results
    ]
    
    return QueryResponse(answer=answer, sources=sources)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
