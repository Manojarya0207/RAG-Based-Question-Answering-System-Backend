from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentResponse(BaseModel):
    document_id: str
    status: str
    filename: str

class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)

class SourceChunk(BaseModel):
    chunk_index: int
    score: float
    text: str
    filename: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    status: str # pending, processing, ready, failed
    created_at: float
