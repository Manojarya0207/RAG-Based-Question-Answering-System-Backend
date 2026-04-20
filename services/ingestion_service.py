import os
import uuid
import PyPDF2
from typing import List, Dict, Any
import time

class IngestionService:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, file_path: str) -> str:
        """
        Extracts text from PDF or TXT files.
        """
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif ext.lower() == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_from_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into overlapping chunks.
        """
        # Split by words for simplicity, roughly approximating tokens
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap
            
        return chunks

    def process_document(self, file_path: str, doc_id: str, filename: str) -> List[Dict[str, Any]]:
        """
        Full pipeline: extract -> chunk -> metadata preparation.
        """
        text = self.extract_text(file_path)
        chunks = self.chunk_text(text)
        
        chunks_metadata = []
        for idx, chunk in enumerate(chunks):
            chunks_metadata.append({
                "text": chunk,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": idx
            })
            
        return chunks_metadata
