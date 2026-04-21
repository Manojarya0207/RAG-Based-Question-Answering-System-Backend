import os
import uuid
import PyPDF2
from typing import List, Dict, Any
import time

class IngestionService:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
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
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {str(e)}")
            
        if not text.strip():
            raise ValueError("No text could be extracted from this PDF. It might be a scanned image or protected.")
        return text

    def _extract_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks using a recursive character splitting strategy.
        This respects paragraph and sentence boundaries better than simple word splits.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        # Use the next separator
        if not separators:
            return [text[0:self.chunk_size]] # Force split if no separators left

        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = text.split(separator)
        final_chunks = []
        current_chunk = ""

        for split in splits:
            # If adding this split exceeds chunk_size
            if len(current_chunk) + len(separator) + len(split) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk)
                
                # If the split itself is too large, recurse on it
                if len(split) > self.chunk_size:
                    final_chunks.extend(self._recursive_split(split, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = split
            else:
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split

        if current_chunk:
            final_chunks.append(current_chunk)
        
        # Handle overlap (simplified)
        # For true recursive splitting with overlap, usually a more complex windowing is needed.
        # This implementation prioritizes clean breaks over perfect overlap.
        return final_chunks

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
