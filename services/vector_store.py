import faiss
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self, index_path: str = "data/faiss_index.bin", metadata_path: str = "data/metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = [] # List of { "text": str, "doc_id": str, "filename": str, "chunk_index": int }

    def add_documents(self, embeddings: np.ndarray, chunks_metadata: List[Dict[str, Any]]):
        """
        Adds embeddings and their corresponding metadata to the store.
        """
        self.index.add(embeddings)
        self.metadata.extend(chunks_metadata)
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict[ Any, Any]]:
        """
        Searches for the most similar chunks.
        Filters out markers for deleted documents.
        """
        if self.index.ntotal == 0 or not self.metadata:
            return []

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Search for enough results to account for deleted items
        search_k = min(self.index.ntotal, top_k * 10)
        distances, indices = self.index.search(query_vector, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx]
            
            # Skip deleted chunks
            if meta.get('deleted'):
                continue
                
            # If doc_id filter is provided, skip mismatches
            if doc_id and meta['doc_id'] != doc_id:
                continue
                
            results.append({
                "text": meta['text'],
                "doc_id": meta['doc_id'],
                "filename": meta['filename'],
                "chunk_index": meta['chunk_index'],
                "score": float(dist)
            })
            
            if len(results) >= top_k:
                break
                
        return results

    def delete_document(self, doc_id: str):
        """
        Marks all chunks associated with a doc_id as deleted.
        This keeps the metadata list indices stable and aligned with the FAISS index.
        """
        found = False
        for meta in self.metadata:
            if meta['doc_id'] == doc_id:
                meta['deleted'] = True
                found = True
        
        if found:
            self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
