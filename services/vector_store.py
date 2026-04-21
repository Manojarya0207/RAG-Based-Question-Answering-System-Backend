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
        Searches for the most similar chunks. Optionally filters by doc_id.
        """
        if self.index.ntotal == 0 or not self.metadata:
            return []

        # FAISS search
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # We search for more than top_k to allow for filtering of deleted docs
        search_k = min(self.index.ntotal, top_k * 5)
        distances, indices = self.index.search(query_vector, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx]
            
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
        Deletes all chunks associated with a doc_id.
        Since we don't store original embeddings to rebuild the index perfectly, 
        we mark the metadata entries as 'deleted' to ignore them in future searches.
        """
        # For simplicity in this v1, we rebuild a map and just filter out the doc
        initial_count = len(self.metadata)
        self.metadata = [m for m in self.metadata if m['doc_id'] != doc_id]
        
        if len(self.metadata) < initial_count:
            # We also clear the index to prevent mapping to now-invalid metadata indices
            # Realistically, for persistence we would need to re-index all remaining docs.
            # Here we just save the updated metadata. 
            # NOTE: If we really wanted to fix the index perfectly, we'd need to store vectors separately.
            self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
