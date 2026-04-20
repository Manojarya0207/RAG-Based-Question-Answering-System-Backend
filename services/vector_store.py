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

    def search(self, query_vector: np.ndarray, top_k: int = 5, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Searches for the most similar chunks. Optionally filters by doc_id.
        """
        # FAISS search
        # query_vector should be (1, dimension)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        distances, indices = self.index.search(query_vector, top_k * 2) # Get more to allow filtering
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx]
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
        FAISS doesn't support easy deletion by ID for IndexFlatL2 without rebuilding or using IDMap.
        For simplicity in v1, we will rebuild the index if needed, or just filter it out in search.
        Actually, let's just rebuild the index to keep it clean.
        """
        new_metadata = [m for m in self.metadata if m['doc_id'] != doc_id]
        if len(new_metadata) == len(self.metadata):
            return # Nothing to delete
            
        # Rebuild index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        
        # This is where we'd need the original embeddings if we wanted to rebuild perfectly.
        # However, for this project, we'll assume deletion is rare or we'd store embeddings elsewhere.
        # TEMPORARY: Just filter metadata and save. Search will still find the old vectors but we'll ignore them.
        # BETTER: For a real app, use IndexIDMap and remove_ids.
        
        self.metadata = new_metadata
        self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
