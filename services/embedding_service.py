import torch
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.
        Returns a numpy array of shape (len(texts), embedding_dimension).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype('float32')

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a 1D embedding vector.
        """
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding.astype('float32')
