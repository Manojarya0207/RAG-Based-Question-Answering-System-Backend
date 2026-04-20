import os
import numpy as np
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initializes the EmbeddingService using OpenAI's API.
        """
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found for EmbeddingService")
        self.client = OpenAI(api_key=self.api_key)

    def encode(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Encodes a list of texts into embeddings using OpenAI API with batching.
        Returns a numpy array of shape (len(texts), 1536).
        """
        # Filter out empty or whitespace-only strings
        texts = [t if (t and t.strip()) else "empty chunk" for t in texts]
        
        if not texts:
            return np.array([]).reshape(0, 1536).astype('float32')
            
        all_embeddings = []
        
        # Process in batches to avoid API limits and handle large documents
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch {i//batch_size} with OpenAI: {str(e)}")
                raise e

        return np.array(all_embeddings).astype('float32')

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a 1D embedding vector.
        """
        if not query or not query.strip():
            query = "empty query"
            
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model
            )
            return np.array(response.data[0].embedding).astype('float32')
        except Exception as e:
            print(f"Error encoding query with OpenAI: {str(e)}")
            raise e
