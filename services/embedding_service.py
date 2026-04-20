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
        This significantly reduces local RAM usage compared to local models.
        """
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found for EmbeddingService")
        self.client = OpenAI(api_key=self.api_key)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of texts into embeddings using OpenAI API.
        Returns a numpy array of shape (len(texts), 1536).
        """
        if not texts:
            return np.array([]).reshape(0, 1536).astype('float32')
            
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings).astype('float32')
        except Exception as e:
            print(f"Error encoding with OpenAI: {str(e)}")
            # Return zero vectors if failed to avoid crashing the pipeline entirely, 
            # though usually it's better to let it fail or handle in calling service.
            raise e

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a 1D embedding vector.
        """
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model
            )
            return np.array(response.data[0].embedding).astype('float32')
        except Exception as e:
            print(f"Error encoding query with OpenAI: {str(e)}")
            raise e
