import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # We don't raise error here yet, but will fail on call if not set
            print("Warning: OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=self.api_key)

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generates an answer using OpenAI GPT based on the provided context.
        """
        if not self.api_key:
            return "Error: OpenAI API key not configured. Please add it to your .env file."

        context_text = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer this question."

Context:
{context_text}

Question: {question}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a specialized RAG assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with OpenAI: {str(e)}"
