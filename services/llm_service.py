import os
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        
        if self.mock_mode:
            print("LLMService running in MOCK_MODE")
        elif not self.api_key:
            # We don't raise error here yet, but will fail on call if not set
            print("Warning: OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=self.api_key)

    def generate_answer(self, question: str, context_chunks: List[str], history: List[Dict[str, str]] = None) -> str:
        """
        Generates an answer using OpenAI GPT based on the provided context and history.
        """
        if self.mock_mode:
            return f"Mock Answer: I have found information about '{question}' in the uploaded documents. The context contains {len(context_chunks)} relevant chunks."

        if not self.api_key:
            return "Error: OpenAI API key not configured. Please add it to your .env file."

        context_text = "\n\n---\n\n".join(context_chunks)
        
        messages = [
            {"role": "system", "content": "You are Antigravity, a helpful and friendly RAG assistant. Your primary goal is to answer questions using the provided document context. If the user greets you or asks general conversational questions, be warm and engaging. If you cannot find the answer in the context but the query is a general knowledge question, feel free to use your own knowledge to help, but mention when the information is not from the uploaded documents."}
        ]
        
        # Add history if provided (limit to last 5 messages for token efficiency)
        if history:
            messages.extend(history[-5:])
            
        # Add context and current question
        if context_chunks:
            prompt = f"""Use the following pieces of context to answer the user's question. 
Context:
{context_text}

Question: {question}

Answer:"""
        else:
            prompt = question # Fallback to direct query for friendly chat if no context exists

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.2,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with OpenAI: {str(e)}"

    def summarize_document(self, text: str) -> str:
        """
        Generates a concise summary of the document text.
        """
        if self.mock_mode:
            return "Mock Summary: This document discusses key aspects of the RAG system and its implementation details."

        if not self.api_key:
            return "Error: OpenAI API key not configured."

        # Take first 4000 characters for summary to avoid token limits
        sample_text = text[:4000]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes documents concisely."},
                    {"role": "user", "content": f"Summarize the following document in one concise paragraph:\n\n{sample_text}"}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"
