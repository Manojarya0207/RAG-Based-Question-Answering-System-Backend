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
        Now includes support for friendly, non-contextual conversation.
        """
        # --- Handle Greetings & General Chat in Mock Mode ---
        if self.mock_mode:
            q_lower = question.lower().strip()
            if any(greet in q_lower for greet in ["hi", "hello", "hey", "good morning", "good evening"]):
                return "Hello! I'm Antigravity, your intelligent assistant. How can I help you with your documents or questions today?"
            if "what is ai" in q_lower or "explain ai" in q_lower:
                return "**Artificial Intelligence (AI)** is a branch of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. This includes things like: \n\n* **Learning** (improving through data)\n* **Reasoning** (using rules to reach conclusions)\n* **Perception** (understanding visual or auditory inputs)\n\nIn our current setup, I use AI to analyze your documents and answer questions based on their specific content!"
            if not context_chunks and len(q_lower) < 50:
                 return f"I'm here to help! I don't see any specific documents related to '{question}' right now, but I can chat with you or analyze any files you upload."
            
            return f"Mock Answer: I have found information about '{question}' in the uploaded documents. The context contains {len(context_chunks)} relevant chunks. If this were a real query, I would provide a detailed analysis here."

        if not self.api_key:
            return "Error: OpenAI API key not configured. Please add it to your .env file."

        context_text = "\n\n---\n\n".join(context_chunks)
        
        # Refined System Prompt for ChatGPT-like Persona
        system_msg = (
            "You are Antigravity, a sophisticated AI assistant inspired by ChatGPT. "
            "You provide technical, clear, and helpful answers. \n\n"
            "GUIDELINES:\n"
            "1. If context is provided, prioritize it for the answer. "
            "2. If no context is provided or the user is just chatting (e.g., 'hi', 'how are you'), "
            "be conversational, warm, and helpful. Do not mention the lack of documents unless asked. \n"
            "3. Use Markdown for structure. Use bolding, headers, and lists to make your answers readable.\n"
            "4. For code, use triple backticks with language tags.\n"
            "5. Maintain a persona that is intelligent, objective, and supportive."
        )
        
        messages = [{"role": "system", "content": system_msg}]
        
        # Add history if provided
        if history:
            messages.extend(history[-8:]) # Increased history window for better flow
            
        # Add context if exists, otherwise treat as general query
        if context_chunks:
            user_content = f"CONTEXT FROM DOCUMENTS:\n{context_text}\n\nUSER QUESTION: {question}"
        else:
            user_content = question

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7, # Slightly higher for more natural chat
                max_tokens=800
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
