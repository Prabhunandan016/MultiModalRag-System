import os
import html
import logging
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 3000
MAX_QUERY_CHARS   = 500
MAX_TOKENS        = 512


class AnswerGenerator:

    def __init__(self):
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")

        self.client = Groq(api_key=api_key)
        self.model  = "llama-3.1-8b-instant"

    def generate_answer(self, query: str, documents: list) -> str:
        if not query or not query.strip():
            return "Please enter a valid question."

        # Sanitize and truncate inputs
        safe_query   = query.strip()[:MAX_QUERY_CHARS]
        context_parts = []
        total = 0
        for doc in documents:
            chunk = doc.content.strip()
            if total + len(chunk) > MAX_CONTEXT_CHARS:
                chunk = chunk[:MAX_CONTEXT_CHARS - total]
                context_parts.append(chunk)
                break
            context_parts.append(chunk)
            total += len(chunk)

        context = "\n\n".join(context_parts)

        if not context:
            return "No relevant content found to answer your question."

        prompt = (
            "You are a helpful assistant. Use only the context below to answer the question. "
            "If the answer is not in the context, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {safe_query}\n\n"
            "Answer:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.2,
                timeout=30,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", type(e).__name__)
            return "Failed to generate an answer. Please try again."
