import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class AnswerGenerator:

    def __init__(self):
        # use Streamlit secrets on cloud, fallback to .env locally
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Add it to .env locally or Streamlit secrets on cloud.")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def generate_answer(self, query, documents):
        context = "\n".join(doc.content for doc in documents)[:3000]

        prompt = f"""Use the following context to answer the question concisely.

Context:
{context}

Question:
{query}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )

        return response.choices[0].message.content
