import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class AnswerGenerator:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def generate_answer(self, query, documents):
        # cap context to avoid slow/expensive large prompts
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
