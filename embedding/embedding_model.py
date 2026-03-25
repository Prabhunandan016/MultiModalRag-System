from sentence_transformers import SentenceTransformer
from typing import List
from ingestion.document_schema import Document


class EmbeddingModel:

    def __init__(self, model_name: str = "BAAI/bge-small-en"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        # BGE models perform better with this query prefix
        return self.model.encode("Represent this sentence for searching relevant passages: " + text)

    def embed_documents(self, documents: List[Document]):
        texts = [doc.content for doc in documents]
        return self.model.encode(texts, batch_size=64, show_progress_bar=True)
