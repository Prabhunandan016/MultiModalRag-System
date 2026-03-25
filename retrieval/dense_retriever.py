from embedding.embedding_model import EmbeddingModel
from vector_store.vector_db import VectorStore


class DenseRetriever:

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel):

        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query, top_k=10):

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.search(query_embedding, top_k)

        return results