import logging
import chromadb
from typing import List
from ingestion.document_schema import Document

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self):
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("rag_collection")
        except ValueError:
            pass
        self.collection = self.client.create_collection(name="rag_collection")

    def add_documents(self, documents: List[Document], embeddings):
        if not documents:
            raise ValueError("No documents to add.")

        ids       = [str(i) for i in range(len(documents))]
        texts     = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info("Added %d documents to vector store.", len(documents))

    def search(self, query_embedding, top_k: int = 5) -> List[Document]:
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        n = min(top_k, self.collection.count())
        if n == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n
        )

        return [
            Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            )
            for i in range(len(results["documents"][0]))
        ]
