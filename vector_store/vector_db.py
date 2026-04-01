import chromadb
from typing import List
from ingestion.document_schema import Document


class VectorStore:

    def __init__(self):
        self.client = chromadb.Client()

        try:
            self.client.delete_collection("multimodal_rag_collection")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="multimodal_rag_collection"
        )

    def add_documents(self, documents: List[Document], embeddings):
        ids = [str(i) for i in range(len(documents))]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k=5):
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return [
            Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            )
            for i in range(len(results["documents"][0]))
        ]
