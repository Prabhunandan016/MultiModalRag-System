import chromadb
from typing import List
from ingestion.document_schema import Document


class VectorStore:

    def __init__(self, persist_directory="database/chroma_db"):

        self.client = chromadb.PersistentClient(path=persist_directory)

        # delete and recreate to avoid stale data on reload
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

        # convert numpy array to list for chromadb
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        print(f"Stored {len(documents)} documents in vector database")


    def search(self, query_embedding, top_k=5):

        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_docs = []

        for i in range(len(results["documents"][0])):

            doc = Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            )

            retrieved_docs.append(doc)

        return retrieved_docs