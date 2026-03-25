from ingestion.youtube_loader import load_youtube_transcript
from ingestion.pdf_loader import load_pdf
from ingestion.image_loader import load_image

from preprocessing.cleaning import clean_documents
from preprocessing.chunking import chunk_documents

from embedding.embedding_model import EmbeddingModel

from vector_store.vector_db import VectorStore

from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import Reranker

from llm.answer_generator import AnswerGenerator


def build_knowledge_base(source_type, path):

    if source_type == "youtube":
        documents = load_youtube_transcript(path)

    elif source_type == "pdf":
        documents = load_pdf(path)

    elif source_type == "image":
        documents = load_image(path)

    else:
        raise ValueError("Unsupported source type")

    documents = clean_documents(documents)

    documents = chunk_documents(documents)

    return documents


def initialize_vector_store(documents):

    embedder = EmbeddingModel()

    embeddings = embedder.embed_documents(documents)

    dimension = len(embeddings[0])

    vector_db = VectorStore(dimension)

    vector_db.add_documents(documents, embeddings)

    return vector_db, embedder


def initialize_retrievers(vector_db, embedder, documents):

    dense_retriever = DenseRetriever(vector_db, embedder)

    bm25_retriever = BM25Retriever(documents)

    reranker = Reranker()

    return dense_retriever, bm25_retriever, reranker


def hybrid_retrieval(query, dense_retriever, bm25_retriever, reranker):

    dense_results = dense_retriever.retrieve(query)

    bm25_results = bm25_retriever.retrieve(query)

    fused_results = reciprocal_rank_fusion(
        [dense_results, bm25_results]
    )

    final_results = reranker.rerank(query, fused_results)

    return final_results


def main():

    print("Select input source")
    print("1. YouTube")
    print("2. PDF")
    print("3. Image")

    choice = input("Enter option: ")

    if choice == "1":
        source_type = "youtube"
        path = input("Enter YouTube URL: ")

    elif choice == "2":
        source_type = "pdf"
        path = input("Enter PDF file path: ")

    elif choice == "3":
        source_type = "image"
        path = input("Enter image path: ")

    else:
        print("Invalid option")
        return

    print("Building knowledge base...")

    documents = build_knowledge_base(source_type, path)

    print(f"Total processed chunks: {len(documents)}")

    vector_db, embedder = initialize_vector_store(documents)

    dense_retriever, bm25_retriever, reranker = initialize_retrievers(
        vector_db,
        embedder,
        documents
    )

    llm = AnswerGenerator()

    while True:

        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        retrieved_docs = hybrid_retrieval(
            query,
            dense_retriever,
            bm25_retriever,
            reranker
        )

        answer = llm.generate_answer(query, retrieved_docs)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()