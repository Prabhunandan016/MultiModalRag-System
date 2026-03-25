from rank_bm25 import BM25Okapi
from typing import List
from ingestion.document_schema import Document


class BM25Retriever:

    def __init__(self, documents: List[Document]):

        self.documents = documents

        corpus = [doc.content.lower().split() for doc in documents]

        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query, top_k=10):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [self.documents[idx] for idx in ranked_indices[:top_k]]