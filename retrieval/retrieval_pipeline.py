from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.reranker import Reranker


class HybridRetriever:

    def __init__(self, dense_retriever, bm25_retriever):

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = Reranker()

    def retrieve(self, query, top_k=5):

        # dense retrieval
        dense_results = self.dense_retriever.retrieve(query, top_k=10)

        # bm25 retrieval
        bm25_results = self.bm25_retriever.retrieve(query, top_k=10)

        # fuse rankings
        fused_results = reciprocal_rank_fusion(
            [dense_results, bm25_results]
        )

        # rerank results
        final_results = self.reranker.rerank(query, fused_results, top_k)

        return final_results