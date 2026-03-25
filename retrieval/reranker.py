from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):

        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=8):

        pairs = [(query, doc.content) for doc in documents]

        scores = self.model.predict(pairs)

        ranked_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, score in ranked_docs[:top_k]]