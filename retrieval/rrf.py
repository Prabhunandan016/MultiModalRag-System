def reciprocal_rank_fusion(result_lists, k=60):

    scores = {}
    doc_map = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.content[:100]  # use content as stable key

            if doc_id not in scores:
                scores[doc_id] = 0
                doc_map[doc_id] = doc

            scores[doc_id] += 1 / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    return [doc_map[doc_id] for doc_id in sorted_ids]
