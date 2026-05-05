def rerank_documents(query, seen_contents, reranker, top_n=5, count_weight=0.1, rrf_weight=1):
    """
    query: 用户原始问题
    seen_contents: multi_query_retrieve返回的seen_contents
    reranker: CrossEncoder模型
    top_n: 最终保留多少个chunk
    count_weight: chunk出现次数的权重
    """

    # 1. 准备query-doc对
    doc_items=list(seen_contents.values())
    pairs = []
    for item in doc_items:
        doc = item["doc"]
        pairs.append([query, doc.page_content])

    # 2. CrossEncoder打分
    rerank_scores=reranker.predict(pairs)

    # 3. 融合出现次数和rrf得分
    ranked_results = []

    for item, rerank_score in zip(doc_items, rerank_scores):
        doc = item["doc"]
        count = item["count"]
        if "rrf_score" not in item:
            item["rrf_score"] = 0
        rrf_score = item["rrf_score"]

        # 融合得分
        final_score = float(rerank_score) + count_weight * count + rrf_weight * rrf_score

        ranked_results.append({
            "doc": doc,
            "count": count,
            "rerank_score": float(rerank_score),
            "final_score": final_score
        })

    # 4. 排序
    ranked_results = sorted(
        ranked_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    # 5. 打印调试信息
    print("\nRerank结果：")

    for i, item in enumerate(ranked_results[:top_n], 1):
        doc = item["doc"]

        print(f"\nTop {i}")
        print(f"页码: {doc.metadata.get('page', '未知')}")
        print(f"出现次数: {item['count']}")
        print(f"Rerank分数: {item['rerank_score']:.4f}")
        print(f"最终分数: {item['final_score']:.4f}")
        print(doc.page_content[:300])
        print("-" * 60)

    # 6. 返回最终文档
    final_docs = [item["doc"] for item in ranked_results[:top_n]]

    return final_docs, ranked_results