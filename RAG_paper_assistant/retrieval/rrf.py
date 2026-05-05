from retrieval.multi_query import generate_queries
from retrieval.hybrid_retrieval import bm25_retrieve
from core.config import BM25_WEIGHT,VECTOR_WEIGHT

rrf_k = 60

def hybrid_multiquery_rrf_retrieve(vectorstore, bm25, chunks, query, llm, multi_queries=3, vector_topk=3, bm25_topk=3):
  """将向量检索和BM25关键词检索结合起来，同时记录每个chunk的rrf分数"""
  # 1.生成多个query
  queries=generate_queries(llm,query,multi_queries)

  # 3.向量retriever
  retriever = vectorstore.as_retriever(search_kwargs={"k": vector_topk})

  # 4.汇总结果
  seen_contents={}
  for query in queries:
    vector_docs = retriever.invoke(query)
    bm25_docs = bm25_retrieve(query,llm,bm25,chunks,bm25_topk)

    # 5.去重+统计RRF分数
    # 先统计向量检索
    for i, doc in enumerate(vector_docs,1):
      content=doc.page_content.strip()
      doc_id=(doc.metadata.get("page",-1),content[:100])
      rrf_score1 = VECTOR_WEIGHT/(rrf_k+i)
      if doc_id not in seen_contents:
        seen_contents[doc_id]={"doc":doc,"count":0,"rrf_score":0}
      seen_contents[doc_id]["count"]+=1
      seen_contents[doc_id]["rrf_score"]+=rrf_score1

    # 统计BM25检索
    for i, doc in enumerate(bm25_docs,1):
      content=doc.page_content.strip()
      doc_id=(doc.metadata.get("page",-1),content[:100])
      rrf_score2 = BM25_WEIGHT/(rrf_k+i)
      if doc_id not in seen_contents:
        seen_contents[doc_id]={"doc":doc,"count":0,"rrf_score":0}
      seen_contents[doc_id]["count"]+=1
      seen_contents[doc_id]["rrf_score"]+=rrf_score2

  unique_docs = [item["doc"] for item in seen_contents.values()]
  return unique_docs, seen_contents