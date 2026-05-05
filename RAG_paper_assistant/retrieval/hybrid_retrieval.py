from retrieval.multi_query import generate_queries
import data_processing.bm25_index as BM25
from data_processing.bm25_index import tokenize

def translate_query_to_English(llm, query):
  """将中文的query翻译为中英双语"""
  prompt=f"""
  将用户输入的问题翻译成适合英文论文检索的简洁英文问题。

  问题：{query}

  要求：
  - 保留原意
  - 尽量使用论文中的常见术语
  - 只输出英文翻译
  """

  translate=llm.invoke(prompt)
  english_query = translate.content.strip()
  return english_query

def bm25_retrieve(query, llm, bm25, chunks, topk=3):
  """BM25索引检索"""
  en_query = translate_query_to_English(llm, query)
  query = query+en_query

  tokenized_query = tokenize(query)
  scores = bm25.get_scores(tokenized_query)

  ranked_indices = sorted(
      range(len(scores)),
      key=lambda i: scores[i],
      reverse=True
  )[:topk]

  result_docs = [chunks[i] for i in ranked_indices]

  return result_docs

def hybrid_multi_query_retrieve(vectorstore, bm25, chunks, query, llm, multi_queries=3, vector_topk=3, bm25_topk=3):
  """将向量检索和BM25关键词检索结合起来"""
  # 1.生成多个query
  queries=generate_queries(llm,query,multi_queries)

  # 3.向量retriever
  retriever = vectorstore.as_retriever(search_kwargs={"k": vector_topk})

  # 4.汇总结果
  all_docs=[]
  for query in queries:
    vector_docs = retriever.invoke(query)
    bm25_docs=bm25_retrieve(query,llm,bm25,chunks,bm25_topk)
    all_docs.extend(vector_docs)
    all_docs.extend(bm25_docs)

  # 5.去重
  seen_contents={}
  for doc in all_docs:
    content = doc.page_content.strip()
    doc_id=(doc.metadata.get("page","未知"),content[:100])
    if doc_id not in seen_contents:
      seen_contents[doc_id]={"doc":doc,"count":0}
    seen_contents[doc_id]["count"]+=1
  unique_docs = [item["doc"] for item in seen_contents.values()]
  print(f"\n原始检索结果数：{len(all_docs)}")
  print(f"去重后结果数：{len(unique_docs)}")

  # 6.打印检索结果（调试用）
  for i, doc in enumerate(unique_docs):
    print(f"\nChunk {i+1}")
    print(f"页码: {doc.metadata.get('page', '未知')}")
    print(doc.page_content[:300])
    print("-" * 50)

  return unique_docs, seen_contents