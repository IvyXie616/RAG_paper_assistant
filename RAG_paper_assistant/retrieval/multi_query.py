def generate_queries(llm,query,multi):
  """基于用户的原问题，生成多个和原问题相关的queries，有助于答案完善"""
  prompt=f"""
  你是一个论文检索助手。
  请针对下面的问题，生成{multi}个语义相近但表达不同的检索问题，以帮助用户更完善地匹配到、并检索论文内容。

  原始问题：{query}

  要求：
  - 每行一个问题
  - 每个问题需要关注不同的角度，如：方法、实验、贡献、局限性
  - 不要编号
  - 不要解释
  """
  queries=llm.invoke(prompt)
  contents=queries.content.strip()
  multi_queries=[line.strip() for line in contents.split("\n") if line.strip()]
  return multi_queries

def multi_query_retrieve(vectorstore, query, llm, multi_queries=3, topk=3):
    # 1. 生成多个query
    generated_queries = generate_queries(llm, query, multi_queries)

    print("生成的检索问题：")
    for i, q in enumerate(generated_queries, 1):
        print(f"{i}. {q}")

    # 2. 建立retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": topk}
    )

    # 3. 对每个query分别检索
    all_docs = []

    for q in generated_queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # 4. 去重
    seen_contents = {}

    # 并记录chunks出现次数
    for doc in all_docs:
      content = doc.page_content.strip()
      doc_id=(doc.metadata.get("page","未知"),content[:100])
      if doc_id not in seen_contents:
        seen_contents[doc_id]={"doc":doc,"count":0}
      seen_contents[doc_id]["count"]+=1

    unique_docs = [item["doc"] for item in seen_contents.values()]

    print(f"\n原始检索结果数：{len(all_docs)}")
    print(f"去重后结果数：{len(unique_docs)}")

    # 5. 打印检索结果（调试用）
    for i, doc in enumerate(unique_docs):
      print(f"\nChunk {i+1}")
      print(f"页码: {doc.metadata.get('page', '未知')}")
      print(doc.page_content[:300])
      print("-" * 50)

    return unique_docs, seen_contents