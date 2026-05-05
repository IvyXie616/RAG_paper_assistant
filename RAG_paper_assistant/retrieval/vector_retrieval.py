def retrieve_docs(vectorstore, query, topk=5):
   """在向量库中检索与query对应向量相似度高的向量，以及对应的文本"""
   retriever = vectorstore.as_retriever(search_kwargs={"k": topk})
   docs = retriever.invoke(query)
   for i, doc in enumerate(docs):
    print(f"Chunk {i+1}")
    print(f"页码: {doc.metadata.get('page', '未知')}")
    print(doc.page_content[:300])
    print("-" * 50)
   return docs