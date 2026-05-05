def generate_answer(llm, docs, query, max):
  """利用初始化的LLM模型生成答案"""
  article_contents=""
  for i, doc in enumerate(docs,1):
    article_contents=article_contents+f"文本{i} | 来源文章:{doc.metadata.get('source')} |{doc.page_content[:max]}\n"

  references = []
  for i, doc in enumerate(docs):
    ref = {
        "page": doc.metadata.get("page", "未知"),
        "content": doc.page_content[:200]
    }
    references.append(ref)

  prompt=f"""
  你是一个论文解读小助手，你需要基于以下来源于一篇或多篇论文片段文本回答用户的问题。

  论文文本：
  {article_contents}

  用户问题:
  {query}

  生成回复的要求：
  - 只基于提供的论文文本回答
  - 如果没有相关信息，请礼貌地回答自己不知道
  """
  response=llm.invoke(prompt)
  print(f"最终prompt长度{len(prompt)}")
  return response.content.strip()