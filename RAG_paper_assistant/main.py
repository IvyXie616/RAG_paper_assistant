from pipeline.rag_pipeline import RAGPipeline_Sections
from pipeline.rag_pipeline import RAGPipeline_ParentChild
from sentence_transformers import CrossEncoder
import core.config as config

reranker = CrossEncoder(config.RERANKER_MODEL)

pipeline = RAGPipeline_ParentChild(reranker)
pipeline2 = RAGPipeline_Sections(reranker)
pipeline.build_index(urls=["https://sai.sysu.edu.cn/basic/393"])

answer, docs = pipeline.query("中山大学人工智能学院三大研究方向是什么？")

print("\n📝 回答：")
print(answer)

print("\n📄 引用来源：")
for doc in docs:
    print("="*60)
    print("章节:",doc.metadata["section"])
    print("页码:",doc.metadata["page"])
    print("原文（节选）:",doc.page_content[:200])
    print("="*60)