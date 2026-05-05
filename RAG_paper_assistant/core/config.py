# 文本分块参数
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
SEPARATORS = ['\n\n', '\n', ',', '.', ';']

PARENT_CHUNK_SIZE = 400
PARENT_OVERLAP = 100
CHILD_CHUNK_SIZE = 100
CHILD_OVERLAP = 30

MODEL_NAME = "qwen-turbo" #用于生成的LLM模型
EMBEDDING_MODEL = "text-embedding-v3" #用于embedding建立向量库的模型
RERANKER_MODEL = "BAAI/bge-reranker-base" #用于CrossEncoder,建立reranker的模型

TOP_K = 5 #最终检索选出的，用于生成答复的文本数
MAX_CONTEXT = 800 #最终放入prompt的文本的长度

# multi-query检索参数，bybrid检索参数
MULTI_QUERIES = 4 #一个query转换生成的多个query数量
VECTOR_TOPK = 4 #对于每个query，向量检索出VECTOR_TOPK个doc
BM25_TOPK = 4 #对于每个query，BM25关键词检索出BM25_TOPK个doc
COUNT_WEIGHT = 0.0 #某一文档在检索时出现次数在最终排序时的权重
RRF_WEIGHT = 0.8

BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5