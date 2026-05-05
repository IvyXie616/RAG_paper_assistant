from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

def build_embedding(EMBEDDING_MODEL):
   embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)
   return embeddings

def build_vectorstore(chunks, EMBEDDING):
   """使用FAISS库创建向量数据库,将之前的分块转换为向量，并在向量库中建立索引"""
   vectorstore = FAISS.from_documents(chunks, EMBEDDING)
   return vectorstore