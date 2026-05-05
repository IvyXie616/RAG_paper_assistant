from data_processing.loader import load_pdf, load_multi_pdfs, load_multi_urls
from data_processing.section_splitter import split_by_sections, chunk_sections
from data_processing.parent_child_chunk import (
    build_parent_chunks,
    build_child_chunks,
    build_parent_map,
    parent_child_chunk_sections
)
from data_processing.bm25_index import build_bm25_index

from core.vectorstore import build_vectorstore, build_embedding
from core.llm import get_llm

from retrieval.rrf import hybrid_multiquery_rrf_retrieve
from retrieval.rerank import rerank_documents
from retrieval.parent_retrieval import retrieve_parent_docs
import core.config as config
from pipeline.generator import generate_answer

class RAGPipeline_ParentChild:
    def __init__(self, reranker):
        self.llm = get_llm(config.MODEL_NAME)
        self.reranker = reranker
        self.embedding = build_embedding(config.EMBEDDING_MODEL)
        self.vectorstore = None
        self.bm25 = None
        self.chunks = None
        self.parent_map = None
    
    # ===============================
    # Step 1: 构建索引（只做一次）
    # ===============================
    def build_index(self, file_paths=None, urls=None):
        print("📄 加载文档...")
        pdf_documents = []
        web_documents = []
        if file_paths:
            if isinstance(file_paths,list):
                pdf_documents = load_multi_pdfs(file_paths)
            else:
                pdf_documents = load_pdf(file_paths)

        if urls:
            web_documents = load_multi_urls(urls)
        documents = pdf_documents+web_documents

        print("🔍 按章节切分...")
        sections = split_by_sections(documents)

        print("🧱 构建 Parent chunks, Child chunks, Parent map...")
        parent_docs, child_docs, parent_map = parent_child_chunk_sections(
            sections,
            config.PARENT_CHUNK_SIZE,
            config.PARENT_OVERLAP,
            config.CHILD_CHUNK_SIZE,
            config.CHILD_OVERLAP
        )
        self.parent_map = parent_map

        print("📦 构建向量库（Child）...")
        self.vectorstore = build_vectorstore(child_docs, self.embedding)

        print("📚 构建 BM25 索引...")
        self.bm25, _ = build_bm25_index(child_docs)
        self.chunks = child_docs

        print("✅ 索引构建完成！")
    
    # ===============================
    # Step 2: 查询
    # ===============================
    def query(self, question, top_k=config.TOP_K):
        if self.vectorstore is None:
            raise ValueError("请先调用 build_index()")

        print(f"\n🧠 用户问题: {question}")
        # ---------------------------
        # 1. Hybrid + MultiQuery + rrf
        # ---------------------------
        child_docs, seen_contents = hybrid_multiquery_rrf_retrieve(
            vectorstore=self.vectorstore,
            bm25=self.bm25,
            chunks=self.chunks,
            query=question,
            llm=self.llm,
            multi_queries=config.MULTI_QUERIES,
            vector_topk=config.VECTOR_TOPK,
            bm25_topk=config.BM25_TOPK
        )

        # ---------------------------
        # 2. Rerank（CrossEncoder）
        # ---------------------------
        reranked_child_docs, ranked_results = rerank_documents(
            query=question,
            seen_contents=seen_contents,
            reranker=self.reranker,
            top_n=top_k,
            count_weight=config.COUNT_WEIGHT,
            rrf_weight=config.RRF_WEIGHT
        )

        # ---------------------------
        # 3. Child → Parent
        # ---------------------------
        parent_docs = retrieve_parent_docs(
            reranked_child_docs,
            self.parent_map
        )

        # 控制数量（非常重要）
        parent_docs = parent_docs[:top_k]

        # ---------------------------
        # 4. 生成答案
        # ---------------------------
        answer = generate_answer(self.llm, parent_docs, question, config.MAX_CONTEXT)

        return answer, parent_docs
    
class RAGPipeline_Sections:
    def __init__(self,reranker):
        self.llm = get_llm(config.MODEL_NAME)
        self.reranker = reranker
        self.embedding = build_embedding(config.EMBEDDING_MODEL)
        self.vectorstore = None
        self.bm25 = None
        self.chunks = None
    
    # ===============================
    # Step 1: 构建索引（只做一次）
    # ===============================
    def build_index(self, file_paths=None, urls=None):
        print("📄 加载PDF...")
        pdf_documents = []
        web_documents = []
        if file_paths:
            if isinstance(file_paths,list):
                pdf_documents = load_multi_pdfs(file_paths)
            else:
                pdf_documents = load_pdf(file_paths)

        if urls:
            web_documents = load_multi_urls(urls)
        documents = pdf_documents+web_documents
        print("🔍 按章节切分...")
        sections = split_by_sections(documents)

        print("🧱 在每个章节中滑块分割")
        docs = chunk_sections(sections, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print("📦 构建向量库（Child）...")
        self.vectorstore = build_vectorstore(docs, self.embedding)

        print("📚 构建 BM25 索引...")
        self.bm25, _ = build_bm25_index(docs)
        self.chunks = docs

        print("✅ 索引构建完成！")
    
    # ===============================
    # Step 2: 查询
    # ===============================
    def query(self, question, top_k=config.TOP_K):
        if self.vectorstore is None:
            raise ValueError("请先调用 build_index()")

        print(f"\n🧠 用户问题: {question}")
        # ---------------------------
        # 1. Hybrid + MultiQuery + rrf
        # ---------------------------
        _, seen_contents = hybrid_multiquery_rrf_retrieve(
            vectorstore=self.vectorstore,
            bm25=self.bm25,
            chunks=self.chunks,
            query=question,
            llm=self.llm,
            multi_queries=config.MULTI_QUERIES,
            vector_topk=config.VECTOR_TOPK,
            bm25_topk=config.BM25_TOPK
        )

        # ---------------------------
        # 2. Rerank（CrossEncoder）
        # ---------------------------
        reranked_docs, ranked_results = rerank_documents(
            query=question,
            seen_contents=seen_contents,
            reranker=self.reranker,
            top_n=top_k,
            count_weight=config.COUNT_WEIGHT,
            rrf_weight=config.RRF_WEIGHT
        )

        # ---------------------------
        # 3. 生成答案
        # ---------------------------
        answer = generate_answer(self.llm, reranked_docs, question, config.MAX_CONTEXT)

        return answer, reranked_docs