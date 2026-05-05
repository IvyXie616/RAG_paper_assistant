import streamlit as st
import tempfile
import os
import sys

# 获取项目根目录（app的上一级）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from pipeline.rag_pipeline import RAGPipeline_ParentChild
from pipeline.rag_pipeline import RAGPipeline_Sections
from sentence_transformers import CrossEncoder
import core.config as config

# ===============================
# 页面基础配置
# ===============================
st.set_page_config(
    page_title="RAG论文助手",
    layout="wide"
)
st.title("📄 RAG论文智能问答助手")

# ===============================
# 初始化 Pipeline（缓存）
# ===============================
@st.cache_resource
def load_pipeline():
    reranker = CrossEncoder(config.RERANKER_MODEL)
    pipeline = RAGPipeline_ParentChild(reranker)
    return pipeline

pipeline = load_pipeline()

# ===============================
# 上传PDF
# ===============================
st.sidebar.header("📂 上传论文")

uploaded_files = st.sidebar.file_uploader(
    "上传PDF文件",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.header("🌐 网页输入")

url_inputs = st.sidebar.text_area(
    "输入一个或多个URL（每行一个）",
    placeholder="https://example1.com\nhttps://example2.com"
)

if uploaded_files is not None or url_inputs is not None:
    file_paths = []
    # 保存临时文件
    if uploaded_files:
        for i, file in enumerate(uploaded_files,1):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
                file_paths.append(tmp_path)

            st.sidebar.success(f"✅ 文件{i}上传成功")
    
    # 获取所有url
    url_inputs = url_inputs.split('\n')
    url_inputs = [url.strip() for url in url_inputs if url.strip()]

    # 构建索引按钮
    if st.sidebar.button("📦 构建索引"):
        with st.spinner("正在构建索引（第一次会较慢）..."):
            pipeline.build_index(file_paths=file_paths, urls=url_inputs)
        st.sidebar.success("✅ 索引构建完成！")

    # 清理临时文件（可选）# ✅ 构建完成后删除
    for path in file_paths:
        try:
            os.remove(path)
        except Exception as e:
            print(f"删除临时文件失败: {e}")# os.remove(tmp_path)


# ===============================
# 问答区域
# ===============================
st.header("💬 提问")

question = st.text_input("请输入你的问题：")

if question:
    if pipeline.vectorstore is None:
        st.warning("⚠️ 请先上传PDF并构建索引")
    else:
        with st.spinner("正在检索并生成答案..."):
            answer, docs = pipeline.query(question)

        # ===============================
        # 显示答案
        # ===============================
        st.subheader("📝 回答")
        st.write(answer)

        # ===============================
        # 显示引用来源
        # ===============================
        st.subheader("📚 引用来源")

        for i, doc in enumerate(docs, 1):
            with st.expander(f"📄 引用 {i} | 来源：{doc.metadata.get('source')} | 页码: {doc.metadata.get('page')} | Section: {doc.metadata.get('section')}"):
                st.write(doc.page_content[:800])  # 防止太长