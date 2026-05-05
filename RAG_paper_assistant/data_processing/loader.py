from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
import re

def clean_text(text: str) -> str:
    # 1. 去掉多余空白
    text = re.sub(r"\s+", " ", text)

    # 2. 修复断词（informa- tion → information）
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    # 3. 去掉页码（如 "Page 1" 或单独数字行）
    text = re.sub(r"\bPage \d+\b", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # 4. 去掉常见页眉（arXiv / conference）
    text = re.sub(r"arXiv:\S+", "", text)
    text = re.sub(r"Proceedings of.*", "", text)

    # 5. 去掉多余换行
    text = re.sub(r"\n+", "\n", text)

    return text.strip()

def load_pdf(file_path):
    """将准备好的pdf文件转换为doc文件，即文本"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        file_name = str(file_path).split('\\')[-1]
        doc.metadata["source"]=file_name.strip()
    return documents

def load_multi_pdfs(file_paths):
    """处理多个文件，并把它们放在同一个数据库内"""
    all_docs = []
    for path in file_paths:
        docs = load_pdf(path)
        all_docs.extend(docs)
    return all_docs

def load_web(url):
    """通过url加载web的文档"""
    loader = WebBaseLoader(url)
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"]=url
    return documents

def load_multi_urls(urls):
    """处理多个url"""
    all_docs = []
    for url in urls:
        docs = load_web(url)
        all_docs.extend(docs)
    return all_docs