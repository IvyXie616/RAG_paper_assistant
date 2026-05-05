from rank_bm25 import BM25Okapi
import jieba
import re

def detect_language(text):
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        return "zh"
    return "en"

def tokenize(text):
    # 检测所有英文单词
    words = re.findall(r"\b\w+\b", text.lower())

    # 检测所有中文汉字
    chinese = list(jieba.cut(text))
    return words+chinese

def build_bm25_index(chunks):
    """建立BM25关键词索引字典"""
    corpus = []
    for doc in chunks:
        sec =  doc.metadata["section"]
        if "reference" in sec.lower():
            break
        corpus.append(doc.page_content)

    tokenized_corpus = [ tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, corpus
