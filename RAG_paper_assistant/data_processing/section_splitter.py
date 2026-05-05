from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

def split_documents(documents, chunk_size=300, chunk_overlap=50, separators=['\nn', '\n', ',', '.', ';']):
    """分块：按字数分块，但有重叠"""
    text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=chunk_size,
       chunk_overlap=chunk_overlap,
       separators=separators
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

SECTION_PATTERNS = [
    r"(?i)^Abstract$",
    r"(?i)^\d*\.?\s*Abstract$",

    r"(?i)^\d*\.?\s*introduction$",

    r"(?i)^\d*\.?\s*background$",

    r"(?i)^\d*\.?\s*related work$",
    r"(?i)^\d*\.?\s*related works$",

    r"(?i)^\d*\.?\s*method$",
    r"(?i)^\d*\.?\s*methods$",
    r"(?i)^\d*\.?\s*methodology$",
    r"(?i)^\d*\.?\s*approach$",

    r"(?i)^\d*\.?\s*experiment$",
    r"(?i)^\d*\.?\s*experiments$",
    r"(?i)^\d*\.?\s*experimental setup$",
    r"(?i)^\d*\.?\s*evaluation$",

    r"(?i)^\d*\.?\s*result$",
    r"(?i)^\d*\.?\s*results$",

    r"(?i)^\d*\.?\s*discussion$",
    r"(?i)^\d*\.?\s*discussions$",

    r"(?i)^\d*\.?\s*conclusion$",
    r"(?i)^\d*\.?\s*conclusions$",

    r"(?i)^\d*\.?\s*limitation$",
    r"(?i)^\d*\.?\s*limitations$",

    r"(?i)^\d*\.?\s*future work$",
    
    r"(?i)^Reference$",
    r"(?i)^\d*\.?\s*reference$",
    r"(?i)^\d*\.?\s*references$"
]

def is_section_title(line):
    line = line.strip()

    for pattern in SECTION_PATTERNS:
        if re.match(pattern, line):
            return True

    return False

def split_by_sections(documents):
    """按章节分块
    documents:一个doc列表，每个doc表示文档的一页
    """
    sections=[] #用于存储总的章节信息
    current_title="Unknown" #当前章节名
    current_content=[] #当前章节内容
    current_source="Unknown" #当前读到的论文名

    page=-1
    for doc in documents:
        page=doc.metadata.get("page",-1)
        lines=doc.page_content.split('\n')
        current_source=doc.metadata.get("source","Unknown")

        if current_content and page>0:
        #即使是同一个章节的内容，处于不同页的话则用不同的字典表示
            sections.append({
                "source":current_source,
                "title":current_title,
                "content":"\n".join(current_content),
                "page":page-1
            })
            current_content=[] #清空上一页的内容

        for line in lines:
            s_line=line.strip()
            if not s_line:
                continue #遇到空行则跳过

            if is_section_title(s_line):
            #若这一行是标题
                if current_content:
                    sections.append({
                        "source":current_source,
                        "title":current_title,
                        "content":"\n".join(current_content),
                        "page":page
                    })
                current_title=s_line
                current_content=[]
            else:
                current_content.append(s_line)

    if current_content:
        sections.append({
            "source":current_source,
            "title":current_title,
            "content":"\n".join(current_content),
            "page":page
        })
    return sections

def chunk_sections(sections, chunk_size=500, chunk_overlap=50):
  """用按照章节分出的sections创建chunks"""
  splitter = RecursiveCharacterTextSplitter( chunk_size=chunk_size, chunk_overlap=chunk_overlap )

  chunked_docs = []

  for sec in sections:
    chunks=splitter.split_text(sec["content"])

    for chunk in chunks:
      doc=Document(
          page_content=chunk,
          metadata={
              "source":sec["source"],
              "section":sec["title"],
              "page":sec["page"]
          }
      )
      chunked_docs.append(doc)
  return chunked_docs