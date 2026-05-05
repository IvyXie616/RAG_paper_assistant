from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import uuid

def build_parent_chunks(sections, parent_chunk_size=800, parent_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap
    )
    parent_docs = []

    for section in sections:
        chunks = splitter.split_text(section["content"])

        for chunk in chunks:
            parent_id = str(uuid.uuid4())

            parent_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": section["source"],
                        "section": section["title"],
                        "page": section["page"],
                        "parent_id": parent_id
                    }
                )
            )

    return parent_docs


def build_child_chunks(parent_docs, child_chunk_size=200, child_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap
    )
    child_docs = []

    for parent in parent_docs:
        child_chunks = splitter.split_text(parent.page_content)

        for chunk in child_chunks:
            child_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": parent.metadata["source"],
                        "parent_id": parent.metadata["parent_id"],
                        "section": parent.metadata["section"],
                        "page": parent.metadata["page"]
                    }
                )
            )

    return child_docs

def build_parent_map(parent_docs):
    parent_map = {}
    for doc in parent_docs:
        parent_id = doc.metadata["parent_id"]
        parent_map[parent_id] = doc

    return parent_map

def parent_child_chunk_sections(sections, parent_chunk_size=800, parent_overlap=100, child_chunk_size=200, child_overlap=50):
    parent_docs=build_parent_chunks(sections, parent_chunk_size, parent_overlap)
    child_docs=build_child_chunks(parent_docs, child_chunk_size, child_overlap)
    parent_map=build_parent_map(parent_docs)
    return parent_docs, child_docs, parent_map