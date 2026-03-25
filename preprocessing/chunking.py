from typing import List
from ingestion.document_schema import Document


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: List[Document], chunk_size: int = 300, overlap: int = 50) -> List[Document]:

    chunked_docs = []

    for doc in documents:
        if not doc.content.strip():
            continue

        for chunk in chunk_text(doc.content, chunk_size, overlap):
            if chunk.strip():
                chunked_docs.append(Document(content=chunk, metadata=doc.metadata))

    return chunked_docs
