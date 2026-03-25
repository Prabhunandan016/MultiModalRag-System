import re
from typing import List
from ingestion.document_schema import Document


def clean_text(text: str) -> str:

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:()\-]", "", text)
    text = re.sub(r"([.,!?])\1+", r"\1", text)

    return text.strip()


def clean_documents(documents: List[Document]) -> List[Document]:

    cleaned_docs = []

    for doc in documents:
        cleaned_text = clean_text(doc.content)
        if cleaned_text:
            cleaned_docs.append(Document(content=cleaned_text, metadata=doc.metadata))

    return cleaned_docs
