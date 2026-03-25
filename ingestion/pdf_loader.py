import fitz
from ingestion.document_schema import Document
from ingestion.utils import logger


def load_pdf(file_path: str):

    logger.info(f"Loading PDF: {file_path}")

    doc = fitz.open(file_path)

    documents = []

    for page_number, page in enumerate(doc):

        text = page.get_text()

        if not text.strip():
            continue

        document = Document(
            content=text,
            metadata={
                "source": "pdf",
                "page": page_number + 1,
                "file": file_path
            }
        )

        documents.append(document)

    logger.info(f"{len(documents)} pages extracted")

    return documents