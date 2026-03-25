from ingestion.youtube_loader import load_youtube_transcript
from ingestion.pdf_loader import load_pdf
from ingestion.image_loader import load_image
from ingestion.utils import logger


def ingest_source(source_type: str, path: str):

    logger.info(f"Starting ingestion: {source_type}")

    if source_type == "youtube":

        return load_youtube_transcript(path)

    elif source_type == "pdf":

        return load_pdf(path)

    elif source_type == "image":

        return load_image(path)

    else:

        raise ValueError("Unsupported source type")


if __name__ == "__main__":

    print("Select source type:")
    print("1 - YouTube")
    print("2 - PDF")
    print("3 - Image")

    choice = input("Enter option: ")

    if choice == "1":
        url = input("Enter YouTube URL: ")
        docs = ingest_source("youtube", url)

    elif choice == "2":
        path = input("Enter PDF path: ")
        docs = ingest_source("pdf", path)

    elif choice == "3":
        path = input("Enter Image path: ")
        docs = ingest_source("image", path)

    else:
        print("Invalid choice")
        exit()

    print(f"\nLoaded {len(docs)} documents\n")

    print(docs[0])