import re
from youtube_transcript_api import YouTubeTranscriptApi
from ingestion.document_schema import Document
from ingestion.utils import logger


def extract_video_id(url: str):
    """
    Extract the video ID from a YouTube URL.
    """

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)

    if match:
        return match.group(1)

    raise ValueError("Invalid YouTube URL")


def load_youtube_transcript(url: str):
    """
    Load transcript from a YouTube video and convert it
    into Document objects for the RAG pipeline.
    """

    video_id = extract_video_id(url)

    logger.info(f"Fetching transcript for video: {video_id}")

    api = YouTubeTranscriptApi()

    try:

        transcript_list = api.list(video_id)

        try:
            transcript = transcript_list.find_transcript(['en'])

        except:
            try:
                transcript = transcript_list.find_transcript(['te'])
            except:
                transcript = transcript_list.find_generated_transcript(
                    [t.language_code for t in transcript_list]
                )

        transcript_data = transcript.fetch()

    except Exception as e:
        logger.error("Failed to retrieve transcript")
        raise e

    documents = []

    for entry in transcript_data:

        doc = Document(
            content=entry.text,
            metadata={
                "source": "youtube",
                "video_id": video_id,
                "timestamp": entry.start,
                "duration": entry.duration
            }
        )

        documents.append(doc)

    logger.info(f"{len(documents)} transcript segments loaded")

    return documents