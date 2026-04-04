import re
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from ingestion.document_schema import Document
from ingestion.utils import logger


def extract_video_id(url: str) -> str:
    if not url or not isinstance(url, str):
        raise ValueError("Invalid YouTube URL.")
    url = url.strip()
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Could not extract video ID from URL. Please check the URL and try again.")
    return match.group(1)


def load_youtube_transcript(url: str):
    video_id = extract_video_id(url)
    logger.info("Fetching transcript for video: %s", video_id)

    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(["en"])
        except Exception:
            try:
                transcript = transcript_list.find_transcript(["te"])
            except Exception:
                transcript = transcript_list.find_generated_transcript(
                    [t.language_code for t in transcript_list]
                )
        transcript_data = transcript.fetch()
    except Exception as e:
        logger.error("Failed to retrieve transcript for %s", video_id)
        raise e

    documents = [
        Document(
            content=entry.text.strip(),
            metadata={
                "source": "youtube",
                "video_id": video_id,
                "timestamp": entry.start,
                "duration": entry.duration
            }
        )
        for entry in transcript_data
        if entry.text and entry.text.strip()
    ]

    logger.info("%d transcript segments loaded.", len(documents))
    return documents
