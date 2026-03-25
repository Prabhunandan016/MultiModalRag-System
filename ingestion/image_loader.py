import cv2
import pytesseract
from PIL import Image
from ingestion.document_schema import Document
from ingestion.utils import logger, setup_tesseract


def preprocess_image(image_path):

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 3)

    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    return thresh


def load_image(image_path):

    setup_tesseract()

    logger.info(f"Processing image: {image_path}")

    processed = preprocess_image(image_path)

    text = pytesseract.image_to_string(processed)

    document = Document(
        content=text,
        metadata={
            "source": "image",
            "file": image_path
        }
    )

    return [document]