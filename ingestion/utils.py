import logging
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def setup_tesseract():
    import pytesseract
    path = shutil.which("tesseract")
    if path:
        pytesseract.pytesseract.tesseract_cmd = path
        logger.info(f"Tesseract detected at {path}")
    else:
        raise EnvironmentError("Tesseract OCR not installed")
