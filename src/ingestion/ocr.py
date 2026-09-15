"""
OCR wrapper around PaddleOCR-VL.

Used only for pages that don't yield usable text through direct PDF text
extraction (i.e. scanned/image-based pages). Keeping this isolated means the
loader can decide per-page whether OCR is even needed, which keeps ingestion
fast for text-native PDFs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ocr_engine = None  # lazy singleton, PaddleOCR-VL init is expensive


@dataclass
class OcrResult:
    text: str
    confidence: float
    page_number: int


def _get_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR-VL engine")
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False,
        )
    return _ocr_engine


def ocr_image(image_path: str, page_number: int = 0) -> OcrResult:
    """Run PaddleOCR-VL on a single page image and return joined text
    with a mean confidence score across detected text lines."""
    engine = _get_engine()
    result = engine.ocr(image_path, cls=True)

    lines, confidences = [], []
    for page_result in result:
        if not page_result:
            continue
        for _, (text, conf) in page_result:
            lines.append(text)
            confidences.append(conf)

    joined_text = "\n".join(lines)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    if not joined_text.strip():
        logger.warning("OCR produced no text for page %s (%s)", page_number, image_path)

    return OcrResult(text=joined_text, confidence=avg_conf, page_number=page_number)
