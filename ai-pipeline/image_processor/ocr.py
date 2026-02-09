"""
OCR Engine for CV Document Processing
=====================================

Multi-engine OCR support:
- Tesseract OCR (default, open-source)
- EasyOCR (deep learning based)
- PaddleOCR (high accuracy)
- Google Cloud Vision (cloud-based)
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PIL import Image
from pathlib import Path


class OCRProvider(str, Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    GOOGLE_VISION = "google_vision"


@dataclass
class OCRResult:
    """OCR extraction result."""
    text: str
    confidence: float
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    word_level: List[Dict[str, Any]] = field(default_factory=list)
    line_level: List[Dict[str, Any]] = field(default_factory=list)
    raw_data: Optional[Any] = None


@dataclass
class BoundingBox:
    """Text bounding box."""
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float


class OCREngine:
    """Multi-engine OCR processor."""

    def __init__(
        self,
        provider: OCRProvider = OCRProvider.TESSERACT,
        languages: List[str] = None,
        gpu: bool = False,
    ):
        self.provider = provider
        self.languages = languages or ["en", "tr"]  # Default: English + Turkish
        self.gpu = gpu
        self._engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the selected OCR engine."""
        if self.provider == OCRProvider.TESSERACT:
            self._init_tesseract()
        elif self.provider == OCRProvider.EASYOCR:
            self._init_easyocr()
        elif self.provider == OCRProvider.PADDLEOCR:
            self._init_paddleocr()
        elif self.provider == OCRProvider.GOOGLE_VISION:
            self._init_google_vision()

    def _init_tesseract(self):
        """Initialize Tesseract OCR."""
        import pytesseract
        # Check if tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                "Tesseract not found. Install with: "
                "brew install tesseract (macOS) or "
                "apt-get install tesseract-ocr (Ubuntu)"
            ) from e
        self._engine = pytesseract

    def _init_easyocr(self):
        """Initialize EasyOCR."""
        import easyocr
        self._engine = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            verbose=False
        )

    def _init_paddleocr(self):
        """Initialize PaddleOCR."""
        from paddleocr import PaddleOCR
        lang = "en" if "en" in self.languages else self.languages[0]
        self._engine = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=self.gpu,
            show_log=False
        )

    def _init_google_vision(self):
        """Initialize Google Cloud Vision."""
        from google.cloud import vision
        self._engine = vision.ImageAnnotatorClient()

    def extract_text(
        self,
        image: Union[np.ndarray, Image.Image, str, Path, bytes],
        detailed: bool = False,
    ) -> OCRResult:
        """
        Extract text from image.

        Args:
            image: Input image
            detailed: Include word/line level details

        Returns:
            OCRResult with extracted text and metadata
        """
        if self.provider == OCRProvider.TESSERACT:
            return self._extract_tesseract(image, detailed)
        elif self.provider == OCRProvider.EASYOCR:
            return self._extract_easyocr(image, detailed)
        elif self.provider == OCRProvider.PADDLEOCR:
            return self._extract_paddleocr(image, detailed)
        elif self.provider == OCRProvider.GOOGLE_VISION:
            return self._extract_google_vision(image, detailed)

    def _extract_tesseract(
        self,
        image: Union[np.ndarray, Image.Image, str, Path, bytes],
        detailed: bool
    ) -> OCRResult:
        """Extract text using Tesseract."""
        import pytesseract

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, bytes):
            import io
            image = Image.open(io.BytesIO(image))

        # Language string for Tesseract
        lang_str = "+".join(self.languages)

        # Basic text extraction
        text = pytesseract.image_to_string(image, lang=lang_str)

        # Get detailed data if requested
        word_level = []
        line_level = []
        bounding_boxes = []
        confidence = 0.0

        if detailed:
            data = pytesseract.image_to_data(
                image, lang=lang_str, output_type=pytesseract.Output.DICT
            )

            confidences = []
            current_line = []
            current_line_num = 0

            for i in range(len(data["text"])):
                if data["text"][i].strip():
                    conf = float(data["conf"][i])
                    if conf > 0:
                        confidences.append(conf)

                    word_data = {
                        "text": data["text"][i],
                        "confidence": conf,
                        "x": data["left"][i],
                        "y": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                        "line_num": data["line_num"][i],
                        "block_num": data["block_num"][i],
                    }
                    word_level.append(word_data)
                    bounding_boxes.append(word_data)

                    # Group by line
                    if data["line_num"][i] != current_line_num:
                        if current_line:
                            line_level.append({
                                "text": " ".join([w["text"] for w in current_line]),
                                "words": current_line,
                                "line_num": current_line_num,
                            })
                        current_line = []
                        current_line_num = data["line_num"][i]
                    current_line.append(word_data)

            # Add last line
            if current_line:
                line_level.append({
                    "text": " ".join([w["text"] for w in current_line]),
                    "words": current_line,
                    "line_num": current_line_num,
                })

            confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=text.strip(),
            confidence=confidence,
            bounding_boxes=bounding_boxes,
            word_level=word_level,
            line_level=line_level,
            language=lang_str,
        )

    def _extract_easyocr(
        self,
        image: Union[np.ndarray, Image.Image, str, Path, bytes],
        detailed: bool
    ) -> OCRResult:
        """Extract text using EasyOCR."""
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, bytes):
            import io
            image = np.array(Image.open(io.BytesIO(image)))
        elif isinstance(image, (str, Path)):
            image = np.array(Image.open(image))

        results = self._engine.readtext(image)

        text_parts = []
        bounding_boxes = []
        confidences = []

        for bbox, text, conf in results:
            text_parts.append(text)
            confidences.append(conf)

            # Convert bbox to standard format
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            bounding_boxes.append({
                "text": text,
                "confidence": conf,
                "x": int(min(x_coords)),
                "y": int(min(y_coords)),
                "width": int(max(x_coords) - min(x_coords)),
                "height": int(max(y_coords) - min(y_coords)),
                "polygon": bbox,
            })

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text="\n".join(text_parts),
            confidence=avg_confidence,
            bounding_boxes=bounding_boxes,
            raw_data=results,
        )

    def _extract_paddleocr(
        self,
        image: Union[np.ndarray, Image.Image, str, Path, bytes],
        detailed: bool
    ) -> OCRResult:
        """Extract text using PaddleOCR."""
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, bytes):
            import io
            image = np.array(Image.open(io.BytesIO(image)))

        result = self._engine.ocr(image, cls=True)

        text_parts = []
        bounding_boxes = []
        confidences = []

        if result and result[0]:
            for line in result[0]:
                bbox, (text, conf) = line
                text_parts.append(text)
                confidences.append(conf)

                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                bounding_boxes.append({
                    "text": text,
                    "confidence": conf,
                    "x": int(min(x_coords)),
                    "y": int(min(y_coords)),
                    "width": int(max(x_coords) - min(x_coords)),
                    "height": int(max(y_coords) - min(y_coords)),
                    "polygon": bbox,
                })

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text="\n".join(text_parts),
            confidence=avg_confidence,
            bounding_boxes=bounding_boxes,
            raw_data=result,
        )

    def _extract_google_vision(
        self,
        image: Union[np.ndarray, Image.Image, str, Path, bytes],
        detailed: bool
    ) -> OCRResult:
        """Extract text using Google Cloud Vision."""
        from google.cloud import vision

        # Convert to bytes
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if isinstance(image, Image.Image):
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            content = buffer.getvalue()
        elif isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                content = f.read()
        else:
            content = image

        vision_image = vision.Image(content=content)
        response = self._engine.document_text_detection(image=vision_image)

        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        text = response.full_text_annotation.text if response.full_text_annotation else ""

        bounding_boxes = []
        word_level = []

        if detailed and response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join([s.text for s in word.symbols])
                            vertices = word.bounding_box.vertices

                            word_data = {
                                "text": word_text,
                                "confidence": word.confidence,
                                "x": vertices[0].x,
                                "y": vertices[0].y,
                                "width": vertices[2].x - vertices[0].x,
                                "height": vertices[2].y - vertices[0].y,
                            }
                            word_level.append(word_data)
                            bounding_boxes.append(word_data)

        return OCRResult(
            text=text.strip(),
            confidence=0.95,  # Google Vision doesn't provide overall confidence
            bounding_boxes=bounding_boxes,
            word_level=word_level,
            raw_data=response,
        )

    def detect_language(
        self,
        image: Union[np.ndarray, Image.Image, str, Path, bytes]
    ) -> str:
        """Detect document language."""
        if self.provider == OCRProvider.TESSERACT:
            import pytesseract
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            try:
                osd = pytesseract.image_to_osd(image)
                script = osd.split("Script: ")[1].split("\n")[0]
                return script.lower()
            except:
                return "unknown"

        # For other providers, extract text and use langdetect
        result = self.extract_text(image)
        if result.text:
            try:
                from langdetect import detect
                return detect(result.text)
            except:
                pass

        return "unknown"
