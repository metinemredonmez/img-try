"""
Image Processing Module for VidCV Platform
==========================================

Dedicated image processing component for CV document analysis:
- OCR (Optical Character Recognition)
- Image preprocessing and enhancement
- Document layout analysis
- Text extraction from scanned documents
- Multi-format support (PDF, PNG, JPG, TIFF, HEIC)

This module is a critical separate architecture component.
"""

from .ocr import OCREngine
from .preprocessor import ImagePreprocessor
from .layout_analyzer import LayoutAnalyzer
from .document_processor import DocumentProcessor

__all__ = [
    "OCREngine",
    "ImagePreprocessor",
    "LayoutAnalyzer",
    "DocumentProcessor",
]
