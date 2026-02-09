"""
FastAPI Router for Image Processing Service
============================================

REST API endpoints for CV image processing:
- Document upload and processing
- OCR extraction
- Layout analysis
- Batch processing
"""

import io
import base64
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
from PIL import Image

from .document_processor import DocumentProcessor, DocumentResult
from .ocr import OCRProvider, OCRResult
from .layout_analyzer import SectionType


router = APIRouter(prefix="/image-processor", tags=["Image Processing"])


class OCRProviderEnum(str, Enum):
    tesseract = "tesseract"
    easyocr = "easyocr"
    paddleocr = "paddleocr"
    google_vision = "google_vision"


class ProcessingOptions(BaseModel):
    """Options for document processing."""
    ocr_provider: OCRProviderEnum = OCRProviderEnum.tesseract
    languages: List[str] = Field(default=["en", "tr"])
    enhance_contrast: bool = True
    denoise: bool = True
    deskew: bool = True
    use_ml_layout: bool = False
    extract_photo: bool = True


class ContactInfo(BaseModel):
    """Extracted contact information."""
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None


class SectionData(BaseModel):
    """Extracted section data."""
    section_type: str
    content: str
    confidence: float = 0.0


class PageData(BaseModel):
    """Data for a single page."""
    page_number: int
    text: str
    confidence: float
    has_photo: bool
    sections: List[SectionData]


class ProcessingResponse(BaseModel):
    """Complete processing response."""
    success: bool
    filename: str
    total_pages: int
    combined_text: str
    confidence: float
    contact: ContactInfo
    sections: dict
    pages: List[PageData]
    has_photo: bool
    photo_base64: Optional[str] = None
    metadata: dict


class OCROnlyResponse(BaseModel):
    """OCR-only extraction response."""
    success: bool
    text: str
    confidence: float
    word_count: int
    language: str


class LayoutResponse(BaseModel):
    """Layout analysis response."""
    success: bool
    page_width: int
    page_height: int
    columns: int
    has_photo: bool
    photo_location: Optional[dict] = None
    detected_sections: List[str]
    block_count: int


# Global processor instance (lazy loaded)
_processor: Optional[DocumentProcessor] = None


def get_processor(options: ProcessingOptions) -> DocumentProcessor:
    """Get or create document processor with options."""
    return DocumentProcessor(
        ocr_provider=OCRProvider(options.ocr_provider.value),
        languages=options.languages,
        use_gpu=False,
        use_ml_layout=options.use_ml_layout,
    )


@router.post("/process", response_model=ProcessingResponse)
async def process_document(
    file: UploadFile = File(...),
    ocr_provider: OCRProviderEnum = OCRProviderEnum.tesseract,
    languages: str = "en,tr",
    extract_photo: bool = True,
):
    """
    Process a CV document with full analysis.

    - **file**: CV document (PDF, PNG, JPG, etc.)
    - **ocr_provider**: OCR engine to use
    - **languages**: Comma-separated language codes
    - **extract_photo**: Extract profile photo if detected

    Returns complete analysis including text, sections, and contact info.
    """
    try:
        # Read file content
        content = await file.read()

        # Create options
        options = ProcessingOptions(
            ocr_provider=ocr_provider,
            languages=languages.split(","),
            extract_photo=extract_photo,
        )

        # Process document
        processor = get_processor(options)
        result = processor.process(content, filename=file.filename)

        # Extract photo if requested
        photo_base64 = None
        if extract_photo and result.structured_data.get('has_photo'):
            photo = processor.extract_photo(result)
            if photo is not None:
                # Convert to base64
                img = Image.fromarray(photo)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                photo_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Build response
        pages_data = []
        for page in result.pages:
            sections = []
            for section_type, blocks in page.layout.sections.items():
                if blocks:
                    section_text = " ".join(b.text for b in blocks if b.text)
                    if section_text.strip():
                        sections.append(SectionData(
                            section_type=section_type.value,
                            content=section_text.strip(),
                            confidence=sum(b.confidence for b in blocks) / len(blocks) if blocks else 0,
                        ))

            pages_data.append(PageData(
                page_number=page.page_number,
                text=page.ocr_result.text,
                confidence=page.ocr_result.confidence,
                has_photo=page.layout.has_photo,
                sections=sections,
            ))

        contact_data = result.structured_data.get('contact', {})

        return ProcessingResponse(
            success=True,
            filename=result.filename,
            total_pages=result.total_pages,
            combined_text=result.combined_text,
            confidence=result.confidence,
            contact=ContactInfo(**contact_data),
            sections=result.structured_data.get('sections', {}),
            pages=pages_data,
            has_photo=result.structured_data.get('has_photo', False),
            photo_base64=photo_base64,
            metadata=result.metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr", response_model=OCROnlyResponse)
async def extract_text(
    file: UploadFile = File(...),
    ocr_provider: OCRProviderEnum = OCRProviderEnum.tesseract,
    languages: str = "en,tr",
):
    """
    Extract text from document using OCR only.

    Faster than full processing, suitable for quick text extraction.
    """
    try:
        content = await file.read()

        options = ProcessingOptions(
            ocr_provider=ocr_provider,
            languages=languages.split(","),
        )

        processor = get_processor(options)
        result = processor.process(content, filename=file.filename)

        word_count = len(result.combined_text.split())

        return OCROnlyResponse(
            success=True,
            text=result.combined_text,
            confidence=result.confidence,
            word_count=word_count,
            language=",".join(options.languages),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/layout", response_model=LayoutResponse)
async def analyze_layout(
    file: UploadFile = File(...),
    use_ml: bool = False,
):
    """
    Analyze document layout only.

    Returns structure information without full text extraction.
    """
    try:
        content = await file.read()

        options = ProcessingOptions(use_ml_layout=use_ml)
        processor = get_processor(options)
        result = processor.process(content, filename=file.filename)

        if not result.pages:
            raise HTTPException(status_code=400, detail="No pages found in document")

        first_page = result.pages[0]
        layout = first_page.layout

        detected_sections = [
            section.value
            for section, blocks in layout.sections.items()
            if blocks
        ]

        return LayoutResponse(
            success=True,
            page_width=layout.page_width,
            page_height=layout.page_height,
            columns=layout.columns,
            has_photo=layout.has_photo,
            photo_location={"bbox": layout.photo_location} if layout.photo_location else None,
            detected_sections=detected_sections,
            block_count=len(layout.blocks),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def batch_process(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    ocr_provider: OCRProviderEnum = OCRProviderEnum.tesseract,
    languages: str = "en,tr",
):
    """
    Process multiple documents in batch.

    Returns job ID for status tracking.
    """
    # For now, process synchronously
    # TODO: Implement async batch processing with Celery

    results = []

    options = ProcessingOptions(
        ocr_provider=ocr_provider,
        languages=languages.split(","),
    )
    processor = get_processor(options)

    for file in files:
        try:
            content = await file.read()
            result = processor.process(content, filename=file.filename)

            results.append({
                "filename": file.filename,
                "success": True,
                "total_pages": result.total_pages,
                "confidence": result.confidence,
                "text_length": len(result.combined_text),
                "has_photo": result.structured_data.get('has_photo', False),
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })

    return {
        "success": True,
        "total_files": len(files),
        "processed": len([r for r in results if r.get("success")]),
        "failed": len([r for r in results if not r.get("success")]),
        "results": results,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "image-processor"}


@router.get("/providers")
async def list_providers():
    """List available OCR providers."""
    return {
        "providers": [
            {
                "id": "tesseract",
                "name": "Tesseract OCR",
                "description": "Open-source OCR engine, requires system installation",
                "languages": "100+ languages",
                "speed": "fast",
                "accuracy": "good",
            },
            {
                "id": "easyocr",
                "name": "EasyOCR",
                "description": "Deep learning based, pure Python",
                "languages": "80+ languages",
                "speed": "medium",
                "accuracy": "very good",
            },
            {
                "id": "paddleocr",
                "name": "PaddleOCR",
                "description": "High accuracy, multi-language support",
                "languages": "80+ languages",
                "speed": "medium",
                "accuracy": "excellent",
            },
            {
                "id": "google_vision",
                "name": "Google Cloud Vision",
                "description": "Cloud-based, requires API key",
                "languages": "100+ languages",
                "speed": "fast",
                "accuracy": "excellent",
            },
        ],
        "default": "tesseract",
    }
