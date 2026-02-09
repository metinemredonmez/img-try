"""
Document Processor for CV Analysis
==================================

Main orchestrator for CV document processing:
- PDF/Image loading
- Multi-page document handling
- Format conversion
- Complete processing pipeline
"""

import os
import io
import tempfile
from typing import List, Dict, Any, Optional, Union, BinaryIO
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
import numpy as np

from .preprocessor import ImagePreprocessor
from .ocr import OCREngine, OCRResult, OCRProvider
from .layout_analyzer import LayoutAnalyzer, DocumentLayout, SectionType


@dataclass
class PageResult:
    """Processing result for a single page."""
    page_number: int
    image: np.ndarray
    preprocessed_image: np.ndarray
    ocr_result: OCRResult
    layout: DocumentLayout
    extracted_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentResult:
    """Complete document processing result."""
    filename: str
    total_pages: int
    pages: List[PageResult]
    combined_text: str
    structured_data: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentProcessor:
    """
    Main document processor for CV analysis.

    Orchestrates the complete processing pipeline:
    1. Load document (PDF, images)
    2. Convert to images
    3. Preprocess images
    4. OCR text extraction
    5. Layout analysis
    6. Section extraction
    7. Structured data output
    """

    SUPPORTED_FORMATS = {
        '.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif',
        '.bmp', '.gif', '.webp', '.heic', '.heif'
    }

    def __init__(
        self,
        ocr_provider: OCRProvider = OCRProvider.TESSERACT,
        languages: List[str] = None,
        use_gpu: bool = False,
        use_ml_layout: bool = False,
    ):
        """
        Initialize document processor.

        Args:
            ocr_provider: OCR engine to use
            languages: Languages for OCR
            use_gpu: Use GPU acceleration
            use_ml_layout: Use ML-based layout detection
        """
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine(
            provider=ocr_provider,
            languages=languages or ["en", "tr"],
            gpu=use_gpu,
        )
        self.layout_analyzer = LayoutAnalyzer(use_ml_model=use_ml_layout)

    def process(
        self,
        source: Union[str, Path, bytes, BinaryIO],
        filename: Optional[str] = None,
    ) -> DocumentResult:
        """
        Process a CV document.

        Args:
            source: File path, bytes, or file-like object
            filename: Original filename (for format detection)

        Returns:
            DocumentResult with all extracted information
        """
        # Determine file format
        if isinstance(source, (str, Path)):
            file_path = Path(source)
            filename = filename or file_path.name
            ext = file_path.suffix.lower()

            with open(file_path, 'rb') as f:
                content = f.read()
        elif isinstance(source, bytes):
            content = source
            ext = self._detect_format(content, filename)
        else:
            content = source.read()
            ext = self._detect_format(content, filename)

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")

        # Convert to images
        images = self._load_document(content, ext)

        # Process each page
        pages = []
        all_text_parts = []
        total_confidence = 0.0

        for i, img in enumerate(images):
            page_result = self._process_page(img, i + 1)
            pages.append(page_result)
            all_text_parts.append(page_result.ocr_result.text)
            total_confidence += page_result.ocr_result.confidence

        # Combine results
        combined_text = "\n\n".join(all_text_parts)
        avg_confidence = total_confidence / len(pages) if pages else 0.0

        # Extract structured data from all pages
        structured_data = self._extract_structured_data(pages)

        return DocumentResult(
            filename=filename or "document",
            total_pages=len(pages),
            pages=pages,
            combined_text=combined_text,
            structured_data=structured_data,
            confidence=avg_confidence,
            metadata={
                "format": ext,
                "ocr_provider": self.ocr_engine.provider.value,
            }
        )

    def _detect_format(
        self,
        content: bytes,
        filename: Optional[str]
    ) -> str:
        """Detect file format from content or filename."""
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in self.SUPPORTED_FORMATS:
                return ext

        # Magic number detection
        if content[:4] == b'%PDF':
            return '.pdf'
        elif content[:8] == b'\x89PNG\r\n\x1a\n':
            return '.png'
        elif content[:2] == b'\xff\xd8':
            return '.jpg'
        elif content[:4] in (b'II*\x00', b'MM\x00*'):
            return '.tiff'
        elif content[:6] in (b'GIF87a', b'GIF89a'):
            return '.gif'
        elif content[:4] == b'RIFF' and content[8:12] == b'WEBP':
            return '.webp'

        return '.png'  # Default assumption

    def _load_document(
        self,
        content: bytes,
        ext: str
    ) -> List[np.ndarray]:
        """Load document and convert to images."""
        if ext == '.pdf':
            return self._load_pdf(content)
        else:
            return self._load_image(content)

    def _load_pdf(self, content: bytes) -> List[np.ndarray]:
        """Convert PDF to images."""
        try:
            # Try pdf2image first (requires poppler)
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(
                content,
                dpi=300,
                fmt='png',
            )
            return [np.array(img) for img in images]
        except ImportError:
            pass

        try:
            # Fallback to PyMuPDF
            import fitz  # PyMuPDF

            doc = fitz.open(stream=content, filetype="pdf")
            images = []

            for page in doc:
                # Render at 300 DPI
                mat = fitz.Matrix(300/72, 300/72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(np.array(img))

            doc.close()
            return images
        except ImportError:
            raise ImportError(
                "PDF processing requires pdf2image or PyMuPDF. "
                "Install with: pip install pdf2image or pip install PyMuPDF"
            )

    def _load_image(self, content: bytes) -> List[np.ndarray]:
        """Load image file."""
        img = Image.open(io.BytesIO(content))

        # Handle multi-frame images (TIFF, GIF)
        images = []
        try:
            while True:
                frame = img.copy()
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                images.append(np.array(frame))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        if not images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images = [np.array(img)]

        return images

    def _process_page(
        self,
        image: np.ndarray,
        page_number: int
    ) -> PageResult:
        """Process a single page."""
        # Preprocess image
        preprocessed = self.preprocessor.preprocess(
            image,
            enhance_contrast=True,
            denoise=True,
            deskew=True,
            binarize=False,
        )

        # Correct orientation
        preprocessed = self.preprocessor.correct_orientation(preprocessed)

        # OCR extraction
        ocr_result = self.ocr_engine.extract_text(preprocessed, detailed=True)

        # Layout analysis
        layout = self.layout_analyzer.analyze(image, ocr_result)

        # Extract page data
        extracted_data = self._extract_page_data(ocr_result, layout)

        return PageResult(
            page_number=page_number,
            image=image,
            preprocessed_image=preprocessed,
            ocr_result=ocr_result,
            layout=layout,
            extracted_data=extracted_data,
        )

    def _extract_page_data(
        self,
        ocr_result: OCRResult,
        layout: DocumentLayout
    ) -> Dict[str, Any]:
        """Extract structured data from a page."""
        data = {}

        # Extract text by section
        for section_type, blocks in layout.sections.items():
            if blocks:
                section_text = " ".join(b.text for b in blocks if b.text)
                if section_text.strip():
                    data[section_type.value] = section_text.strip()

        # Extract contact information using patterns
        contact_info = self._extract_contact_info(ocr_result.text)
        if contact_info:
            data['contact'] = contact_info

        return data

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information using regex patterns."""
        import re

        contact = {}

        # Email pattern
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, text)
        if emails:
            contact['email'] = emails[0]

        # Phone pattern (international format)
        phone_pattern = r'[\+]?[(]?[0-9]{1,3}[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?[-\s\.]?[0-9]{3,4}[-\s\.]?[0-9]{3,4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact['phone'] = phones[0].strip()

        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        if linkedin:
            contact['linkedin'] = f"https://{linkedin[0]}"

        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        github = re.findall(github_pattern, text, re.IGNORECASE)
        if github:
            contact['github'] = f"https://{github[0]}"

        # Website pattern
        website_pattern = r'https?://(?:www\.)?[\w\.-]+\.\w+(?:/[\w\.-]*)*'
        websites = re.findall(website_pattern, text)
        other_websites = [w for w in websites if 'linkedin' not in w.lower() and 'github' not in w.lower()]
        if other_websites:
            contact['website'] = other_websites[0]

        return contact

    def _extract_structured_data(
        self,
        pages: List[PageResult]
    ) -> Dict[str, Any]:
        """Combine and structure data from all pages."""
        structured = {
            'contact': {},
            'sections': {},
            'has_photo': False,
            'photo_location': None,
            'layout_info': {
                'total_pages': len(pages),
                'columns': 1,
            }
        }

        # Combine data from all pages
        for page in pages:
            # Merge contact info
            if 'contact' in page.extracted_data:
                structured['contact'].update(page.extracted_data['contact'])

            # Merge sections
            for key, value in page.extracted_data.items():
                if key != 'contact':
                    if key in structured['sections']:
                        structured['sections'][key] += "\n" + value
                    else:
                        structured['sections'][key] = value

            # Photo detection
            if page.layout.has_photo:
                structured['has_photo'] = True
                structured['photo_location'] = {
                    'page': page.page_number,
                    'bbox': page.layout.photo_location,
                }

            # Layout info
            if page.layout.columns > structured['layout_info']['columns']:
                structured['layout_info']['columns'] = page.layout.columns

        return structured

    def extract_photo(
        self,
        result: DocumentResult
    ) -> Optional[np.ndarray]:
        """Extract profile photo from processed document."""
        if not result.structured_data.get('has_photo'):
            return None

        photo_info = result.structured_data.get('photo_location')
        if not photo_info:
            return None

        page_num = photo_info['page']
        bbox = photo_info['bbox']

        if page_num <= len(result.pages) and bbox:
            page = result.pages[page_num - 1]
            x, y, w, h = bbox

            # Add margin
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)

            return page.image[y:y+h+2*margin, x:x+w+2*margin]

        return None


# Convenience function for quick processing
def process_cv(
    file_path: Union[str, Path],
    ocr_provider: OCRProvider = OCRProvider.TESSERACT,
) -> DocumentResult:
    """
    Quick function to process a CV document.

    Args:
        file_path: Path to CV document
        ocr_provider: OCR engine to use

    Returns:
        DocumentResult with extracted information
    """
    processor = DocumentProcessor(ocr_provider=ocr_provider)
    return processor.process(file_path)
