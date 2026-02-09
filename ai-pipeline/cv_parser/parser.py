"""
CV Parser Module
Extracts structured data from CV files using AI

Supports:
- Text-based PDFs and DOCX
- Scanned documents (via Image Processor)
- Image files (PNG, JPG)
"""
import os
import io
import json
from typing import Dict, Any, Optional, Union
import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Import image processor for OCR support
try:
    from image_processor import DocumentProcessor, OCRProvider
    IMAGE_PROCESSOR_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSOR_AVAILABLE = False


class CVParser:
    """
    Parse CV documents and extract structured data.

    Supports both text-based and scanned documents via OCR.
    """

    # Supported file formats
    TEXT_FORMATS = {'.pdf', '.doc', '.docx', '.txt', '.rtf'}
    IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.heic'}

    def __init__(
        self,
        use_image_processor: bool = True,
        ocr_provider: str = "tesseract",
        languages: list = None,
    ):
        """
        Initialize CV Parser.

        Args:
            use_image_processor: Enable OCR for scanned documents
            ocr_provider: OCR engine (tesseract, easyocr, paddleocr)
            languages: OCR language codes
        """
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Image processor for OCR
        self.use_image_processor = use_image_processor and IMAGE_PROCESSOR_AVAILABLE
        self.image_processor = None
        self.ocr_languages = languages or ["en", "tr"]

        if self.use_image_processor:
            try:
                self.image_processor = DocumentProcessor(
                    ocr_provider=OCRProvider(ocr_provider),
                    languages=self.ocr_languages,
                )
            except Exception as e:
                print(f"Warning: Could not initialize image processor: {e}")
                self.use_image_processor = False

    async def parse(self, file_url: str) -> Dict[str, Any]:
        """
        Parse CV from URL and extract structured data
        """
        # Download file
        text_content = await self._extract_text(file_url)

        # Use LLM to extract structured data
        parsed_data = await self._extract_with_llm(text_content)

        return parsed_data

    async def _extract_text(self, file_url: str) -> str:
        """
        Extract text content from CV file.

        Uses OCR for scanned documents and images.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            content = response.content

        # Get file extension
        ext = os.path.splitext(file_url.lower())[1]

        # Image files - use OCR directly
        if ext in self.IMAGE_FORMATS:
            return await self._extract_with_ocr(content, file_url)

        # PDF files - try text extraction first, then OCR
        if ext == '.pdf':
            text = await self._extract_pdf(content)
            # If very little text extracted, might be scanned - try OCR
            if len(text.strip()) < 100 and self.use_image_processor:
                ocr_text = await self._extract_with_ocr(content, file_url)
                if len(ocr_text) > len(text):
                    return ocr_text
            return text

        # DOCX files
        if ext in ('.doc', '.docx'):
            return await self._extract_docx(content)

        # Plain text
        return content.decode('utf-8', errors='ignore')

    async def _extract_with_ocr(self, content: bytes, filename: str) -> str:
        """
        Extract text using OCR (for scanned documents and images).

        Args:
            content: File content as bytes
            filename: Original filename for format detection

        Returns:
            Extracted text
        """
        if not self.use_image_processor or not self.image_processor:
            raise ValueError("Image processor not available for OCR")

        try:
            result = self.image_processor.process(content, filename=filename)
            return result.combined_text
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""

    async def parse_with_layout(
        self,
        file_url: str
    ) -> Dict[str, Any]:
        """
        Parse CV with layout analysis.

        Returns both text content and document structure.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(file_url)
            content = response.content

        if not self.use_image_processor or not self.image_processor:
            # Fallback to regular parsing
            return await self.parse(file_url)

        # Process with image processor
        result = self.image_processor.process(content, filename=file_url)

        # Get structured data from layout analysis
        layout_data = result.structured_data

        # Extract text by section
        section_texts = {}
        for section, text in layout_data.get('sections', {}).items():
            if text.strip():
                section_texts[section] = text

        # Use LLM to further structure the data
        parsed_data = await self._extract_with_llm(result.combined_text)

        # Merge layout data with LLM parsing
        if layout_data.get('contact'):
            parsed_data.setdefault('personal_info', {})
            parsed_data['personal_info'].update(layout_data['contact'])

        # Add layout metadata
        parsed_data['_layout'] = {
            'has_photo': layout_data.get('has_photo', False),
            'columns': layout_data.get('layout_info', {}).get('columns', 1),
            'total_pages': result.total_pages,
            'confidence': result.confidence,
        }

        return parsed_data

    async def _extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF"""
        import pdfplumber
        import io

        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    async def _extract_docx(self, content: bytes) -> str:
        """Extract text from DOCX"""
        from docx import Document
        import io

        doc = Document(io.BytesIO(content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    async def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract structured CV data"""
        prompt = """Analyze the following CV text and extract structured information.
Return a JSON object with the following structure:

{
    "personal_info": {
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "+90...",
        "location": "City, Country",
        "linkedin": "URL",
        "website": "URL"
    },
    "summary": "Professional summary text",
    "experience": [
        {
            "title": "Job Title",
            "company": "Company Name",
            "location": "City",
            "start_date": "YYYY-MM",
            "end_date": "YYYY-MM or Present",
            "description": "Job description",
            "achievements": ["achievement 1", "achievement 2"]
        }
    ],
    "education": [
        {
            "degree": "Degree Name",
            "institution": "University Name",
            "location": "City",
            "graduation_date": "YYYY",
            "gpa": "3.5/4.0",
            "field": "Field of Study"
        }
    ],
    "skills": {
        "technical": ["skill1", "skill2"],
        "soft": ["skill1", "skill2"],
        "languages": [{"language": "English", "level": "Fluent"}]
    },
    "certifications": [
        {
            "name": "Certification Name",
            "issuer": "Issuing Organization",
            "date": "YYYY-MM"
        }
    ]
}

CV Text:
"""
        response = await self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a CV parsing expert. Extract structured data from CV text. Always respond with valid JSON."},
                {"role": "user", "content": prompt + text}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        return json.loads(response.choices[0].message.content)

    async def generate_script(
        self,
        cv_data: Dict[str, Any],
        language: str = "tr",
        duration_seconds: int = 60
    ) -> str:
        """Generate video presentation script from CV data"""

        language_prompts = {
            "tr": "Türkçe olarak yaz. Profesyonel ve samimi bir ton kullan.",
            "en": "Write in English. Use a professional yet friendly tone.",
            "de": "Schreiben Sie auf Deutsch. Verwenden Sie einen professionellen, aber freundlichen Ton.",
            "ar": "اكتب باللغة العربية. استخدم نبرة مهنية ودودة."
        }

        prompt = f"""Create a professional video CV script based on the following data.
The script should be for a {duration_seconds}-second video (approximately {duration_seconds * 2} words).

{language_prompts.get(language, language_prompts['en'])}

Structure:
1. Brief introduction (name, current role)
2. Key experience highlights (most relevant 2-3 points)
3. Key skills and strengths
4. What they're looking for / career goals
5. Closing statement

CV Data:
{json.dumps(cv_data, indent=2)}

Generate a natural, engaging script that the person would speak directly to camera.
Do not include any stage directions or [brackets].
"""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert at creating engaging video CV scripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content
