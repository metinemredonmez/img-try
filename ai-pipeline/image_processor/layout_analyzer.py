"""
Document Layout Analyzer for CV Documents
=========================================

Analyzes CV document structure:
- Section detection (Experience, Education, Skills, etc.)
- Text block classification
- Table detection
- Image/photo detection
- Contact information extraction
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SectionType(str, Enum):
    HEADER = "header"
    CONTACT = "contact"
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    LANGUAGES = "languages"
    CERTIFICATIONS = "certifications"
    PROJECTS = "projects"
    AWARDS = "awards"
    REFERENCES = "references"
    PHOTO = "photo"
    OTHER = "other"


class BlockType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    SEPARATOR = "separator"


@dataclass
class LayoutBlock:
    """Detected layout block."""
    block_type: BlockType
    section_type: Optional[SectionType]
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    confidence: float = 0.0
    children: List["LayoutBlock"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentLayout:
    """Complete document layout analysis."""
    blocks: List[LayoutBlock]
    sections: Dict[SectionType, List[LayoutBlock]]
    page_width: int
    page_height: int
    has_photo: bool = False
    photo_location: Optional[Tuple[int, int, int, int]] = None
    columns: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayoutAnalyzer:
    """Analyze CV document layout and structure."""

    # Section keywords for classification
    SECTION_KEYWORDS = {
        SectionType.CONTACT: [
            "contact", "email", "phone", "address", "linkedin", "github",
            "iletişim", "e-posta", "telefon", "adres"
        ],
        SectionType.SUMMARY: [
            "summary", "objective", "profile", "about", "overview",
            "özet", "profil", "hakkımda"
        ],
        SectionType.EXPERIENCE: [
            "experience", "employment", "work history", "career",
            "deneyim", "iş deneyimi", "kariyer"
        ],
        SectionType.EDUCATION: [
            "education", "academic", "qualification", "degree",
            "eğitim", "öğrenim", "akademik"
        ],
        SectionType.SKILLS: [
            "skills", "competencies", "expertise", "technical",
            "yetenekler", "beceriler", "teknik"
        ],
        SectionType.LANGUAGES: [
            "language", "languages",
            "dil", "diller", "yabancı dil"
        ],
        SectionType.CERTIFICATIONS: [
            "certification", "certificate", "license",
            "sertifika", "lisans"
        ],
        SectionType.PROJECTS: [
            "project", "portfolio",
            "proje", "portfolyo"
        ],
        SectionType.AWARDS: [
            "award", "achievement", "honor",
            "ödül", "başarı"
        ],
        SectionType.REFERENCES: [
            "reference", "referral",
            "referans"
        ],
    }

    def __init__(self, use_ml_model: bool = False):
        """
        Initialize layout analyzer.

        Args:
            use_ml_model: Use ML-based layout detection (requires layoutparser)
        """
        self.use_ml_model = use_ml_model
        self._model = None

        if use_ml_model:
            self._init_ml_model()

    def _init_ml_model(self):
        """Initialize ML-based layout detection model."""
        try:
            import layoutparser as lp
            # Use PubLayNet model for document layout detection
            self._model = lp.Detectron2LayoutModel(
                "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            )
        except ImportError:
            print("Warning: layoutparser not installed, using rule-based detection")
            self.use_ml_model = False

    def analyze(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        ocr_result: Optional[Any] = None,
    ) -> DocumentLayout:
        """
        Analyze document layout.

        Args:
            image: Input image
            ocr_result: Optional OCR result for text-based analysis

        Returns:
            DocumentLayout with detected blocks and sections
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image

        height, width = img.shape[:2]

        # Detect blocks using ML or rule-based approach
        if self.use_ml_model and self._model:
            blocks = self._detect_blocks_ml(img)
        else:
            blocks = self._detect_blocks_rules(img, ocr_result)

        # Detect photo
        has_photo, photo_location = self._detect_photo(img)

        # Classify sections
        sections = self._classify_sections(blocks, ocr_result)

        # Detect column layout
        columns = self._detect_columns(blocks, width)

        return DocumentLayout(
            blocks=blocks,
            sections=sections,
            page_width=width,
            page_height=height,
            has_photo=has_photo,
            photo_location=photo_location,
            columns=columns,
        )

    def _detect_blocks_ml(self, image: np.ndarray) -> List[LayoutBlock]:
        """Detect layout blocks using ML model."""
        import layoutparser as lp

        layout = self._model.detect(image)
        blocks = []

        block_type_map = {
            "Text": BlockType.TEXT,
            "Title": BlockType.HEADING,
            "List": BlockType.LIST,
            "Table": BlockType.TABLE,
            "Figure": BlockType.IMAGE,
        }

        for element in layout:
            block = LayoutBlock(
                block_type=block_type_map.get(element.type, BlockType.TEXT),
                section_type=None,
                x=int(element.block.x_1),
                y=int(element.block.y_1),
                width=int(element.block.width),
                height=int(element.block.height),
                confidence=element.score,
            )
            blocks.append(block)

        return blocks

    def _detect_blocks_rules(
        self,
        image: np.ndarray,
        ocr_result: Optional[Any] = None
    ) -> List[LayoutBlock]:
        """Detect layout blocks using rule-based approach."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        blocks = []
        min_area = 500  # Minimum block area

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Classify block type based on dimensions
            aspect_ratio = w / h if h > 0 else 0

            if aspect_ratio > 5:  # Wide and thin - likely separator
                block_type = BlockType.SEPARATOR
            elif aspect_ratio < 0.3:  # Tall and thin - likely list
                block_type = BlockType.LIST
            elif w > image.shape[1] * 0.8:  # Full width - likely heading or text
                block_type = BlockType.TEXT
            else:
                block_type = BlockType.TEXT

            block = LayoutBlock(
                block_type=block_type,
                section_type=None,
                x=x,
                y=y,
                width=w,
                height=h,
            )
            blocks.append(block)

        # Sort blocks by position (top to bottom, left to right)
        blocks.sort(key=lambda b: (b.y, b.x))

        return blocks

    def _detect_photo(
        self,
        image: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """Detect profile photo in CV."""
        # Use face detection
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Return the largest face (likely the profile photo)
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                return True, (int(x), int(y), int(w), int(h))
        except:
            pass

        return False, None

    def _classify_sections(
        self,
        blocks: List[LayoutBlock],
        ocr_result: Optional[Any] = None
    ) -> Dict[SectionType, List[LayoutBlock]]:
        """Classify blocks into CV sections."""
        sections: Dict[SectionType, List[LayoutBlock]] = {st: [] for st in SectionType}

        if not ocr_result:
            return sections

        # Use OCR text to classify sections
        if hasattr(ocr_result, 'line_level'):
            lines = ocr_result.line_level
        elif hasattr(ocr_result, 'bounding_boxes'):
            lines = ocr_result.bounding_boxes
        else:
            return sections

        current_section = SectionType.HEADER

        for line in lines:
            text = line.get('text', '').lower().strip()

            # Check if this line is a section header
            detected_section = self._detect_section_type(text)
            if detected_section:
                current_section = detected_section
                continue

            # Create block for this line
            block = LayoutBlock(
                block_type=BlockType.TEXT,
                section_type=current_section,
                x=line.get('x', 0),
                y=line.get('y', 0),
                width=line.get('width', 0),
                height=line.get('height', 0),
                text=line.get('text', ''),
                confidence=line.get('confidence', 0),
            )

            sections[current_section].append(block)

        return sections

    def _detect_section_type(self, text: str) -> Optional[SectionType]:
        """Detect section type from text."""
        text_lower = text.lower().strip()

        for section_type, keywords in self.SECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return section_type

        return None

    def _detect_columns(
        self,
        blocks: List[LayoutBlock],
        page_width: int
    ) -> int:
        """Detect number of columns in document."""
        if not blocks:
            return 1

        # Analyze x-positions of blocks
        x_positions = [b.x for b in blocks]
        x_centers = [b.x + b.width / 2 for b in blocks]

        # Check if there are blocks in different horizontal zones
        mid_point = page_width / 2

        left_count = sum(1 for x in x_centers if x < mid_point * 0.7)
        right_count = sum(1 for x in x_centers if x > mid_point * 1.3)

        if left_count > 2 and right_count > 2:
            return 2

        return 1

    def extract_contact_region(
        self,
        image: np.ndarray,
        layout: DocumentLayout
    ) -> Optional[np.ndarray]:
        """Extract contact information region from CV."""
        contact_blocks = layout.sections.get(SectionType.CONTACT, [])

        if not contact_blocks:
            # Assume contact is in the top 20% of the document
            height = layout.page_height
            return image[:int(height * 0.2), :]

        # Find bounding box of all contact blocks
        min_x = min(b.x for b in contact_blocks)
        min_y = min(b.y for b in contact_blocks)
        max_x = max(b.x + b.width for b in contact_blocks)
        max_y = max(b.y + b.height for b in contact_blocks)

        # Add margin
        margin = 20
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(layout.page_width, max_x + margin)
        max_y = min(layout.page_height, max_y + margin)

        return image[min_y:max_y, min_x:max_x]
