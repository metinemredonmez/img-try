"""
Image Preprocessor for CV Documents
===================================

Handles image preprocessing for optimal OCR and text extraction:
- Noise reduction
- Contrast enhancement
- Deskewing
- Binarization
- Resolution normalization
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Union, Tuple, Optional
from pathlib import Path
import io


class ImagePreprocessor:
    """Image preprocessing for CV document analysis."""

    def __init__(
        self,
        target_dpi: int = 300,
        denoise_strength: int = 10,
        contrast_factor: float = 1.5,
    ):
        self.target_dpi = target_dpi
        self.denoise_strength = denoise_strength
        self.contrast_factor = contrast_factor

    def preprocess(
        self,
        image: Union[np.ndarray, Image.Image, bytes, str, Path],
        enhance_contrast: bool = True,
        denoise: bool = True,
        deskew: bool = True,
        binarize: bool = False,
    ) -> np.ndarray:
        """
        Full preprocessing pipeline for CV images.

        Args:
            image: Input image (numpy array, PIL Image, bytes, or path)
            enhance_contrast: Apply contrast enhancement
            denoise: Apply noise reduction
            deskew: Correct image rotation/skew
            binarize: Convert to binary (black/white)

        Returns:
            Preprocessed image as numpy array
        """
        # Convert to numpy array
        img = self._to_numpy(image)

        # Convert to grayscale if colored
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply preprocessing steps
        if denoise:
            gray = self._denoise(gray)

        if enhance_contrast:
            gray = self._enhance_contrast(gray)

        if deskew:
            gray = self._deskew(gray)

        if binarize:
            gray = self._binarize(gray)

        return gray

    def _to_numpy(
        self,
        image: Union[np.ndarray, Image.Image, bytes, str, Path]
    ) -> np.ndarray:
        """Convert various image formats to numpy array."""
        if isinstance(image, np.ndarray):
            return image

        if isinstance(image, Image.Image):
            return np.array(image)

        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if isinstance(image, (str, Path)):
            return cv2.imread(str(image))

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction."""
        return cv2.fastNlMeansDenoising(
            image,
            None,
            self.denoise_strength,
            7,
            21
        )

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew/rotation."""
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100,
            minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return image

        # Calculate average angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 45:  # Filter near-horizontal lines
                angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        # Rotate image to correct skew
        if abs(median_angle) > 0.5:  # Only rotate if significant skew
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

        return image

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary using adaptive thresholding."""
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    def resize_for_ocr(
        self,
        image: np.ndarray,
        min_height: int = 800
    ) -> np.ndarray:
        """Resize image for optimal OCR performance."""
        h, w = image.shape[:2]

        if h < min_height:
            scale = min_height / h
            new_w = int(w * scale)
            image = cv2.resize(
                image, (new_w, min_height),
                interpolation=cv2.INTER_CUBIC
            )

        return image

    def remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove black borders from scanned documents."""
        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Find largest contour (document)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Add small margin
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def detect_orientation(self, image: np.ndarray) -> int:
        """
        Detect document orientation.

        Returns:
            Rotation angle (0, 90, 180, 270)
        """
        # Use Tesseract OSD if available, otherwise use heuristics
        try:
            import pytesseract
            osd = pytesseract.image_to_osd(image)
            angle = int(osd.split("Rotate: ")[1].split("\n")[0])
            return angle
        except:
            # Fallback: use aspect ratio heuristics
            h, w = image.shape[:2]
            # CVs are typically portrait
            if w > h * 1.2:  # Landscape
                return 90
            return 0

    def correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """Auto-correct document orientation."""
        angle = self.detect_orientation(image)

        if angle == 0:
            return image

        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image
