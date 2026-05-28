"""
ocr.py  –  Extract raw text from PDF or image files.

Returns an OcrResult containing:
  - full_text : the complete document text (single string)
  - page_map  : list of (page_num, char_start, char_end) tuples so callers
                can map a character offset back to a page number
  - source    : 'pdfplumber' | 'tesseract'
  - warnings  : list of strings
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class OcrResult:
    full_text: str = ""
    page_map: List[Tuple[int, int, int]] = field(default_factory=list)
    source: str = "unknown"
    warnings: List[str] = field(default_factory=list)

    def page_for_offset(self, offset: int) -> int:
        """Return the 1-based page number for a character offset."""
        for page_num, start, end in self.page_map:
            if start <= offset < end:
                return page_num
        return -1


def extract(file_path: str | Path) -> OcrResult:
    """
    Main entry point.  Detects file type and dispatches to the right engine.
    PDF  → try pdfplumber (text-layer), fallback to Tesseract via pdf2image
    Image → Tesseract directly
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _from_pdf(path)
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return _from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ──────────────────────────────────────────────────────────────────────────────
# PDF extraction
# ──────────────────────────────────────────────────────────────────────────────

def _from_pdf(path: Path) -> OcrResult:
    """Try pdfplumber first (preserves text layer).  Fall back to Tesseract."""
    try:
        import pdfplumber
        result = _pdfplumber_extract(path)
        # If we got very little text, the PDF is probably scanned → use OCR
        if len(result.full_text.strip()) < 50:
            result.warnings.append(
                "pdfplumber returned very little text; switching to Tesseract OCR."
            )
            return _tesseract_pdf(path)
        return result
    except ImportError:
        pass

    return _tesseract_pdf(path)


def _pdfplumber_extract(path: Path) -> OcrResult:
    import pdfplumber

    pages_text: list[str] = []
    warnings: list[str] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages_text.append(txt)

    # Build full_text and page_map
    full_text = ""
    page_map: list[tuple[int, int, int]] = []
    for i, txt in enumerate(pages_text, start=1):
        start = len(full_text)
        full_text += txt + "\n"
        end = len(full_text)
        page_map.append((i, start, end))

    return OcrResult(
        full_text=full_text,
        page_map=page_map,
        source="pdfplumber",
        warnings=warnings,
    )


def _tesseract_pdf(path: Path) -> OcrResult:
    """Convert PDF pages to images then run Tesseract."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise RuntimeError(
            "pdf2image is required for scanned-PDF OCR.  "
            "Install it with: pip install pdf2image  (also needs poppler-utils)"
        )
    
    poppler_path = os.environ.get("POPPLER_PATH")
    
    images = convert_from_path(str(path), dpi=300, poppler_path=poppler_path)
    full_text = ""
    page_map: list[tuple[int, int, int]] = []
    warnings: list[str] = []

    for i, img in enumerate(images, start=1):
        txt = _tesseract_image(img)
        start = len(full_text)
        full_text += txt + "\n"
        end = len(full_text)
        page_map.append((i, start, end))

    return OcrResult(
        full_text=full_text,
        page_map=page_map,
        source="tesseract",
        warnings=warnings,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Image extraction
# ──────────────────────────────────────────────────────────────────────────────

def _from_image(path: Path) -> OcrResult:
    from PIL import Image

    img = Image.open(path)
    txt = _tesseract_image(img)
    page_map = [(1, 0, len(txt))]
    return OcrResult(
        full_text=txt,
        page_map=page_map,
        source="tesseract",
        warnings=[],
    )


def _tesseract_image(img) -> str:
    try:
        import pytesseract
        # Προσθήκη για ανάγνωση του TESSERACT_CMD στα Windows
        tesseract_cmd = os.environ.get("TESSERACT_CMD")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        return pytesseract.image_to_string(img, config="--psm 6")
    except ImportError:
        raise RuntimeError(
            "pytesseract is required for image OCR.  "
            "Install it with: pip install pytesseract  (also needs tesseract-ocr binary)"
        )