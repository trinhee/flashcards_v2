#!/usr/bin/env python3
"""
pdf_to_text.py - faithful, page-by-page text extraction from course PDFs.

Pipeline position:

    PDF -> pdf_to_text.py -> raw text -> LLM preprocessing -> course notes
        -> flashcard-generation prompt -> flashcards_to_anki.py -> Anki

This tool does NOT summarize, rewrite or reorganize anything. It recovers as
much of the source text as it can, keeps page boundaries, labels how each page
was extracted, and flags pages that plain text cannot represent well, so the
downstream LLM (or you) knows where to look.

Direct extraction uses PyMuPDF (pip install pymupdf). OCR is optional and is
only used on pages where direct extraction is insufficient
(pip install pillow pytesseract, plus the Tesseract program).

Run `python pdf_to_text.py --help` for usage.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pymupdf  # PyMuPDF >= 1.24
except ImportError:
    try:
        import fitz as pymupdf  # older PyMuPDF
    except ImportError:
        pymupdf = None

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

DEFAULT_OCR_THRESHOLD = 50   # pages with fewer non-whitespace chars count as "very little text"
DEFAULT_OCR_DPI = 300
DEFAULT_IMAGE_DPI = 150
ANALYSIS_DPI = 36            # low-res render used only to measure visible "ink"
INK_DELTA = 48               # grey-level difference from the background that counts as ink
BLANK_INK_RATIO = 0.001      # below this the page is visually blank
NONTEXT_INK_RATIO = 0.02     # visible content outside text blocks
GARBAGE_RATIO = 0.10         # share of U+FFFD / private-use / control chars => garbled text layer
MIN_ALNUM_RATIO = 0.15
SUPPLEMENT_IMAGE_COVERAGE = 0.40  # images this large may contain text -> OCR as a supplement
COMPLEX_IMAGE_COVERAGE = 0.30
SIGNIFICANT_IMAGE_AREA = 0.02
COMPLEX_IMAGE_COUNT = 3
COMPLEX_DRAWING_COUNT = 50
FRAGMENTED_BLOCKS = 12
FRAGMENTED_MEDIAN_CHARS = 25
LOW_OCR_CONFIDENCE = 60.0

MATH_FONT_RE = re.compile(r"CMMI|CMSY|CMEX|CMBSY|MSAM|MSBM|Math|STIX|MTExtra|Euclid|esint|rsfs", re.I)
MATH_CHARS = set(
    "\u2211\u220f\u222b\u222e\u221a\u221e\u2248\u2260\u2264\u2265\u00b1\u2213\u00d7\u00f7\u2202\u2207"
    "\u2208\u2209\u2282\u2286\u222a\u2229\u2200\u2203\u2192\u21d2\u21d4\u2194"
    "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b8\u03bb\u03bc\u03c0\u03c1\u03c3\u03c4\u03c6\u03c7\u03c8\u03c9"
    "\u0393\u0394\u0398\u039b\u03a0\u03a3\u03a6\u03a8\u03a9"
)

INSTALL_PYMUPDF_MSG = "PyMuPDF is required. Install it with:\n    pip install pymupdf"
INSTALL_OCR_MSG = (
    "--ocr needs the Python packages Pillow and pytesseract. Install them with:\n"
    "    pip install pymupdf pillow pytesseract\n"
    "You also need the Tesseract OCR program itself (see README.md):\n"
    "    Windows: winget install UB-Mannheim.TesseractOCR\n"
    "    macOS:   brew install tesseract\n"
    "    Linux:   sudo apt install tesseract-ocr"
)
TESSERACT_MISSING_MSG = (
    "The Tesseract OCR program was not found. pytesseract is installed, but it needs the\n"
    "separate Tesseract executable:\n"
    "    Windows: winget install UB-Mannheim.TesseractOCR   (then open a new terminal)\n"
    "    macOS:   brew install tesseract\n"
    "    Linux:   sudo apt install tesseract-ocr\n"
    "If it is installed but not on PATH, pass its location with --tesseract-cmd, e.g.\n"
    '    --tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"'
)
WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)


class PdfToTextError(Exception):
    """A normal user-facing problem. Printed without a traceback."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageAnalysis:
    chars: int = 0                 # non-whitespace characters in the direct text
    garbage_ratio: float = 0.0
    alnum_ratio: float = 1.0
    ink_ratio: float = 0.0         # share of pixels that differ from the background
    nontext_ink_ratio: float = 0.0 # ...of which lie outside the text blocks
    image_count: int = 0           # images covering >= SIGNIFICANT_IMAGE_AREA of the page
    image_coverage: float = 0.0
    drawing_count: int = 0
    table_count: int = 0
    block_count: int = 0
    median_block_chars: float = 0.0
    multi_column: bool = False
    order_jumps: int = 0
    has_math: bool = False


@dataclass
class PageResult:
    number: int
    text: str = ""
    method: str = "DIRECT"         # DIRECT | OCR | DIRECT + OCR | FAILED
    ocr_text: str = ""             # only for DIRECT + OCR
    ocr_reason: str | None = None  # why OCR was needed (even if --ocr was not given)
    ocr_confidence: float | None = None
    blank: bool = False
    low_confidence: bool = False
    complex_content: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    analysis: PageAnalysis = field(default_factory=PageAnalysis)
    image_path: Path | None = None

    @property
    def needs_review(self) -> bool:
        return bool(self.warnings)

    @property
    def used_ocr(self) -> bool:
        return "OCR" in self.method

    @property
    def is_problem(self) -> bool:
        return self.needs_review or self.used_ocr or self.ocr_reason is not None


@dataclass
class Options:
    ocr: bool = False
    ocr_threshold: int = DEFAULT_OCR_THRESHOLD
    ocr_dpi: int = DEFAULT_OCR_DPI
    ocr_lang: str = "eng"
    sort: bool = False
    verbose: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def count_math(text: str) -> int:
    return sum(1 for ch in text if ch in MATH_CHARS)


def format_page_list(numbers: list[int]) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10'"""
    parts: list[str] = []
    start = prev = None
    for n in sorted(numbers):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = n
    if start is not None:
        parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def parse_page_range(spec: str, page_count: int) -> list[int]:
    """'1-5,8,10-12' -> [1, 2, 3, 4, 5, 8, 10, 11, 12]. Also accepts '7-' (to the end)."""
    pages: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d*))?", part)
        if not m:
            raise PdfToTextError(f"Invalid page range {part!r}. Use forms like 1-10, 1,3,5 or 1-5,8,10-12.")
        start = int(m.group(1))
        end = start if m.group(2) is None else (int(m.group(2)) if m.group(2) else page_count)
        if start < 1 or end < start:
            raise PdfToTextError(f"Invalid page range {part!r}.")
        if end > page_count:
            raise PdfToTextError(f"Page range {part!r} goes past the end of the document ({page_count} pages).")
        pages.update(range(start, end + 1))
    if not pages:
        raise PdfToTextError(f"No pages selected by --pages {spec!r}.")
    return sorted(pages)


def _popcount(value: int) -> int:
    return value.bit_count() if hasattr(value, "bit_count") else bin(value).count("1")


# ---------------------------------------------------------------------------
# Extraction and analysis
# ---------------------------------------------------------------------------

def clean_extracted_text(text: str) -> str:
    """Safe, non-semantic cleanup only: line endings, NUL/control characters,
    trailing spaces, runs of 3+ blank lines. Wording and line structure are kept."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # at most two consecutive blank lines
    return text.strip("\n")


def extract_page_text(page: Any, sort: bool = False) -> tuple[str, list[tuple]]:
    """Direct text-layer extraction, block by block.

    Blocks are kept in the PDF's own content order by default (sort=False). That
    order follows the author's reading order in most digital PDFs and keeps
    multi-column text intact; sorting by position (sort=True) can interleave
    columns, but helps with slides whose text boxes were stored out of order.
    Blocks are separated by a blank line; lines within a block are kept as-is.
    Ligatures (fi, fl) are expanded to plain letters; nothing else is changed.
    """
    flags = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
    blocks = page.get_text("blocks", flags=flags, sort=sort)
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    text = "\n\n".join(b[4].strip("\n") for b in text_blocks)
    return text, text_blocks


def _measure_ink(page: Any, text_blocks: list[tuple]) -> tuple[float, float]:
    """Return (ink_ratio, nontext_ink_ratio) from a low-resolution greyscale render.

    "Ink" is any pixel noticeably different from the most common (background)
    grey level, so dark-themed slides work too. Ink outside the text blocks is
    visible content that the text layer does not account for (images, vector
    diagrams, text drawn as outlines, scanned content, ...)."""
    zoom = ANALYSIS_DPI / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False)
    w, h = pix.width, pix.height
    if w == 0 or h == 0:
        return 0.0, 0.0
    samples = bytes(pix.samples)
    if pix.stride != w:
        samples = b"".join(samples[y * pix.stride: y * pix.stride + w] for y in range(h))
    hist = [samples.count(v) for v in range(256)]
    background = max(range(256), key=hist.__getitem__)
    ink = samples.translate(bytes(1 if abs(v - background) > INK_DELTA else 0 for v in range(256)))
    total_ink = ink.count(1)

    outside = bytearray(b"\x01") * (w * h)
    ox, oy, pad = page.rect.x0, page.rect.y0, 3
    for b in text_blocks:
        x0 = max(0, int((b[0] - ox - pad) * zoom))
        x1 = min(w, int((b[2] - ox + pad) * zoom) + 1)
        y0 = max(0, int((b[1] - oy - pad) * zoom))
        y1 = min(h, int((b[3] - oy + pad) * zoom) + 1)
        if x1 > x0:
            zeros = bytes(x1 - x0)
            for y in range(y0, y1):
                outside[y * w + x0: y * w + x1] = zeros
    nontext = _popcount(int.from_bytes(ink, "big") & int.from_bytes(outside, "big"))
    return total_ink / (w * h), nontext / (w * h)


def analyze_page(page: Any, text: str, text_blocks: list[tuple]) -> PageAnalysis:
    a = PageAnalysis()
    page_rect = page.rect
    page_area = max(page_rect.width * page_rect.height, 1.0)

    non_ws = [ch for ch in text if not ch.isspace()]
    a.chars = len(non_ws)
    if non_ws:
        bad = sum(1 for ch in non_ws if ch == "\ufffd" or unicodedata.category(ch) in ("Co", "Cc", "Cs"))
        a.garbage_ratio = bad / len(non_ws)
        a.alnum_ratio = sum(1 for ch in non_ws if ch.isalnum()) / len(non_ws)

    a.ink_ratio, a.nontext_ink_ratio = _measure_ink(page, text_blocks)

    covered = 0.0
    for info in page.get_image_info():
        rect = pymupdf.Rect(info["bbox"]) & page_rect
        if rect.is_empty:
            continue
        area = rect.width * rect.height / page_area
        covered += area
        if area >= SIGNIFICANT_IMAGE_AREA:
            a.image_count += 1
    a.image_coverage = min(covered, 1.0)

    try:
        a.drawing_count = len(page.get_drawings())
    except Exception:
        a.drawing_count = 0
    try:
        a.table_count = len(page.find_tables().tables)
    except Exception:
        a.table_count = 0

    sizes = [count_chars(b[4]) for b in text_blocks]
    a.block_count = len(sizes)
    a.median_block_chars = statistics.median(sizes) if sizes else 0.0

    # Two columns: substantial blocks in each half of the page that overlap vertically.
    mid = page_rect.x0 + page_rect.width / 2
    margin = page_rect.width * 0.05
    big = [b for b in text_blocks if count_chars(b[4]) >= 40]
    left = [b for b in big if b[2] <= mid + margin]
    right = [b for b in big if b[0] >= mid - margin]
    a.multi_column = len(left) >= 2 and len(right) >= 2 and any(
        l[1] < r[3] and r[1] < l[3] for l in left for r in right
    )

    # Output order jumping back up the page without moving to a column on the right.
    for prev, nxt in zip(text_blocks, text_blocks[1:]):
        if nxt[1] < prev[1] - 24 and nxt[0] < prev[2] - 10:
            a.order_jumps += 1

    fonts = " ".join(str(f[3]) for f in page.get_fonts())
    a.has_math = bool(MATH_FONT_RE.search(fonts)) or count_math(text) >= 3
    return a


def is_low_quality_text(a: PageAnalysis, threshold: int) -> str | None:
    """Return a reason if the direct text is missing, garbled, or suspiciously short."""
    if a.chars == 0:
        return None if a.ink_ratio < BLANK_INK_RATIO else "no text layer, but the page has visible content"
    if a.garbage_ratio > GARBAGE_RATIO:
        return f"text layer looks garbled ({a.garbage_ratio:.0%} unreadable characters)"
    if a.chars >= 20 and a.alnum_ratio < MIN_ALNUM_RATIO:
        return f"text layer looks garbled (only {a.alnum_ratio:.0%} letters/digits)"
    if a.chars < threshold and (a.nontext_ink_ratio >= NONTEXT_INK_RATIO or a.image_coverage >= 0.10):
        return f"only {a.chars} characters of text, but the page has other visible content"
    return None


def page_needs_ocr(a: PageAnalysis, threshold: int) -> tuple[str, str] | None:
    """Decide whether a page should be OCR'd. Returns (mode, reason) or None.

    mode "replace":    the text layer is empty or garbled, so OCR text replaces it.
    mode "supplement": the text layer is usable but incomplete (e.g. big images
                       that may contain text), so OCR text is added after it.
    """
    reason = is_low_quality_text(a, threshold)
    if reason:
        garbled = a.chars == 0 or "garbled" in reason
        return ("replace" if garbled else "supplement"), reason
    if a.image_coverage >= SUPPLEMENT_IMAGE_COVERAGE:
        return "supplement", f"images cover {a.image_coverage:.0%} of the page and may contain text"
    return None


def detect_complex_page(a: PageAnalysis) -> list[str]:
    """Reasons a page may hold content that plain text cannot represent well."""
    reasons = []
    if a.image_coverage >= COMPLEX_IMAGE_COVERAGE:
        reasons.append(f"images cover {a.image_coverage:.0%} of the page")
    elif a.image_count >= COMPLEX_IMAGE_COUNT:
        reasons.append(f"{a.image_count} images")
    if a.drawing_count >= COMPLEX_DRAWING_COUNT:
        reasons.append(f"{a.drawing_count} vector drawing elements - possible diagram, chart or flowchart")
    if a.table_count:
        reasons.append(f"{a.table_count} table(s) detected - text kept, table layout not reconstructed")
    if a.block_count >= FRAGMENTED_BLOCKS and a.median_block_chars < FRAGMENTED_MEDIAN_CHARS:
        reasons.append(f"fragmented text ({a.block_count} short text blocks) - possible diagram labels")
    return reasons


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def check_ocr_available(tesseract_cmd: str | None, lang: str) -> str:
    """Make sure pytesseract, Pillow and the Tesseract program are usable. Returns the version."""
    try:
        import pytesseract  # type: ignore[import-not-found]
        from PIL import Image  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        raise PdfToTextError(INSTALL_OCR_MSG) from None

    if tesseract_cmd:
        if not Path(tesseract_cmd).is_file():
            raise PdfToTextError(f"--tesseract-cmd file not found: {tesseract_cmd}")
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    elif not shutil.which("tesseract") and sys.platform == "win32":
        for candidate in WINDOWS_TESSERACT_PATHS:
            if Path(candidate).is_file():
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
    try:
        version = str(pytesseract.get_tesseract_version())
    except (pytesseract.TesseractNotFoundError, OSError):
        raise PdfToTextError(TESSERACT_MISSING_MSG) from None

    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception:
        installed = set()
    missing = [code for code in lang.split("+") if installed and code not in installed]
    if missing:
        raise PdfToTextError(
            f"Tesseract language data not installed: {', '.join(missing)}. "
            f"Installed: {', '.join(sorted(installed)) or 'none'}."
        )
    return version


def ocr_page(page: Any, dpi: int = DEFAULT_OCR_DPI, lang: str = "eng") -> tuple[str, float | None]:
    """Render the page and OCR it with Tesseract. Returns (text, mean word confidence 0-100).

    Text is rebuilt from Tesseract's word boxes: words joined by spaces, lines by
    newlines, paragraphs/blocks separated by a blank line."""
    import pytesseract  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), colorspace=pymupdf.csRGB, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

    lines: list[str] = []
    current_key = current_par = None
    words: list[str] = []
    confidences: list[float] = []
    for i, word in enumerate(data["text"]):
        if int(data["level"][i]) != 5 or not str(word).strip():
            continue
        par = (data["block_num"][i], data["par_num"][i])
        key = (*par, data["line_num"][i])
        if key != current_key:
            if words:
                lines.append(" ".join(words))
            if current_par is not None and par != current_par:
                lines.append("")
            words, current_key, current_par = [], key, par
        words.append(str(word))
        try:
            conf = float(data["conf"][i])
            if conf >= 0:
                confidences.append(conf)
        except (TypeError, ValueError):
            pass
    if words:
        lines.append(" ".join(words))
    mean_conf = sum(confidences) / len(confidences) if confidences else None
    return "\n".join(lines), mean_conf


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------

def process_page(page: Any, number: int, opts: Options) -> PageResult:
    result = PageResult(number=number)
    try:
        raw, blocks = extract_page_text(page, sort=opts.sort)
        result.text = clean_extracted_text(raw)
        a = result.analysis = analyze_page(page, result.text, blocks)
    except Exception as exc:  # a single damaged page should not stop the whole document
        result.method = "FAILED"
        result.warnings.append(f"DIRECT EXTRACTION FAILED ({type(exc).__name__}: {exc})")
        result.ocr_reason = "direct extraction failed"
        if opts.ocr:
            _apply_ocr(page, result, "replace", opts)
        return result

    need = page_needs_ocr(a, opts.ocr_threshold)
    if a.chars == 0 and need is None:
        result.blank = True
        result.notes.append("PAGE APPEARS BLANK")

    if need:
        mode, reason = need
        result.ocr_reason = reason
        if opts.ocr:
            _apply_ocr(page, result, mode, opts)
        elif mode == "replace":
            result.warnings.append(f"NO USABLE TEXT LAYER ({reason}) - rerun with --ocr or inspect the page image")
            result.low_confidence = True
        else:
            result.warnings.append(f"PAGE TEXT MAY BE INCOMPLETE ({reason}) - rerun with --ocr or inspect the page image")

    reasons = detect_complex_page(a)
    if reasons:
        result.complex_content = True
        result.warnings.append("PAGE MAY CONTAIN SIGNIFICANT NON-TEXT CONTENT (" + "; ".join(reasons) + ")")
    if not opts.sort and a.order_jumps >= 2 and a.order_jumps >= 0.25 * max(a.block_count - 1, 1):
        result.warnings.append("TEXT ORDER MAY NOT MATCH THE VISUAL LAYOUT (try --sort)")
    if a.multi_column:
        result.notes.append("MULTI-COLUMN LAYOUT DETECTED - text kept in the PDF's reading order")
    if a.has_math and not result.used_ocr:
        result.notes.append("CONTAINS MATHEMATICAL NOTATION - fractions, subscripts and superscripts may be flattened")
    return result


def _apply_ocr(page: Any, result: PageResult, mode: str, opts: Options) -> None:
    try:
        text, conf = ocr_page(page, opts.ocr_dpi, opts.ocr_lang)
    except Exception as exc:
        result.warnings.append(f"OCR FAILED ({type(exc).__name__}: {exc})")
        result.low_confidence = True
        return
    text = clean_extracted_text(text)
    result.ocr_confidence = conf
    if mode == "replace":
        result.text = text
        result.method = "OCR"
    else:
        result.ocr_text = text
        result.method = "DIRECT + OCR"

    ocr_chars = count_chars(text)
    problems = []
    if conf is not None and conf < LOW_OCR_CONFIDENCE:
        problems.append(f"mean OCR confidence {conf:.0f}%")
    if mode == "replace" and ocr_chars < opts.ocr_threshold:
        problems.append(f"OCR recovered only {ocr_chars} characters")
    if problems:
        result.low_confidence = True
        result.warnings.append("LOW CONFIDENCE EXTRACTION (" + "; ".join(problems) + ")")
    if result.analysis.has_math or count_math(text) >= 3:
        result.warnings.append("OCR WAS USED ON MATHEMATICAL CONTENT - formulas need manual review")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render_page(result: PageResult) -> str:
    lines = [f"--- PAGE {result.number} ---", f"[EXTRACTION: {result.method}]"]
    if result.ocr_reason and result.used_ocr:
        conf = f", confidence {result.ocr_confidence:.0f}%" if result.ocr_confidence is not None else ""
        lines.append(f"[OCR REASON: {result.ocr_reason}{conf}]")
    lines += [f"[WARNING: {w}]" for w in result.warnings]
    lines += [f"[NOTE: {n}]" for n in result.notes]
    body = [result.text] if result.text else []
    if result.method == "DIRECT + OCR":
        body.append("[OCR TEXT FROM PAGE IMAGE - may repeat text above]\n" + (result.ocr_text or "(no text recognized)"))
    return "\n".join(lines) + ("\n\n" + "\n\n".join(body) if body else "") + "\n"


def build_output(source: Path, page_count: int, results: list[PageResult]) -> str:
    header = [
        f"[SOURCE: {source.name}]",
        f"[PAGES IN DOCUMENT: {page_count} | PAGES EXTRACTED: {format_page_list([r.number for r in results])}]",
        "[RAW EXTRACTION by pdf_to_text.py - not cleaned, summarized or reordered. "
        "[WARNING] lines mark pages that may need review.]",
    ]
    return "\n".join(header) + "\n\n" + "\n".join(render_page(r) for r in results)


def write_output(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise PdfToTextError(f"Could not write {path}: {exc.strerror or exc}") from None


def save_page_image(page: Any, path: Path, dpi: int = DEFAULT_IMAGE_DPI) -> None:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False)
    try:
        pix.save(str(path))
    except Exception as exc:
        raise PdfToTextError(f"Could not save page image {path}: {exc}") from None


def print_summary(source: Path, output: Path, page_count: int, results: list[PageResult],
                  images_dir: Path | None, opts: Options) -> None:
    def pages(pred) -> list[int]:
        return [r.number for r in results if pred(r)]

    direct = pages(lambda r: r.method == "DIRECT")
    replaced = pages(lambda r: r.method == "OCR")
    supplemented = pages(lambda r: r.method == "DIRECT + OCR")
    failed = pages(lambda r: r.method == "FAILED")
    skipped_ocr = pages(lambda r: r.ocr_reason and not r.used_ocr)
    low = pages(lambda r: r.low_confidence)
    complex_pages = pages(lambda r: r.complex_content)
    blank = pages(lambda r: r.blank)
    review = pages(lambda r: r.needs_review)
    saved = pages(lambda r: r.image_path is not None)

    def fmt(nums: list[int]) -> str:
        return f"{len(nums)} page(s)" + (f"  [{format_page_list(nums)}]" if nums else "")

    print()
    print(f"Input file:            {source}")
    print(f"Pages:                 {len(results)}" + (f" of {page_count}" if len(results) != page_count else ""))
    print(f"Direct extraction:     {fmt(direct)}")
    if opts.ocr:
        print(f"OCR fallback:          {fmt(replaced + supplemented)}"
              + (f"  ({len(replaced)} replaced, {len(supplemented)} added to direct text)" if supplemented else ""))
    elif skipped_ocr:
        print(f"Would benefit from OCR: {fmt(skipped_ocr)}  -> rerun with --ocr")
    if failed:
        print(f"Extraction failed:     {fmt(failed)}")
    print(f"Low-confidence pages:  {fmt(low)}")
    print(f"Non-text content:      {fmt(complex_pages)}")
    if blank:
        print(f"Blank pages:           {fmt(blank)}")
    if images_dir is not None:
        print(f"Problem-page images saved: {len(saved)}" + (f"  (in {images_dir})" if saved else ""))
    print(f"Output:                {output}")
    print()
    print(f"Pages needing review:  {format_page_list(review) if review else 'none'}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def open_pdf(path: Path, password: str | None) -> Any:
    if not path.exists():
        raise PdfToTextError(f"Input file not found: {path}")
    if path.is_dir():
        raise PdfToTextError(f"Input path is a directory, not a PDF: {path}")
    try:
        with open(path, "rb") as fh:
            head = fh.read(1024)
    except OSError as exc:
        raise PdfToTextError(f"Could not read {path}: {exc.strerror or exc}") from None
    if b"%PDF-" not in head:
        raise PdfToTextError(f"{path} does not look like a PDF file (no %PDF header).")
    try:
        doc = pymupdf.open(str(path), filetype="pdf")
    except Exception as exc:
        raise PdfToTextError(f"Could not open {path} - the PDF appears to be corrupted ({exc}).") from None
    if doc.needs_pass:
        if not password:
            raise PdfToTextError(f"{path} is password-protected. Supply the password with --password.")
        if not doc.authenticate(password):
            raise PdfToTextError("The password given with --password is not correct for this PDF.")
    if doc.page_count == 0:
        raise PdfToTextError(f"{path} contains no pages (the file may be damaged).")
    return doc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_to_text.py",
        description=(
            "Extract the raw text of a course PDF page by page, for later cleanup by an LLM. "
            "Nothing is summarized or rewritten. Each page is labelled with how it was extracted, "
            "and pages that may need review (scans, diagrams, tables, image-heavy slides) are flagged."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python pdf_to_text.py lecture.pdf                       -> lecture_raw.txt
  python pdf_to_text.py lecture.pdf --output notes/week3.txt
  python pdf_to_text.py lecture.pdf --ocr                 (OCR only the pages that need it)
  python pdf_to_text.py lecture.pdf --save-problem-pages  (PNG of each flagged page)
  python pdf_to_text.py lecture.pdf --ocr --save-problem-pages --pages 1-5,8,10-12

exit codes:
  0  success (check "Pages needing review" in the summary)
  1  error (file missing, not a PDF, encrypted, corrupted, cannot write, OCR unavailable)
""",
    )
    parser.add_argument("input", help="PDF file to extract")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="text file to write (default: <input name>_raw.txt next to the PDF)")
    parser.add_argument("--ocr", action="store_true",
                        help="OCR pages whose text layer is missing, garbled or incomplete "
                             "(needs: pip install pillow pytesseract, plus the Tesseract program)")
    parser.add_argument("--save-problem-pages", action="store_true",
                        help="save a PNG of every flagged / OCR'd page into <input name>_problem_pages/")
    parser.add_argument("--pages", metavar="RANGE", help="only these pages, e.g. 1-10 or 1,3,5 or 1-5,8,10-12")
    parser.add_argument("--ocr-threshold", type=int, default=DEFAULT_OCR_THRESHOLD, metavar="N",
                        help=f"pages with fewer than N non-whitespace characters plus other visible content "
                             f"are treated as weak extractions (default {DEFAULT_OCR_THRESHOLD})")
    parser.add_argument("--ocr-lang", default="eng", metavar="LANG",
                        help="Tesseract language(s), e.g. eng or eng+fra (default eng)")
    parser.add_argument("--ocr-dpi", type=int, default=DEFAULT_OCR_DPI, metavar="DPI",
                        help=f"render resolution for OCR (default {DEFAULT_OCR_DPI})")
    parser.add_argument("--image-dpi", type=int, default=DEFAULT_IMAGE_DPI, metavar="DPI",
                        help=f"resolution of saved problem-page images (default {DEFAULT_IMAGE_DPI})")
    parser.add_argument("--tesseract-cmd", metavar="PATH", help="path to tesseract(.exe) if it is not on PATH")
    parser.add_argument("--sort", action="store_true",
                        help="order text blocks top-to-bottom, left-to-right instead of the PDF's own order "
                             "(can fix scrambled slides; can break multi-column pages)")
    parser.add_argument("--password", metavar="PW", help="password for an encrypted PDF")
    parser.add_argument("-v", "--verbose", action="store_true", help="print per-page details")
    return parser


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def run(args: argparse.Namespace) -> int:
    if pymupdf is None:
        raise PdfToTextError(INSTALL_PYMUPDF_MSG)
    # MuPDF's own warnings and PyMuPDF's package suggestions are noise for most users.
    for quiet in (lambda: pymupdf.TOOLS.mupdf_display_errors(args.verbose),
                  lambda: pymupdf.no_recommend_layout(),
                  lambda: pymupdf.set_messages(stream=sys.stderr)):
        try:
            quiet()
        except Exception:
            pass

    source = Path(args.input)
    output = Path(args.output) if args.output else source.with_name(f"{source.stem}_raw.txt")
    if output.suffix.lower() == ".pdf" or (source.exists() and output.resolve() == source.resolve()):
        raise PdfToTextError("The output would overwrite a PDF. Choose a .txt file for --output.")
    if not output.parent.exists():
        raise PdfToTextError(f"Output directory does not exist: {output.parent}")
    opts = Options(ocr=args.ocr, ocr_threshold=args.ocr_threshold, ocr_dpi=args.ocr_dpi,
                   ocr_lang=args.ocr_lang, sort=args.sort, verbose=args.verbose)

    doc = open_pdf(source, args.password)
    page_numbers = parse_page_range(args.pages, doc.page_count) if args.pages else list(range(1, doc.page_count + 1))
    if args.ocr:
        version = check_ocr_available(args.tesseract_cmd, args.ocr_lang)
        if args.verbose:
            print(f"Using Tesseract {version}")

    results: list[PageResult] = []
    show_progress = sys.stderr.isatty() and not args.verbose and len(page_numbers) > 5
    for i, number in enumerate(page_numbers, start=1):
        if show_progress:
            print(f"\rProcessing page {i}/{len(page_numbers)}...", end="", file=sys.stderr, flush=True)
        result = process_page(doc[number - 1], number, opts)
        results.append(result)
        if args.verbose:
            a = result.analysis
            print(f"page {number:>4}: {result.method:<12} chars={a.chars:<6} ink={a.ink_ratio:.1%} "
                  f"non-text ink={a.nontext_ink_ratio:.1%} images={a.image_count} ({a.image_coverage:.0%}) "
                  f"drawings={a.drawing_count} tables={a.table_count} blocks={a.block_count}"
                  + (f" ocr-conf={result.ocr_confidence:.0f}" if result.ocr_confidence is not None else "")
                  + (f"  ! {len(result.warnings)} warning(s)" if result.warnings else ""))
    if show_progress:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    write_output(output, build_output(source, doc.page_count, results))

    images_dir = None
    if args.save_problem_pages:
        images_dir = output.parent / f"{source.stem}_problem_pages"
        problems = [r for r in results if r.is_problem]
        if problems:
            try:
                images_dir.mkdir(exist_ok=True)
                for old in images_dir.glob("page_*.png"):  # stale images from an earlier run
                    old.unlink()
            except OSError as exc:
                raise PdfToTextError(f"Could not prepare {images_dir}: {exc.strerror or exc}") from None
            digits = max(3, len(str(doc.page_count)))
            for r in problems:
                r.image_path = images_dir / f"page_{r.number:0{digits}d}.png"
                save_page_image(doc[r.number - 1], r.image_path, args.image_dpi)

    print_summary(source, output, doc.page_count, results, images_dir, opts)
    doc.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("ocr_threshold", "ocr_dpi", "image_dpi"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive number")
    try:
        return run(args)
    except PdfToTextError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
