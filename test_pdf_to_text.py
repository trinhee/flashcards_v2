"""Tests for pdf_to_text.py. Run with:  python -m unittest -v

Test PDFs are generated on the fly with PyMuPDF. Real OCR needs the Tesseract
program, so OCR behaviour is tested with a stand-in pytesseract module; one test
runs real Tesseract when it is installed.
"""

import io
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pdf_to_text as p2t

pymupdf = p2t.pymupdf
needs_pymupdf = unittest.skipIf(pymupdf is None, "PyMuPDF not installed")

try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

LECTURE_HTML = """
<h1>Lecture 3: Relational Databases</h1>
<p>A <b>relation</b> is a set of tuples. Each tuple has the same attributes.</p>
<ul><li>Primary key: uniquely identifies a row</li><li>Foreign key: references another table</li></ul>
<ol><li>First normal form</li><li>Second normal form</li></ol>
<p>Entropy: H(X) = &#8722;&#8721; p(x) log p(x), and &#963;&#178; &#8805; 0 for all &#952; &#8712; &#937;.</p>
<pre>SELECT name FROM students WHERE gpa &gt; 3.5;</pre>
<p>Figure 1: caption text for an ER diagram. Caf&#233; na&#239;ve r&#233;sum&#233;.</p>
"""


def _text_page(doc, html, rect=None):
    page = doc.new_page(width=612, height=792)
    page.insert_htmlbox(rect or pymupdf.Rect(50, 50, 562, 742), html)
    return page


def _text_as_image(text, width=500, height=300):
    """PNG bytes of a picture that *shows* text (no text layer)."""
    tmp = pymupdf.open()
    page = tmp.new_page(width=width, height=height)
    page.insert_htmlbox(pymupdf.Rect(20, 20, width - 20, height - 20), f"<p style='font-size:22px'>{text}</p>")
    png = page.get_pixmap(dpi=150).tobytes("png")
    tmp.close()
    return png


def make_text_pdf(path):
    doc = pymupdf.open()
    _text_page(doc, LECTURE_HTML)
    _text_page(doc, "<h2>Page two heading</h2><p>Second page paragraph with enough words to be real content.</p>")
    page = doc.new_page(width=612, height=792)  # two columns
    left = " ".join(f"Left column sentence {i} about transactions." for i in range(12))
    right = " ".join(f"Right column sentence {i} about indexing." for i in range(12))
    page.insert_htmlbox(pymupdf.Rect(40, 60, 296, 400), f"<p>{left}</p><p>{left}</p>")
    page.insert_htmlbox(pymupdf.Rect(316, 60, 572, 400), f"<p>{right}</p><p>{right}</p>")
    doc.save(path)


def make_mixed_pdf(path):
    doc = pymupdf.open()
    _text_page(doc, "<h1>Normal page</h1><p>Plain lecture text that extracts directly without problems.</p>")
    # 2: small title + large picture of text
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 60), "Slide: Query plans", fontsize=20)
    page.insert_image(pymupdf.Rect(50, 100, 562, 520), stream=_text_as_image("Hash join beats nested loop for big inputs"))
    # 3: vector diagram (flowchart) with short labels
    page = doc.new_page(width=612, height=792)
    for i in range(14):
        y = 60 + i * 48
        page.draw_rect(pymupdf.Rect(100, y, 220, y + 30), color=(0, 0, 0), fill=(0.8, 0.9, 1))
        page.draw_line((220, y + 15), (380, y + 15))
        page.draw_rect(pymupdf.Rect(380, y, 500, y + 30), color=(0, 0, 0))
        page.draw_line((160, y + 30), (160, y + 48))
        page.insert_text((110, y + 20), f"Step {i}", fontsize=10)
        page.insert_text((390, y + 20), f"Out {i}", fontsize=10)
    # 4: ruled table
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 70), "Comparison", fontsize=16)
    xs, ys = [60, 220, 380, 540], [90 + 26 * r for r in range(9)]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    for r in range(8):
        for c, cell in enumerate((f"Engine {r}", f"{r * 10} ms", "B-tree")):
            page.insert_text((xs[c] + 6, ys[r] + 18), cell, fontsize=11)
    # 5: blank
    doc.new_page(width=612, height=792)
    # 6: dark slide with white text
    page = doc.new_page(width=792, height=612)
    page.draw_rect(page.rect, fill=(0.1, 0.1, 0.15), color=None)
    page.insert_htmlbox(pymupdf.Rect(60, 60, 732, 552),
                        "<h1 style='color:white'>Dark theme slide</h1><p style='color:white'>"
                        "White text on a dark background should still extract directly.</p>")
    doc.save(path)


def make_scanned_pdf(path):
    src = pymupdf.open()
    _text_page(src, "<h1>Scanned page one</h1><p>This text only exists as pixels.</p>")
    _text_page(src, "<h1>Scanned page two</h1><p>More pixels, no text layer.</p>")
    doc = pymupdf.open()
    for page in src:
        png = page.get_pixmap(dpi=150).tobytes("png")
        new = doc.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=png)
    doc.save(path)


def fake_pytesseract(text="RECOGNISED TEXT LINE ONE\nLINE TWO", conf=91.0, found=True):
    """A stand-in pytesseract module that records how many pages it was asked to OCR."""
    mod = types.ModuleType("pytesseract")
    mod.calls = 0

    class TesseractNotFoundError(EnvironmentError):
        pass

    mod.TesseractNotFoundError = TesseractNotFoundError
    mod.pytesseract = types.SimpleNamespace(tesseract_cmd="tesseract")
    mod.Output = types.SimpleNamespace(DICT="dict")

    def get_tesseract_version():
        if not found:
            raise TesseractNotFoundError()
        return "5.4.0"

    def image_to_data(image, lang="eng", output_type=None):
        mod.calls += 1
        data = {k: [] for k in ("level", "block_num", "par_num", "line_num", "text", "conf")}
        for line_no, line in enumerate(text.split("\n"), start=1):
            for word in line.split():
                for k, v in zip(data, (5, 1, 1, line_no, word, conf)):
                    data[k].append(v)
        return data

    mod.get_tesseract_version = get_tesseract_version
    mod.get_languages = lambda config="": ["eng", "osd"]
    mod.image_to_data = image_to_data
    return mod


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = p2t.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def split_pages(text):
    """{page_number: page_section_text} from the output file."""
    pages = {}
    for chunk in text.split("--- PAGE ")[1:]:
        number, rest = chunk.split(" ---", 1)
        pages[int(number)] = rest
    return pages


class Helpers(unittest.TestCase):
    def test_parse_page_range(self):
        self.assertEqual(p2t.parse_page_range("1-3", 10), [1, 2, 3])
        self.assertEqual(p2t.parse_page_range("1,3,5", 10), [1, 3, 5])
        self.assertEqual(p2t.parse_page_range("1-5,8,10-12", 12), [1, 2, 3, 4, 5, 8, 10, 11, 12])
        self.assertEqual(p2t.parse_page_range("9-", 10), [9, 10])
        for bad in ("0", "5-2", "a", "1-20", "3,,x"):
            with self.assertRaises(p2t.PdfToTextError):
                p2t.parse_page_range(bad, 10)

    def test_format_page_list(self):
        self.assertEqual(p2t.format_page_list([8, 1, 2, 3, 13, 9]), "1-3, 8-9, 13")

    def test_clean_is_conservative(self):
        raw = "Title  \r\n\r\n\r\n\r\n\r\n- bullet\x00 one\t(tab)\r\n    indented code\n\x0c"
        self.assertEqual(p2t.clean_extracted_text(raw), "Title\n\n\n- bullet one\t(tab)\n    indented code")
        repeated = "Header\nHeader\nx = y² + ∑ z"
        self.assertEqual(p2t.clean_extracted_text(repeated), repeated)


@needs_pymupdf
class Extraction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_pdf_direct(self):
        pdf = self.dir / "lecture.pdf"
        make_text_pdf(pdf)
        code, out, err = run_cli(pdf)
        self.assertEqual(code, 0, err)
        result = self.dir / "lecture_raw.txt"           # default name
        text = result.read_bytes().decode("utf-8")       # valid UTF-8
        pages = split_pages(text)
        self.assertEqual(sorted(pages), [1, 2, 3])
        self.assertEqual(text.count("--- PAGE "), 3)
        self.assertTrue(all("[EXTRACTION: DIRECT]" in p for p in pages.values()))
        p1 = pages[1]
        for expected in ("Lecture 3: Relational Databases", "Primary key: uniquely identifies a row",
                         "SELECT name FROM students WHERE gpa > 3.5;", "∑", "σ", "∈",
                         "Café naïve résumé", "Figure 1: caption"):
            self.assertIn(expected, p1)
        self.assertIn("MATHEMATICAL NOTATION", p1)
        self.assertNotIn("[WARNING", p1)
        self.assertIn("MULTI-COLUMN", pages[3])
        # stream order keeps the left column before the right column
        self.assertLess(pages[3].rindex("transactions"), pages[3].index("indexing"))
        self.assertIn("Pages needing review:  none", out)

    def test_mixed_pdf_flags_problem_pages(self):
        pdf = self.dir / "mixed.pdf"
        make_mixed_pdf(pdf)
        code, out, err = run_cli(pdf, "--save-problem-pages", "-o", self.dir / "custom name.txt")
        self.assertEqual(code, 0, err)
        pages = split_pages((self.dir / "custom name.txt").read_text(encoding="utf-8"))
        self.assertNotIn("[WARNING", pages[1])
        self.assertIn("Slide: Query plans", pages[2])
        self.assertIn("NON-TEXT CONTENT", pages[2])
        self.assertIn("rerun with --ocr", pages[2])
        self.assertIn("vector drawing elements", pages[3])
        self.assertIn("Step 13", pages[3])
        self.assertIn("table(s) detected", pages[4])
        self.assertIn("Engine 7", pages[4])
        self.assertIn("PAGE APPEARS BLANK", pages[5])
        self.assertNotIn("[WARNING", pages[5])
        self.assertIn("Dark theme slide", pages[6])
        self.assertNotIn("[WARNING", pages[6])
        self.assertIn("Pages needing review:  2-4", out)
        images = sorted(p.name for p in (self.dir / "mixed_problem_pages").iterdir())
        self.assertEqual(images, ["page_002.png", "page_003.png", "page_004.png"])

    def test_scanned_pdf_without_ocr_is_flagged(self):
        pdf = self.dir / "scan.pdf"
        make_scanned_pdf(pdf)
        code, out, _ = run_cli(pdf)
        self.assertEqual(code, 0)
        pages = split_pages((self.dir / "scan_raw.txt").read_text(encoding="utf-8"))
        for p in pages.values():
            self.assertIn("NO USABLE TEXT LAYER", p)
        self.assertIn("Would benefit from OCR: 2 page(s)", out)
        self.assertIn("Pages needing review:  1-2", out)

    @unittest.skipUnless(HAVE_PIL, "Pillow not installed")
    def test_ocr_runs_only_where_needed(self):
        pdf = self.dir / "mixed.pdf"
        make_mixed_pdf(pdf)
        fake = fake_pytesseract()
        with mock.patch.dict(sys.modules, {"pytesseract": fake}):
            code, out, err = run_cli(pdf, "--ocr")
        self.assertEqual(code, 0, err)
        self.assertEqual(fake.calls, 1)  # only the image-heavy page 2
        pages = split_pages((self.dir / "mixed_raw.txt").read_text(encoding="utf-8"))
        self.assertIn("[EXTRACTION: DIRECT + OCR]", pages[2])
        self.assertIn("Slide: Query plans", pages[2])     # direct text kept
        self.assertIn("RECOGNISED TEXT LINE ONE\nLINE TWO", pages[2])
        for n in (1, 3, 4, 5, 6):
            self.assertIn("[EXTRACTION: DIRECT]", pages[n])
        self.assertIn("OCR fallback:          1 page(s)", out)

    @unittest.skipUnless(HAVE_PIL, "Pillow not installed")
    def test_scanned_pdf_with_ocr(self):
        pdf = self.dir / "scan.pdf"
        make_scanned_pdf(pdf)
        fake = fake_pytesseract(text="Scanned page one", conf=40.0)
        with mock.patch.dict(sys.modules, {"pytesseract": fake}):
            code, out, _ = run_cli(pdf, "--ocr", "--save-problem-pages")
        self.assertEqual(code, 0)
        self.assertEqual(fake.calls, 2)
        pages = split_pages((self.dir / "scan_raw.txt").read_text(encoding="utf-8"))
        self.assertIn("[EXTRACTION: OCR]", pages[1])
        self.assertIn("LOW CONFIDENCE EXTRACTION", pages[1])
        self.assertIn("Low-confidence pages:  2 page(s)", out)
        self.assertEqual(len(list((self.dir / "scan_problem_pages").glob("page_*.png"))), 2)

    @unittest.skipUnless(shutil.which("tesseract") and HAVE_PIL, "Tesseract not installed")
    def test_real_tesseract(self):
        pdf = self.dir / "scan.pdf"
        make_scanned_pdf(pdf)
        code, _, err = run_cli(pdf, "--ocr")
        self.assertEqual(code, 0, err)
        self.assertIn("Scanned page one", (self.dir / "scan_raw.txt").read_text(encoding="utf-8"))

    def test_pages_option(self):
        pdf = self.dir / "lecture.pdf"
        make_text_pdf(pdf)
        code, _, _ = run_cli(pdf, "--pages", "1,3")
        self.assertEqual(code, 0)
        text = (self.dir / "lecture_raw.txt").read_text(encoding="utf-8")
        self.assertEqual(sorted(split_pages(text)), [1, 3])
        self.assertIn("PAGES EXTRACTED: 1, 3", text)


@needs_pymupdf
class Errors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assertCleanError(self, args, expected):
        code, _, err = run_cli(*args)
        self.assertEqual(code, 1)
        self.assertIn(expected, err)
        self.assertNotIn("Traceback", err)

    def test_missing_file(self):
        self.assertCleanError([self.dir / "nope.pdf"], "not found")

    def test_not_a_pdf(self):
        fake = self.dir / "notes.pdf"
        fake.write_text("just text", encoding="utf-8")
        self.assertCleanError([fake], "does not look like a PDF")

    def test_corrupted(self):
        bad = self.dir / "bad.pdf"
        bad.write_bytes(b"%PDF-1.7\n" + b"\x00garbage" * 50)
        self.assertCleanError([bad], "corrupted")

    def test_encrypted(self):
        pdf = self.dir / "locked.pdf"
        doc = pymupdf.open()
        _text_page(doc, "<p>secret</p>")
        doc.save(pdf, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
        self.assertCleanError([pdf], "password-protected")
        self.assertCleanError([pdf, "--password", "wrong"], "not correct")
        self.assertEqual(run_cli(pdf, "--password", "u")[0], 0)

    def test_unwritable_output(self):
        pdf = self.dir / "lecture.pdf"
        make_text_pdf(pdf)
        self.assertCleanError([pdf, "-o", self.dir / "missing_dir" / "x.txt"], "does not exist")

    def test_ocr_python_packages_missing(self):
        pdf = self.dir / "lecture.pdf"
        make_text_pdf(pdf)
        with mock.patch.dict(sys.modules, {"pytesseract": None}):
            self.assertCleanError([pdf, "--ocr"], "pip install pymupdf pillow pytesseract")

    @unittest.skipUnless(HAVE_PIL, "Pillow not installed")
    def test_tesseract_program_missing(self):
        pdf = self.dir / "lecture.pdf"
        make_text_pdf(pdf)
        with mock.patch.dict(sys.modules, {"pytesseract": fake_pytesseract(found=False)}), \
                mock.patch.object(p2t, "WINDOWS_TESSERACT_PATHS", ()), \
                mock.patch.object(p2t.shutil, "which", return_value=None):
            self.assertCleanError([pdf, "--ocr"], "Tesseract OCR program was not found")

    def test_pymupdf_missing(self):
        with mock.patch.object(p2t, "pymupdf", None):
            self.assertCleanError([self.dir / "x.pdf"], "pip install pymupdf")


if __name__ == "__main__":
    unittest.main()
