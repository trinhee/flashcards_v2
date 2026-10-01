# Course PDFs → Anki Flashcards

Two small Python tools that turn course material into multiple-choice Anki cards, with an LLM doing the reading and writing in between:

```
PDF ──► pdf_to_text.py ──► raw text ──► LLM preprocessing prompt ──► structured course notes
                                                                            │
Anki ◄── flashcards_to_anki.py ◄── flashcards.txt ◄── flashcard-generation prompt
```

| Step | Who does it | Job |
|---|---|---|
| 1. Raw extraction | `pdf_to_text.py` | Get **all** of the text out of the PDF, page by page, and flag pages that need a closer look. No interpretation. |
| 2. Cleanup / structuring | LLM (preprocessing prompt) | Fix reading order, rebuild headings and tables, drop slide boilerplate, organize topics |
| 3. Flashcard generation | LLM (flashcard prompt) | Write the multiple-choice cards as a Python list |
| 4. Anki conversion | `flashcards_to_anki.py` | Check every card and export a TSV (or `.apkg`) for Anki |

Each tool does one job. The scripts don't interpret content, and the LLM steps don't have to deal with file formats.

## Files

| File | Purpose |
|---|---|
| `pdf_to_text.py` | Step 1: PDF → raw text, with optional OCR fallback |
| `flashcards_to_anki.py` | Step 4: LLM flashcards → Anki TSV / `.apkg` |
| `anki_templates/front_template.html` | Front template for the Anki note type |
| `anki_templates/back_template.html` | Back template for the Anki note type |
| `anki_templates/styling.css` | Card styling (light/dark mode, desktop/mobile) |
| `samples/` | Example flashcard inputs: valid, malformed, and a syntax error |
| `test_pdf_to_text.py`, `test_flashcards_to_anki.py` | Test suites (`python -m unittest -v`) |
| `requirements.txt` | Python packages (see installation) |

## Installation

Requires Python 3.9+.

```
pip install -r requirements.txt
```

| Package | Needed for | Required? |
|---|---|---|
| `pymupdf` | `pdf_to_text.py` | Yes, for PDF extraction |
| `pillow`, `pytesseract` | `pdf_to_text.py --ocr` | Optional |
| `genanki` | `flashcards_to_anki.py --apkg` | Optional |

`flashcards_to_anki.py` needs nothing beyond the standard library for its default TSV export.

OCR also needs the **Tesseract program**, which pip cannot install. See [Installing Tesseract](#installing-tesseract-for---ocr).

---

# Part 1: `pdf_to_text.py`, PDF → raw text

## What it does (and doesn't do)

It extracts the text of every page **as faithfully as possible**:
- It keeps page boundaries.
- It labels how each page was extracted.
- It flags pages that plain text can't represent well.

It does **not** summarize, paraphrase, reorder, merge lines, remove repeated headers, fix math, or describe diagrams. Those jobs belong to the LLM preprocessing step, and that step works better when it gets the complete, unaltered text.

The only cleanup it does is safe and mechanical:
- normalizing line endings
- removing NUL and other control characters
- trimming trailing spaces
- collapsing runs of more than two blank lines
- expanding typographic ligatures (`ﬁ` → `fi`)

## Normal extraction

```
python pdf_to_text.py lecture.pdf
```

This writes `lecture_raw.txt` next to the PDF (or use `--output custom_name.txt`) and prints a summary:

```
Input file:            lecture.pdf
Pages:                 42
Direct extraction:     36 page(s)  [1-7, 9-12, 14-20, 22-42]
OCR fallback:          4 page(s)  [8, 13, 21, 30]  (2 replaced, 2 added to direct text)
Low-confidence pages:  2 page(s)  [8, 13]
Non-text content:      5 page(s)  [8, 13, 17, 21, 33]
Problem-page images saved: 6  (in lecture_problem_pages)
Output:                lecture_raw.txt

Pages needing review:  8, 13, 17, 21, 33
```

### Output format

```
[SOURCE: lecture.pdf]
[PAGES IN DOCUMENT: 42 | PAGES EXTRACTED: 1-42]
[RAW EXTRACTION by pdf_to_text.py - not cleaned, summarized or reordered. ...]

--- PAGE 1 ---
[EXTRACTION: DIRECT]

Lecture 3: Relational Databases

•  Primary key: uniquely identifies a row
•  Foreign key: references another table

--- PAGE 8 ---
[EXTRACTION: OCR]
[OCR REASON: no text layer, but the page has visible content, confidence 54%]
[WARNING: LOW CONFIDENCE EXTRACTION (mean OCR confidence 54%)]
[WARNING: OCR WAS USED ON MATHEMATICAL CONTENT - formulas need manual review]

...OCR text...

--- PAGE 17 ---
[EXTRACTION: DIRECT]
[WARNING: PAGE MAY CONTAIN SIGNIFICANT NON-TEXT CONTENT (63 vector drawing elements - possible diagram, chart or flowchart)]

...diagram labels...
```

| Label | Meaning |
|---|---|
| `[EXTRACTION: DIRECT]` | Text came from the PDF's own text layer. This is the most accurate source. |
| `[EXTRACTION: OCR]` | The page had no usable text layer, so the text was read from an image of the page. |
| `[EXTRACTION: DIRECT + OCR]` | The text layer was kept, and OCR of the full page was added after it under `[OCR TEXT FROM PAGE IMAGE …]`. Used when large images may contain text. Some text may appear twice. |
| `[EXTRACTION: FAILED]` | The page could not be read (damaged page). The rest of the document still processes. |
| `[WARNING: …]` | The page may be incomplete or misleading. It's listed under "Pages needing review". |
| `[NOTE: …]` | Information only: blank page, multi-column layout, or math notation whose layout may be flattened. |

Each block of text on the page is followed by a blank line. Headings, bullets, numbered lists, code, captions, labels and formulas keep the characters the PDF contains.

### How weak pages are detected

An empty extraction is **not** assumed to mean a blank page. For every page the tool measures:
- **Text quality.** It counts characters and checks for garbage: replacement characters, private-use glyphs, or almost no letters or digits (a broken font encoding).
- **Visible "ink".** It renders the page at low resolution and measures how much of it differs from the background. This works on dark-themed slides too. In particular, it measures how much ink lies **outside** the text blocks.
- **Images.** It counts them and measures how much of the page they cover.
- **Vector drawings.** A high count means a diagram, chart or flowchart.
- **Tables.** Ruled tables are detected with PyMuPDF's table finder.
- **Fragmentation.** Many tiny text blocks usually means diagram labels.

From these signals:

| Situation | Result |
|---|---|
| No text, no ink | `[NOTE: PAGE APPEARS BLANK]` |
| No text but visible content (scan, image slide) | Needs OCR (replace) |
| Garbled text layer | Needs OCR (replace) |
| Fewer than `--ocr-threshold` characters (default 50) plus other visible content | Needs OCR (supplement) |
| Images cover ≥ 40% of the page | Needs OCR (supplement), because the images may contain text |
| Many images, image-heavy, many drawing elements, a table, or fragmented text | `[WARNING: PAGE MAY CONTAIN SIGNIFICANT NON-TEXT CONTENT (…)]` |

Without `--ocr`, pages that need OCR are flagged `… rerun with --ocr …`, and the summary lists them under "Would benefit from OCR".

### Reading order

Text blocks are output in the **order stored in the PDF**. I tested this against sorting blocks by position (`--sort`), and stream order is the better default:
- On a two-column page, sorting by position interleaves the columns (left 1, right 1, left 2, …). Stream order keeps each column intact.
- Most digital PDFs, including exported slides, store text in the author's reading order.

Some slide decks store their text boxes out of order. The tool detects text that repeatedly jumps back up the page and flags it with `[WARNING: TEXT ORDER MAY NOT MATCH THE VISUAL LAYOUT (try --sort)]`. Re-run those files with `--sort`. Either way, the LLM preprocessing step can fix minor ordering issues.

### Tables and formulas

- **Tables:** the cell text is extracted like any other text, usually one row per line. The table layout is **not** reconstructed, and the page is flagged when a table is detected. The LLM can usually rebuild the table from the text. If not, send it the page image.
- **Formulas:** text-layer math is kept exactly as the PDF encodes it (`H(X) = −∑ p(x) log p(x)`, `σ² ≥ 0`), with no conversion to LaTeX. Fractions, subscripts and superscripts may be flattened onto one line, so such pages get a `[NOTE]`. OCR of math is unreliable, so OCR'd pages with math get a `[WARNING]`.

## OCR fallback (`--ocr`)

```
python pdf_to_text.py lecture.pdf --ocr
```

The tool always tries direct extraction first. It only renders a page at 300 DPI and runs Tesseract on it when that page fails the checks above, so a normal digital PDF doesn't get OCR'd at all. Direct text is preferred because it is exact, while OCR can misread characters. A page whose OCR confidence is below 60%, or that still yields almost no text, gets `[WARNING: LOW CONFIDENCE EXTRACTION]`.

| Option | Default | Purpose |
|---|---|---|
| `--ocr-lang eng+fra` | `eng` | Tesseract language(s). The language data must be installed. |
| `--ocr-dpi 400` | `300` | Higher can help with small print, but is slower |
| `--ocr-threshold 80` | `50` | Character count below which a page with other visible content counts as weak |
| `--tesseract-cmd PATH` | auto | Location of `tesseract.exe` if it isn't on PATH |

### Installing Tesseract (for `--ocr`)

`pip install pillow pytesseract` installs only the Python side. You also need the Tesseract program itself:

| OS | Command |
|---|---|
| Windows | `winget install UB-Mannheim.TesseractOCR`, or the installer from <https://github.com/UB-Mannheim/tesseract/wiki>. Then open a **new** terminal. |
| macOS | `brew install tesseract` |
| Debian/Ubuntu | `sudo apt install tesseract-ocr` |

Check it with `tesseract --version`. On Windows, the script also looks in `C:\Program Files\Tesseract-OCR\`, so it works even if the installer didn't add Tesseract to PATH. For other languages, install the matching language data, e.g. `sudo apt install tesseract-ocr-fra`. On Windows, select the languages in the installer.

If something is missing, the script stops with a message saying exactly what to install.

## Saving problem pages (`--save-problem-pages`)

```
python pdf_to_text.py lecture.pdf --ocr --save-problem-pages
```

This saves a PNG (150 DPI; change it with `--image-dpi`) of every page that needed OCR, has a warning, or has too little usable text. The images go to `lecture_problem_pages/page_008.png`, `page_013.png`, and so on, next to the output file. Old `page_*.png` files in that folder are replaced on each run.

### When to send a page image to the LLM instead

Text alone is enough for most pages. Send the **page image** to a vision-capable LLM, along with or instead of the extracted text, when:
- The page is a **diagram, flowchart, chart or graph** (`vector drawing elements`, `fragmented text`). The labels are extracted, but the arrows, axes and relationships between them are not.
- The page has a **table** whose rows come out ambiguous, or merged or empty cells.
- **Formulas** got flattened (fractions, sums, matrices) or were OCR'd.
- The page has `LOW CONFIDENCE EXTRACTION` or `NO USABLE TEXT LAYER`, or is a screenshot of code or a handwritten slide.
- The extracted text doesn't make sense in your read-through.

A practical approach is to run your preprocessing prompt on `lecture_raw.txt`, then attach the few PNGs from "Pages needing review" with a note like *"Pages 13 and 17 are diagrams; use these images for them."*

## Other options

| Option | Purpose |
|---|---|
| `--pages 1-10` / `1,3,5` / `1-5,8,10-12` / `9-` | Extract only some pages. Page markers keep the real page numbers. |
| `--sort` | Order blocks by position instead of PDF order (see Reading order) |
| `--password PW` | Open an encrypted PDF |
| `--verbose`, `-v` | Per-page statistics (characters, ink, images, drawings, tables) and MuPDF warnings |

Exit codes: `0` success (check "Pages needing review"); `1` for an error: file missing, not a PDF, encrypted, corrupted, output not writable, or OCR not available.

---

# Part 2: Flashcards → Anki

`flashcards_to_anki.py` turns multiple-choice flashcards generated by an LLM into a file you can import into Anki. Before exporting, it checks every card and reports any problems.

- **TSV export (the default; recommended).** It uses only the Python standard library. The file is plain text, so you can open it and check it before importing.
- **`.apkg` export (optional).** This builds a ready-to-open Anki deck and needs `pip install genanki`.

The script never rewrites your questions, answers or distractors. The only change it can make is to the order of the options, and only when you pass `--shuffle-options`.

Requires Anki 2.1.55+ (for older Anki versions, see `--no-header`).

## Workflow

### 1. Extract and prepare the course material
Run `python pdf_to_text.py lecture.pdf` (see Part 1). Then run your **preprocessing prompt** on `lecture_raw.txt`, plus images of any problem pages, to get clean, structured course notes.

### 2. Generate the flashcards
Run your **flashcard-generation prompt** on that text. The output must be a Python list of dictionaries in exactly this shape (JSON with the same keys also works):

```python
[
    {
        'question': '<b>What is a foreign key?</b>',
        'answer': 'A field that references a primary key in another table',
        'options': [
            'A field that uniquely identifies every row',
            'A field that references a primary key in another table',
            'A field used only for sorting data',
            'A field that cannot contain duplicate values'
        ]
    }
]
```

Rules the output must follow:
- Every card has `question`, `answer` and `options`. An `explanation` key is also allowed and goes into the Explanation field.
- `options` holds exactly 4 different, non-empty strings.
- `answer` is character-for-character identical to one of the options.
- Questions and options may contain HTML (`<b>`, `<i>`, `<code>`, `<br>`, …). To show a literal `<`, for example `List<int>`, write `&lt;` (`List&lt;int>`).
- Tell the LLM to vary which position (A–D) holds the correct answer.

### 3. Save the output
Paste the LLM output into a text file, for example `database_chapter_3.txt`. Save it as UTF-8, which is the default in modern editors, including Windows Notepad.

The script handles the usual wrapping for you: surrounding blank lines, one ```` ```python ```` / ```` ``` ```` code fence, and a leading `flashcards = `. It will **not** guess its way past real problems. If the LLM added chatty text before or after the list, or the output was cut off, the script stops and tells you what to fix.

### 4. Run the converter

```
python flashcards_to_anki.py database_chapter_3.txt
```

This writes `database_chapter_3_anki.tsv` next to the input and prints a report:

```
INVALID CARDS (1) - not exported:
  Card 5 (question: "Answer not in options")
    - 'answer' does not match any of the 4 options (answer: 'E')

SUMMARY
  Total cards:         87
  Valid cards:         84
  Invalid cards:       2
  Duplicate cards:     1
  Duplicate questions: 0

Correct answer distribution:
  A:   22 ( 26.2%)
  B:   20 ( 23.8%)
  C:   21 ( 25.0%)
  D:   21 ( 25.0%)
```

Invalid and duplicate cards are listed with their card number and reason, then skipped. The valid cards are still exported. To fix a skipped card, edit the `.txt` file and run the script again.

### 5. Import into Anki
Set up the note type once (see below). After that, each new deck is: **File → Import → pick the `.tsv` → Import.**

---

## Command-line options

| Option | What it does |
|---|---|
| `--output FILE`, `-o` | Where to write the TSV. Default: `<input>_anki.tsv` |
| `--strict` | Export **nothing** if any card is invalid or duplicated (exit code 3) |
| `--shuffle-options` | Shuffle each card's four options. The correct answer stays correct, and its letter is recalculated |
| `--seed N` | Makes the shuffle reproducible. Without it, the script picks a seed and prints it |
| `--apkg` | Also build an `.apkg` deck in the same folder as the TSV |
| `--deck-name NAME` | Deck name. Use `::` for subdecks, e.g. `"Databases::Midterm 1"` |
| `--tags "a b"` | Space-separated tags added to every note, e.g. `"cs340 ch3"` |
| `--report FILE` | Save the validation report to a file |
| `--no-header` | Leave out the `#` header lines (only for Anki older than 2.1.55) |
| `--write-templates DIR` | Write the front/back templates and CSS into `DIR` |

Examples:

```
python flashcards_to_anki.py ml_week4.txt --strict --report ml_week4_report.txt
python flashcards_to_anki.py ml_week4.txt --shuffle-options --seed 42
python flashcards_to_anki.py ml_week4.txt --apkg --deck-name "Machine Learning::Midterm 1" --tags "ml midterm1"
```

### What gets checked

Errors (the card is skipped):
- The entry is not a dictionary.
- A required key is missing, or there is an unexpected key. A likely typo gets a suggestion, e.g. `option` → `options`.
- `question` or `answer` is not a string, or is empty. A value that is only HTML (e.g. `<b></b>`) or an obvious placeholder also counts as empty.
- `options` is not a list, or does not have exactly 4 items.
- An option is empty, not a string, or a duplicate. Options that differ only in case, spacing or HTML also count as duplicates.
- `answer` doesn't exactly match an option. If it matches only when case or spacing is ignored, the script says so but does not auto-correct it.
- The card is a duplicate: same question, answer and options (in any order) as an earlier card.

Warnings (the card is exported; review it):
- The same question appears again with a different answer or options. Anki matches notes by their first field, so on import it may update or skip one of them.
- The text contains something like `List<int>`, which Anki would hide as an unknown HTML tag.
- The options already start with `A.` / `B)`, so the letters would appear twice.
- The correct-answer positions are highly unbalanced. This uses a chi-square test and warns when p < 0.01.

---

## Setting up the Anki note type (one time)

You do this once. Every deck from every course can then reuse the same note type.

1. In Anki, go to **Tools → Manage Note Types → Add**.
2. Choose **Add: Basic** and name it exactly **`Multiple Choice (A-D)`**. The TSV header asks Anki for this name, so the import dialog selects it automatically.
3. Select the new note type and click **Fields…**. Rename or add fields until there are exactly these 8, **in this order**:

   | # | Field |
   |---|---|
   | 1 | `Question` |
   | 2 | `OptionA` |
   | 3 | `OptionB` |
   | 4 | `OptionC` |
   | 5 | `OptionD` |
   | 6 | `CorrectAnswer` |
   | 7 | `CorrectLetter` |
   | 8 | `Explanation` |

   (Rename `Front` → `Question` and `Back` → `OptionA`, then add the other six.)
4. Click **Cards…** and paste in:
   - **Front Template:** the contents of [`anki_templates/front_template.html`](anki_templates/front_template.html)
   - **Back Template:** the contents of [`anki_templates/back_template.html`](anki_templates/back_template.html)
   - **Styling:** the contents of [`anki_templates/styling.css`](anki_templates/styling.css)
5. Save.

**Front:** the question with options A–D. **Back:** the same question and options with the correct option highlighted in green, then `Correct Answer: B. …`, and the explanation if there is one. HTML in the fields is rendered, and no JavaScript is used. The back template puts `{{text:CorrectLetter}}` in a CSS class to do the highlighting.

<details>
<summary>Front template</summary>

```html
<div class="mcq">
  <div class="mcq-question">{{Question}}</div>
  <div class="mcq-options">
    <div class="mcq-option"><span class="mcq-letter">A.</span><span class="mcq-text">{{OptionA}}</span></div>
    <div class="mcq-option"><span class="mcq-letter">B.</span><span class="mcq-text">{{OptionB}}</span></div>
    <div class="mcq-option"><span class="mcq-letter">C.</span><span class="mcq-text">{{OptionC}}</span></div>
    <div class="mcq-option"><span class="mcq-letter">D.</span><span class="mcq-text">{{OptionD}}</span></div>
  </div>
</div>
```
</details>

<details>
<summary>Back template</summary>

```html
<div class="mcq mcq-back correct-{{text:CorrectLetter}}">
  <div class="mcq-question">{{Question}}</div>
  <div class="mcq-options">
    <div class="mcq-option opt-A"><span class="mcq-letter">A.</span><span class="mcq-text">{{OptionA}}</span></div>
    <div class="mcq-option opt-B"><span class="mcq-letter">B.</span><span class="mcq-text">{{OptionB}}</span></div>
    <div class="mcq-option opt-C"><span class="mcq-letter">C.</span><span class="mcq-text">{{OptionC}}</span></div>
    <div class="mcq-option opt-D"><span class="mcq-letter">D.</span><span class="mcq-text">{{OptionD}}</span></div>
  </div>

  <hr id="answer">

  <div class="mcq-answer">
    Correct Answer: <span class="mcq-answer-text">{{CorrectLetter}}. {{CorrectAnswer}}</span>
  </div>

  {{#Explanation}}
  <div class="mcq-explanation">
    <div class="mcq-explanation-label">Explanation</div>
    {{Explanation}}
  </div>
  {{/Explanation}}
</div>
```
</details>

The CSS is in [`anki_templates/styling.css`](anki_templates/styling.css). It covers light and dark mode (desktop, AnkiMobile and AnkiDroid) and has a phone-width layout.

---

## Importing the TSV

1. **File → Import…** and select the `*_anki.tsv` file.
2. The header lines at the top of the file (`#separator:tab`, `#html:true`, `#notetype:…`, `#columns:…`) pre-fill the dialog. Check that:
   - **Field separator:** Tab
   - **Allow HTML in fields:** on. **This must be on**, or you will see literal `<b>` tags.
   - **Note type:** `Multiple Choice (A-D)`
   - **Deck:** your target deck. It is pre-selected if you used `--deck-name`.
3. **Field mapping.** Each column maps to the field with the same name:

   | TSV column | Anki field |
   |---|---|
   | 1 Question | Question |
   | 2 OptionA | OptionA |
   | 3 OptionB | OptionB |
   | 4 OptionC | OptionC |
   | 5 OptionD | OptionD |
   | 6 CorrectAnswer | CorrectAnswer |
   | 7 CorrectLetter | CorrectLetter |
   | 8 Explanation | Explanation |

4. **Existing notes / duplicates:** *Update* is a good choice. If you fix a card and re-import, the existing note (matched by its Question) is updated instead of duplicated.
5. Click **Import**.

How the TSV is encoded:
- The file is UTF-8 without a BOM, with one note per line.
- Line breaks inside a field become `<br>`, which looks the same when rendered.
- A field that contains a tab or `"`, or starts with `#`, is wrapped in double quotes with inner quotes doubled (standard CSV quoting). Without the quotes, Anki would read a line starting with `#` as a comment.
- Nothing else in the text is changed, and HTML is passed through as-is.

---

## Optional: direct `.apkg` export

```
pip install genanki
python flashcards_to_anki.py database_chapter_3.txt --apkg --deck-name "Databases::Chapter 3"
```

This writes the TSV as usual, plus `database_chapter_3.apkg` in the same folder. Double-click the `.apkg`, or use File → Import in Anki. You don't need to set up the note type first because the package includes it, with the same templates and CSS.

- **Deck name:** `--deck-name`, or the input file name if you leave it out.
- **Deterministic IDs:**
  - The note type ID comes from the note type name, so every `.apkg` you build shares one note type.
  - The deck ID comes from the deck name.
  - Each note's ID comes from its question and set of options, so re-importing an updated file updates notes instead of duplicating them, even after `--shuffle-options`.
- **Name clash:** if you created `Multiple Choice (A-D)` by hand as well, Anki keeps both and renames one. Pick one approach, or delete the unused note type.
- **Without genanki:** you get a message saying to run `pip install genanki`. The TSV has still been written by then.

**Recommendation:** use TSV while you are still adjusting your prompts, because it is easy to inspect, fix and re-import. Switch to `--apkg` once your pipeline is stable, if you find it more convenient.

---

## Answer-position balance

LLMs often put the correct answer in the same slot (frequently B or C). The report shows the A/B/C/D distribution and warns you if it is very unlikely to have happened by chance. You have two options:

- Ask the LLM to spread correct answers evenly and regenerate, **or**
- Run with `--shuffle-options --seed 42`. The option order is randomized and `CorrectLetter` is recalculated, but the wording is untouched. The same seed always produces the same file.

## Exit codes (`flashcards_to_anki.py`)

| Code | Meaning |
|---|---|
| 0 | Exported. Any invalid cards were reported and skipped |
| 1 | Error: file missing or unreadable, parse error, nothing valid, can't write output, or genanki missing for `--apkg` |
| 3 | `--strict` found invalid or duplicate cards, so nothing was exported |

---

# Running the tests

```
python -m unittest -v
```

This runs both test suites. The `pdf_to_text.py` tests build their own PDFs with PyMuPDF: a normal lecture PDF, a mixed one (image-heavy slide, vector flowchart, ruled table, blank page, dark-theme slide) and a scanned image-only PDF. They also cover encrypted, corrupted and fake PDFs.

OCR behaviour is checked with a stand-in for Tesseract that counts how many pages get OCR'd, so those tests run anywhere. One extra test runs real Tesseract when it is installed.

Tests skip automatically when an optional package is missing: PyMuPDF for the PDF tests, Pillow for the OCR tests, genanki for the `.apkg` test.
#   M y   P r o j e c t  
 