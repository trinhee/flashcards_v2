#!/usr/bin/env python3
"""
flashcards_to_anki.py - convert LLM-generated multiple-choice flashcards into
an Anki-importable TSV file (and, optionally, a ready-made .apkg deck).

Input: a text file holding a Python list of dictionaries (JSON also works):

    [
        {
            'question': '<b>What is a foreign key?</b>',
            'answer': 'A field that references a primary key in another table',
            'options': ['...', '...', '...', '...'],
            # 'explanation': '...'   (optional)
        },
    ]

Only the standard library is needed for TSV export. The optional --apkg mode
needs `pip install genanki`.

Run `python flashcards_to_anki.py --help` for usage.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import html
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LETTERS = ("A", "B", "C", "D")
REQUIRED_KEYS = ("question", "answer", "options")
OPTIONAL_KEYS = ("explanation",)
TSV_COLUMNS = (
    "Question",
    "OptionA",
    "OptionB",
    "OptionC",
    "OptionD",
    "CorrectAnswer",
    "CorrectLetter",
    "Explanation",
)
NOTE_TYPE_NAME = "Multiple Choice (A-D)"

# Visible text that almost certainly means the LLM left a slot unfilled.
PLACEHOLDER_TEXT = {"...", "…", "todo", "tbd", "n/a", "placeholder", "question", "answer", "?", "-"}

# Tags that are reasonable to find in flashcard HTML. Anything else that looks
# like a tag (e.g. "List<int>") is flagged, because Anki would swallow it.
KNOWN_HTML_TAGS = {
    "a", "abbr", "b", "big", "blockquote", "br", "center", "cite", "code", "dd", "del",
    "dfn", "div", "dl", "dt", "em", "font", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
    "i", "img", "ins", "kbd", "li", "mark", "ol", "p", "pre", "q", "s", "samp", "small",
    "span", "strike", "strong", "sub", "sup", "table", "tbody", "td", "tfoot", "th",
    "thead", "tr", "tt", "u", "ul", "var", "anki-mathjax",
}

# ---------------------------------------------------------------------------
# Anki templates (also written to disk by --write-templates)
# ---------------------------------------------------------------------------

FRONT_TEMPLATE = """\
<div class="mcq">
  <div class="mcq-question">{{Question}}</div>
  <div class="mcq-options">
    <div class="mcq-option"><span class="mcq-letter">A.</span><span class="mcq-text">{{OptionA}}</span></div>
    <div class="mcq-option"><span class="mcq-letter">B.</span><span class="mcq-text">{{OptionB}}</span></div>
    <div class="mcq-option"><span class="mcq-letter">C.</span><span class="mcq-text">{{OptionC}}</span></div>
    <div class="mcq-option"><span class="mcq-letter">D.</span><span class="mcq-text">{{OptionD}}</span></div>
  </div>
</div>
"""

BACK_TEMPLATE = """\
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
"""

CARD_CSS = """\
/* Multiple Choice (A-D) - works on Anki desktop, AnkiMobile and AnkiDroid. */

.card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 20px;
  line-height: 1.45;
  text-align: left;
  color: #1f2328;
  background-color: #ffffff;
}

.mcq {
  max-width: 760px;
  margin: 0 auto;
  padding: 12px 16px;
}

.mcq-question {
  font-size: 1.1em;
  margin-bottom: 18px;
}

.mcq-option {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin: 8px 0;
  padding: 10px 14px;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  background-color: #f6f8fa;
}

.mcq-letter {
  flex-shrink: 0;
  min-width: 1.5em;
  font-weight: 700;
}

.mcq-text {
  flex: 1;
  overflow-wrap: anywhere;
}

/* Back side: highlight the correct option (no JavaScript needed). */
.correct-A .opt-A,
.correct-B .opt-B,
.correct-C .opt-C,
.correct-D .opt-D {
  border: 2px solid #1a7f37;
  background-color: #dafbe1;
  font-weight: 600;
}

hr#answer {
  border: none;
  border-top: 1px solid #d0d7de;
  margin: 20px 0 14px;
}

.mcq-answer {
  padding: 12px 14px;
  border-left: 5px solid #1a7f37;
  border-radius: 4px;
  background-color: #f0fff4;
  font-weight: 700;
}

.mcq-explanation {
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 8px;
  background-color: #f6f8fa;
  font-size: 0.92em;
}

.mcq-explanation-label {
  margin-bottom: 4px;
  font-size: 0.8em;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #57606a;
}

.mcq code, .mcq pre {
  font-family: "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 0.9em;
}
.mcq code { padding: 1px 4px; border-radius: 4px; background-color: rgba(127, 127, 127, 0.15); }
.mcq pre { padding: 10px; border-radius: 6px; overflow-x: auto; background-color: rgba(127, 127, 127, 0.12); }
.mcq pre code { padding: 0; background: none; }
.mcq img { max-width: 100%; height: auto; }

/* Night mode: .nightMode (desktop / AnkiMobile) and .night_mode (AnkiDroid). */
.card.nightMode, .card.night_mode {
  color: #e6edf3;
  background-color: #1e1e1e;
}
.nightMode .mcq-option, .night_mode .mcq-option {
  border-color: #3d444d;
  background-color: #2a2d31;
}
.nightMode .correct-A .opt-A, .nightMode .correct-B .opt-B,
.nightMode .correct-C .opt-C, .nightMode .correct-D .opt-D,
.night_mode .correct-A .opt-A, .night_mode .correct-B .opt-B,
.night_mode .correct-C .opt-C, .night_mode .correct-D .opt-D {
  border-color: #3fb950;
  background-color: #1b3a26;
}
.nightMode hr#answer, .night_mode hr#answer { border-top-color: #3d444d; }
.nightMode .mcq-answer, .night_mode .mcq-answer {
  border-left-color: #3fb950;
  background-color: #1b3a26;
}
.nightMode .mcq-explanation, .night_mode .mcq-explanation { background-color: #2a2d31; }
.nightMode .mcq-explanation-label, .night_mode .mcq-explanation-label { color: #9da7b3; }

/* Phones */
@media (max-width: 480px) {
  .card { font-size: 18px; }
  .mcq { padding: 8px 4px; }
  .mcq-option { padding: 9px 11px; }
}
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FlashcardError(Exception):
    """A normal user-facing problem. Printed without a traceback."""


@dataclass
class CardResult:
    number: int  # 1-based position in the input list
    card: Any
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicate_of: int | None = None

    @property
    def valid(self) -> bool:
        return not self.errors and self.duplicate_of is None


@dataclass
class DeckValidation:
    results: list[CardResult]
    duplicate_question_groups: list[list[int]]

    @property
    def valid(self) -> list[CardResult]:
        return [r for r in self.results if r.valid]

    @property
    def invalid(self) -> list[CardResult]:
        return [r for r in self.results if r.errors]

    @property
    def duplicates(self) -> list[CardResult]:
        return [r for r in self.results if not r.errors and r.duplicate_of is not None]


@dataclass
class AnswerDistribution:
    counts: dict[str, int]
    total: int
    p_value: float | None
    warnings: list[str]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def visible_text(value: str) -> str:
    """Approximate what Anki would display: tags removed, entities decoded."""
    text = re.sub(r"<[^>]*>", "", value)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize(value: str) -> str:
    """Comparison key used for duplicate detection (never for export)."""
    return visible_text(value).casefold()


def preview(value: Any, limit: int = 70) -> str:
    text = visible_text(value) if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_card(card: Any) -> str:
    if isinstance(card, dict) and isinstance(card.get("question"), str):
        return f'question: "{preview(card["question"])}"'
    return f"content: {preview(card)}"


def suspicious_html(value: str) -> list[str]:
    """Return fragments like '<int>' that the browser would treat as unknown tags."""
    found = []
    for match in re.finditer(r"</?\s*([A-Za-z][A-Za-z0-9-]*)", value):
        tag = match.group(1).lower()
        wellformed = re.match(r"</?\s*[A-Za-z][A-Za-z0-9-]*(\s[^<>]*)?/?>", value[match.start():])
        if tag not in KNOWN_HTML_TAGS or not wellformed:
            end = value.find(">", match.start())
            snippet = value[match.start(): end + 1 if 0 <= end - match.start() < 30 else match.end()]
            found.append(snippet)
    return found


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def clean_llm_output(text: str) -> tuple[str, list[str]]:
    """Strip harmless wrapping (BOM, whitespace, one Markdown code fence,
    a leading `name =` assignment). Returns (cleaned_text, notes)."""
    notes: list[str] = []
    text = text.lstrip("﻿").strip()
    if not text:
        raise FlashcardError("The input file is empty.")

    fence = re.match(r"^```[ \t]*([\w+-]*)[ \t]*\r?\n", text)
    if fence:
        text = text[fence.end():].rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
        else:
            notes.append("Opening ``` fence has no closing fence - check that the LLM output was not cut off.")
        notes.append(f"Removed Markdown code fence (```{fence.group(1)}).")
    elif text.endswith("```"):
        text = text[:-3].rstrip()
        notes.append("Removed trailing Markdown code fence.")

    if "```" in text:
        raise FlashcardError(
            "The input contains more than one Markdown code block. "
            "Save only the flashcard list (one block) to the file."
        )

    assignment = re.match(r"^([A-Za-z_]\w*)\s*=\s*(?=[\[{])", text)
    if assignment:
        text = text[assignment.end():]
        notes.append(f"Ignored leading assignment '{assignment.group(1)} ='.")

    if not text.startswith(("[", "{")):
        first_line = text.splitlines()[0]
        raise FlashcardError(
            "Expected the flashcard list to start with '[', but the file starts with:\n"
            f"    {preview(first_line, 100)}\n"
            "Remove any explanatory text the LLM added before the list."
        )
    if not text.endswith(("]", "}")):
        last_line = text.splitlines()[-1]
        raise FlashcardError(
            "Expected the flashcard list to end with ']', but the file ends with:\n"
            f"    {preview(last_line, 100)}\n"
            "Either the LLM output was cut off (ask it to continue / regenerate) "
            "or there is extra text after the list that should be removed."
        )
    return text, notes


def _format_parse_error(text: str, py_err: Exception, js_err: json.JSONDecodeError) -> str:
    lines = text.splitlines()
    parts = ["Could not parse the flashcard list as a Python literal or as JSON."]

    lineno = getattr(py_err, "lineno", None)
    offset = getattr(py_err, "offset", None)
    msg = getattr(py_err, "msg", None) or str(py_err)
    if lineno and 1 <= lineno <= len(lines):
        parts.append(f"Python parser: {msg} (line {lineno})")
        bad_line = lines[lineno - 1]
        parts.append(f"    {bad_line.rstrip()[:160]}")
        if offset and offset <= 160:
            parts.append("    " + " " * max(offset - 1, 0) + "^")
    else:
        parts.append(f"Python parser: {msg}")
    parts.append(f"JSON parser:   {js_err.msg} (line {js_err.lineno}, column {js_err.colno})")

    hints = []
    if re.search(r"[‘’“”]", text):
        hints.append("The file contains curly quotes (‘ ’ “ ”). Python/JSON strings need straight quotes (' or \").")
    if lineno and 1 <= lineno <= len(lines) and re.search(r"'[^'\n]*\w'\w", lines[lineno - 1]):
        hints.append(
            "An apostrophe inside a single-quoted string (e.g. 'Codd's rule') ends the string early. "
            "Use double quotes for that string or escape it as \\'."
        )
    if isinstance(py_err, ValueError):
        hints.append("Only plain literals are allowed (strings, lists, dicts) - no variables or function calls.")
    if hints:
        parts.append("Hints:")
        parts.extend(f"  - {h}" for h in hints)
    return "\n".join(parts)


def parse_flashcards(text: str) -> tuple[Any, str]:
    """Parse cleaned text. Python literal first (safe: ast.literal_eval), then JSON."""
    try:
        return ast.literal_eval(text), "Python"
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as py_err:
        try:
            return json.loads(text), "JSON"
        except json.JSONDecodeError as js_err:
            raise FlashcardError(_format_parse_error(text, py_err, js_err)) from None


def load_flashcards(path: Path) -> tuple[list[Any], str, list[str]]:
    """Read and parse the input file. Returns (cards, source_format, notes)."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise FlashcardError(f"Input file not found: {path}") from None
    except IsADirectoryError:
        raise FlashcardError(f"Input path is a directory, not a file: {path}") from None
    except OSError as exc:
        raise FlashcardError(f"Could not read {path}: {exc.strerror or exc}") from None

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FlashcardError(
            f"{path} is not valid UTF-8 (bad byte at position {exc.start}). "
            "Re-save the file with UTF-8 encoding and try again."
        ) from None

    cleaned, notes = clean_llm_output(text)
    data, source_format = parse_flashcards(cleaned)

    if isinstance(data, dict):
        raise FlashcardError(
            "The top-level object is a single dictionary, not a list. "
            "Wrap the card(s) in [ ... ]."
        )
    if isinstance(data, tuple):
        raise FlashcardError(
            "The top-level object is a tuple, not a list. "
            "This usually means the list brackets [ ] are missing around the cards."
        )
    if not isinstance(data, list):
        raise FlashcardError(f"The top-level object must be a list of flashcards, got {type(data).__name__}.")
    if not data:
        raise FlashcardError("The flashcard list is empty.")
    return data, source_format, notes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_text_field(name: str, value: Any, errors: list[str]) -> bool:
    if not isinstance(value, str):
        errors.append(f"'{name}' must be a string, got {type(value).__name__}")
        return False
    if not value.strip():
        errors.append(f"'{name}' is empty")
        return False
    if not visible_text(value):
        errors.append(f"'{name}' contains only HTML/whitespace and would display as blank: {value!r}")
        return False
    return True


def validate_card(card: Any, number: int) -> CardResult:
    """Check one card. Never modifies it."""
    result = CardResult(number=number, card=card)
    errors, warnings = result.errors, result.warnings

    if not isinstance(card, dict):
        errors.append(f"expected a dictionary with keys {', '.join(REQUIRED_KEYS)}; got {type(card).__name__}")
        return result

    missing = [k for k in REQUIRED_KEYS if k not in card]
    for key in missing:
        errors.append(f"missing required key '{key}'")
    allowed = set(REQUIRED_KEYS) | set(OPTIONAL_KEYS)
    for key in card:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), list(allowed), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            errors.append(f"unexpected key {key!r}{hint}")
    if missing:
        return result

    question_ok = _check_text_field("question", card["question"], errors)
    answer_ok = _check_text_field("answer", card["answer"], errors)

    if question_ok and visible_text(card["question"]).casefold() in PLACEHOLDER_TEXT:
        errors.append(f"'question' looks like a placeholder: {card['question']!r}")
    if answer_ok and visible_text(card["answer"]).casefold() in PLACEHOLDER_TEXT:
        warnings.append(f"'answer' looks like a placeholder: {card['answer']!r}")

    if "explanation" in card and not isinstance(card["explanation"], str):
        errors.append(f"'explanation' must be a string, got {type(card['explanation']).__name__}")

    options = card["options"]
    options_ok = False
    if not isinstance(options, list):
        errors.append(f"'options' must be a list, got {type(options).__name__}")
    elif len(options) != 4:
        errors.append(f"'options' must contain exactly 4 items, found {len(options)}")
    else:
        options_ok = True
        for i, opt in enumerate(options):
            if not isinstance(opt, str):
                errors.append(f"option {LETTERS[i]} must be a string, got {type(opt).__name__}")
                options_ok = False
            elif not visible_text(opt):
                errors.append(f"option {LETTERS[i]} is empty")
                options_ok = False
        if options_ok:
            seen: dict[str, int] = {}
            for i, opt in enumerate(options):
                key = normalize(opt)
                if key in seen:
                    same = "identical to" if opt == options[seen[key]] else "effectively identical to (ignoring case/spacing/HTML)"
                    errors.append(f"option {LETTERS[i]} is {same} option {LETTERS[seen[key]]}: {preview(opt)!r}")
                    options_ok = False
                else:
                    seen[key] = i

    if options_ok and answer_ok:
        answer = card["answer"]
        if answer not in options:
            near = [LETTERS[i] for i, o in enumerate(options) if normalize(o) == normalize(answer)]
            if near:
                errors.append(
                    f"'answer' does not exactly match any option; it matches option {near[0]} only when "
                    "ignoring case/spacing/HTML. Edit the card so the text is identical."
                )
            else:
                errors.append(f"'answer' does not match any of the 4 options (answer: {preview(answer)!r})")

    # Non-fatal warnings: things that will import but may not look right.
    if not errors:
        fields = [("question", card["question"]), ("answer", card["answer"])]
        fields += [(f"option {LETTERS[i]}", o) for i, o in enumerate(options)]
        if card.get("explanation"):
            fields.append(("explanation", card["explanation"]))
        for name, value in fields:
            bad = suspicious_html(value)
            if bad:
                warnings.append(
                    f"{name} contains {', '.join(repr(b) for b in bad[:3])}, which Anki will treat as an HTML tag "
                    "and hide. If it is meant literally, write &lt; instead of <."
                )
        lettered = [LETTERS[i] for i, o in enumerate(options) if re.match(r"^\s*(\(?[A-Da-d]\)|[A-Da-d][.:)])\s+", o)]
        if len(lettered) >= 2:
            warnings.append(
                f"options {', '.join(lettered)} start with their own letter label (e.g. 'A. ...'); "
                "the Anki template already adds A-D labels."
            )
    return result


def validate_deck(cards: list[Any]) -> DeckValidation:
    """Validate every card, then detect duplicate cards and duplicate questions."""
    results = [validate_card(card, i) for i, card in enumerate(cards, start=1)]

    first_card: dict[tuple, int] = {}
    questions: dict[str, list[int]] = {}
    for r in results:
        if r.errors:
            continue
        c = r.card
        key = (normalize(c["question"]), normalize(c["answer"]), tuple(sorted(normalize(o) for o in c["options"])))
        if key in first_card:
            r.duplicate_of = first_card[key]
            continue
        first_card[key] = r.number
        questions.setdefault(normalize(c["question"]), []).append(r.number)

    groups = [nums for nums in questions.values() if len(nums) > 1]
    by_number = {r.number: r for r in results}
    for nums in groups:
        for n in nums[1:]:
            by_number[n].warnings.append(
                f"same question as card #{nums[0]} but different answer/options. Both are exported, but Anki "
                "matches notes on the first field (Question), so on import it may update or skip one of them."
            )
    return DeckValidation(results=results, duplicate_question_groups=groups)


# ---------------------------------------------------------------------------
# Answer-position analysis and shuffling
# ---------------------------------------------------------------------------

def correct_letter(card: dict) -> str:
    return LETTERS[card["options"].index(card["answer"])]


def _chi2_sf_df3(x: float) -> float:
    """Survival function of the chi-square distribution with 3 degrees of freedom."""
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


def analyze_answer_distribution(cards: list[dict]) -> AnswerDistribution:
    counts = {letter: 0 for letter in LETTERS}
    for card in cards:
        counts[correct_letter(card)] += 1
    total = len(cards)
    warnings: list[str] = []
    p_value = None

    if total >= 8:
        expected = total / 4
        chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
        p_value = _chi2_sf_df3(chi2)
        if p_value < 0.01:
            most = max(counts, key=counts.get)
            least = min(counts, key=counts.get)
            warnings.append(
                f"Correct-answer placement is highly unbalanced: {most} is correct {counts[most] / total:.0%} "
                f"of the time and {least} only {counts[least] / total:.0%} (expected ~25% each; "
                f"chance of this by luck p={p_value:.4f}). Learners may pick up the pattern. "
                "Consider --shuffle-options or asking the LLM to vary the correct position."
            )
    elif total > 0:
        warnings.append(f"Only {total} valid card(s): too few to judge whether answer placement is balanced.")
    return AnswerDistribution(counts=counts, total=total, p_value=p_value, warnings=warnings)


def shuffle_card_options(card: dict, rng: random.Random) -> dict:
    """Return a copy of the card with options shuffled. The answer text is unchanged,
    so the correct option is preserved (its letter is recomputed at export)."""
    new_card = dict(card)
    options = list(card["options"])
    rng.shuffle(options)
    new_card["options"] = options
    assert new_card["answer"] in options
    return new_card


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def card_to_row(card: dict) -> list[str]:
    options = card["options"]
    return [
        card["question"],
        *options,
        card["answer"],
        correct_letter(card),
        card.get("explanation", "") or "",
    ]


def tsv_field(value: str) -> str:
    """Encode one field for Anki's CSV/TSV importer.

    Fields are HTML, so line breaks become <br> (identical on screen, and keeps
    one note per line). Fields containing a tab or a double quote - or starting
    with '#', which Anki would read as a comment - are wrapped in double quotes,
    with inner quotes doubled (standard CSV quoting, which Anki understands).
    Nothing else about the text is changed.
    """
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    if "\t" in value or '"' in value or value.startswith("#"):
        value = '"' + value.replace('"', '""') + '"'
    return value


def _write_text_atomic(path: Path, content: str) -> None:
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
        raise FlashcardError(f"Could not write {path}: {exc.strerror or exc}") from None


def export_tsv(rows: list[list[str]], path: Path, deck_name: str | None = None,
               tags: list[str] | None = None, header: bool = True) -> None:
    """Write an Anki-importable, UTF-8, tab-separated file."""
    lines = []
    if header:
        # File headers understood by Anki 2.1.55+ (they pre-fill the import dialog).
        lines += ["#separator:tab", "#html:true", f"#notetype:{NOTE_TYPE_NAME}"]
        if deck_name:
            lines.append(f"#deck:{deck_name}")
        if tags:
            lines.append(f"#tags:{' '.join(tags)}")
        lines.append("#columns:" + "\t".join(TSV_COLUMNS))
    for row in rows:
        lines.append("\t".join(tsv_field(v) for v in row))
    _write_text_atomic(path, "\n".join(lines) + "\n")


def stable_id(text: str) -> int:
    """Deterministic ID in genanki's recommended range [2**30, 2**31)."""
    digest = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return (1 << 30) + digest % (1 << 30)


def export_apkg(rows: list[list[str]], path: Path, deck_name: str, tags: list[str] | None = None) -> None:
    try:
        import genanki  # type: ignore[import-not-found]
    except ImportError:
        raise FlashcardError(
            "--apkg needs the 'genanki' package, which is not installed.\n"
            "Install it with:\n    pip install genanki\n"
            "(The TSV file was still written and can be imported into Anki directly.)"
        ) from None

    model = genanki.Model(
        stable_id("model:" + NOTE_TYPE_NAME),  # same ID for every deck -> one shared note type
        NOTE_TYPE_NAME,
        fields=[{"name": name} for name in TSV_COLUMNS],
        templates=[{"name": "Multiple Choice", "qfmt": FRONT_TEMPLATE, "afmt": BACK_TEMPLATE}],
        css=CARD_CSS,
    )
    deck = genanki.Deck(stable_id("deck:" + deck_name), deck_name)
    for row in rows:
        fields = [f.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>") for f in row]
        # GUID from question + option set: re-importing an updated file updates the
        # existing note (even after --shuffle-options) instead of duplicating it.
        guid = genanki.guid_for(row[0], *sorted(row[1:5]))
        deck.add_note(genanki.Note(model=model, fields=fields, guid=guid, tags=tags or []))
    try:
        genanki.Package(deck).write_to_file(str(path))
    except OSError as exc:
        raise FlashcardError(f"Could not write {path}: {exc.strerror or exc}") from None


def write_templates(directory: Path) -> list[Path]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FlashcardError(f"Could not create {directory}: {exc.strerror or exc}") from None
    written = []
    for name, content in (("front_template.html", FRONT_TEMPLATE),
                          ("back_template.html", BACK_TEMPLATE),
                          ("styling.css", CARD_CSS)):
        target = directory / name
        _write_text_atomic(target, content)
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_distribution(dist: AnswerDistribution, title: str) -> list[str]:
    lines = [title]
    for letter in LETTERS:
        count = dist.counts[letter]
        pct = count / dist.total * 100 if dist.total else 0.0
        lines.append(f"  {letter}: {count:>4} ({pct:5.1f}%)")
    return lines


def generate_validation_report(
    input_path: Path,
    source_format: str,
    notes: list[str],
    deck: DeckValidation,
    original: AnswerDistribution,
    shuffled: AnswerDistribution | None = None,
    seed: int | None = None,
) -> str:
    lines: list[str] = [f"Flashcard validation report for {input_path} (parsed as {source_format})", ""]

    if notes:
        lines += [f"Note: {n}" for n in notes] + [""]

    if deck.invalid:
        lines.append(f"INVALID CARDS ({len(deck.invalid)}) - not exported:")
        for r in deck.invalid:
            lines.append(f"  Card {r.number} ({describe_card(r.card)})")
            lines += [f"    - {e}" for e in r.errors]
        lines.append("")

    if deck.duplicates:
        lines.append(f"DUPLICATE CARDS ({len(deck.duplicates)}) - not exported:")
        for r in deck.duplicates:
            lines.append(f"  Card {r.number} is a duplicate of card {r.duplicate_of} ({describe_card(r.card)})")
        lines.append("")

    if deck.duplicate_question_groups:
        lines.append(f"DUPLICATE QUESTIONS ({len(deck.duplicate_question_groups)} group(s)) - exported, please review:")
        for nums in deck.duplicate_question_groups:
            first = next(r for r in deck.results if r.number == nums[0])
            lines.append(f"  Cards {', '.join(map(str, nums))} ({describe_card(first.card)})")
        lines.append("")

    card_warnings = [r for r in deck.results if r.warnings and not r.errors and r.duplicate_of is None]
    deck_warnings = original.warnings + (shuffled.warnings if shuffled else [])
    if card_warnings or deck_warnings:
        lines.append("WARNINGS:")
        for r in card_warnings:
            lines.append(f"  Card {r.number} ({describe_card(r.card)})")
            lines += [f"    - {w}" for w in r.warnings]
        for w in original.warnings:
            lines.append(f"  - {w}" if not shuffled else f"  - Before shuffling: {w}")
        if shuffled:
            lines += [f"  - After shuffling: {w}" for w in shuffled.warnings]
        lines.append("")

    lines += [
        "SUMMARY",
        f"  Total cards:         {len(deck.results)}",
        f"  Valid cards:         {len(deck.valid)}",
        f"  Invalid cards:       {len(deck.invalid)}",
        f"  Duplicate cards:     {len(deck.duplicates)}",
        f"  Duplicate questions: {sum(len(g) - 1 for g in deck.duplicate_question_groups)}",
        "",
    ]
    if original.total:
        if shuffled:
            lines += format_distribution(original, "Correct answer distribution (as generated):")
            lines += [""] + format_distribution(shuffled, f"Correct answer distribution (after --shuffle-options, seed {seed}):")
        else:
            lines += format_distribution(original, "Correct answer distribution:")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_anki.tsv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flashcards_to_anki.py",
        description=(
            "Validate LLM-generated multiple-choice flashcards (a Python list of dicts, or JSON) "
            "and convert them to a TSV file for Anki import - optionally also an .apkg deck."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python flashcards_to_anki.py flashcards.txt
  python flashcards_to_anki.py db_ch3.txt --output out/db_ch3.tsv --report db_ch3_report.txt
  python flashcards_to_anki.py db_ch3.txt --strict
  python flashcards_to_anki.py db_ch3.txt --shuffle-options --seed 42
  python flashcards_to_anki.py db_ch3.txt --apkg --deck-name "Databases::Midterm 1"
  python flashcards_to_anki.py --write-templates anki_templates

exit codes:
  0  export succeeded (invalid cards, if any, were reported and skipped)
  1  error (unreadable/unparseable input, nothing valid, cannot write output, genanki missing)
  3  --strict was given and at least one card was invalid or duplicated; nothing exported
""",
    )
    parser.add_argument("input", nargs="?", help="text file containing the LLM output")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="TSV file to write (default: <input name>_anki.tsv next to the input)")
    parser.add_argument("--strict", action="store_true",
                        help="export nothing if any card is invalid or duplicated")
    parser.add_argument("--shuffle-options", action="store_true",
                        help="shuffle each card's options (the correct answer is preserved); off by default")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="random seed for --shuffle-options (default: random, printed so the run can be repeated)")
    parser.add_argument("--apkg", action="store_true",
                        help="also build an .apkg deck in the output directory (requires: pip install genanki)")
    parser.add_argument("--deck-name", metavar="NAME",
                        help="Anki deck name, e.g. \"Machine Learning::Midterm 1\" (default for --apkg: input file name). "
                             "Also pre-selects the deck in the TSV header.")
    parser.add_argument("--tags", metavar="TAGS",
                        help="space-separated Anki tags added to every note, e.g. \"cs340 chapter3\"")
    parser.add_argument("--report", metavar="FILE", help="also save the validation report to this file")
    parser.add_argument("--no-header", action="store_true",
                        help="omit the '#separator/#columns' header lines (only needed for Anki older than 2.1.55)")
    parser.add_argument("--write-templates", metavar="DIR",
                        help="write the recommended Anki front/back templates and CSS into DIR and exit")
    return parser


def configure_console() -> None:
    # Avoid UnicodeEncodeError when printing card text on legacy Windows consoles.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    if input_path.exists() and output_path.resolve() == input_path.resolve():
        raise FlashcardError("The output file would overwrite the input file. Choose a different --output.")
    if output_path.parent and not output_path.parent.exists():
        raise FlashcardError(f"Output directory does not exist: {output_path.parent}")
    tags = args.tags.split() if args.tags else []

    cards, source_format, notes = load_flashcards(input_path)
    print(f"Loaded {len(cards)} card(s) from {input_path}\n")

    deck = validate_deck(cards)
    export_cards = [r.card for r in deck.valid]
    original = analyze_answer_distribution(export_cards) if export_cards else AnswerDistribution(
        {l: 0 for l in LETTERS}, 0, None, [])

    shuffled = None
    seed = None
    if args.shuffle_options and export_cards:
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 1_000_000)
        rng = random.Random(seed)
        export_cards = [shuffle_card_options(c, rng) for c in export_cards]
        shuffled = analyze_answer_distribution(export_cards)

    report = generate_validation_report(input_path, source_format, notes, deck, original, shuffled, seed)
    print(report)
    if args.report:
        _write_text_atomic(Path(args.report), report)
        print(f"Validation report saved to {args.report}")

    problems = len(deck.invalid) + len(deck.duplicates)
    if args.strict and problems:
        print(f"Strict mode: {problems} card(s) invalid or duplicated - nothing was exported.", file=sys.stderr)
        return 3
    if not export_cards:
        raise FlashcardError("No valid cards to export.")

    rows = [card_to_row(c) for c in export_cards]
    export_tsv(rows, output_path, deck_name=args.deck_name, tags=tags, header=not args.no_header)
    print(f"Wrote {len(rows)} card(s) to {output_path}")
    if shuffled:
        print(f"Options were shuffled with seed {seed} (re-run with --seed {seed} to reproduce).")

    if args.apkg:
        deck_name = args.deck_name or input_path.stem.replace("_", " ")
        apkg_path = output_path.with_name(f"{input_path.stem}.apkg")
        export_apkg(rows, apkg_path, deck_name, tags)
        print(f"Wrote Anki package to {apkg_path} (deck: {deck_name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.seed is not None and not args.shuffle_options:
        parser.error("--seed only has an effect together with --shuffle-options")
    if args.tags and any('"' in t for t in args.tags.split()):
        parser.error("tags cannot contain double quotes")

    try:
        if args.write_templates:
            for path in write_templates(Path(args.write_templates)):
                print(f"Wrote {path}")
            if not args.input:
                return 0
        if not args.input:
            parser.error("the input file is required (e.g. python flashcards_to_anki.py flashcards.txt)")
        return run(args)
    except FlashcardError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
