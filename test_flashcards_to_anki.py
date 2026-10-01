"""Tests for flashcards_to_anki.py. Run with:  python -m unittest -v"""

import csv
import io
import random
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import flashcards_to_anki as fta

HERE = Path(__file__).parent
SAMPLES = HERE / "samples"


def card(question="Q?", answer="B", options=None, **extra):
    return {"question": question, "answer": answer, "options": options or ["A", "B", "C", "D"], **extra}


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = fta.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def read_tsv(path):
    """Parse the TSV the way Anki does: skip '#' header lines, standard CSV quoting."""
    lines = [l for l in path.read_text(encoding="utf-8").split("\n") if l and not l.startswith("#")]
    return list(csv.reader(lines, delimiter="\t", quotechar='"'))


class CleaningAndParsing(unittest.TestCase):
    def test_code_fence_and_whitespace(self):
        text, notes = fta.clean_llm_output("\n\n```python\n[1]\n```\n\n")
        self.assertEqual(text, "[1]")
        self.assertTrue(notes)

    def test_leading_prose_is_rejected(self):
        with self.assertRaises(fta.FlashcardError):
            fta.clean_llm_output("Here are your cards:\n[1]")

    def test_truncated_output_is_rejected(self):
        with self.assertRaises(fta.FlashcardError):
            fta.clean_llm_output("[{'question': 'a',")

    def test_json_input(self):
        data, fmt = fta.parse_flashcards('[{"question": "Q", "answer": "A", "options": ["A","B","C","D"], "x": null}]')
        self.assertEqual(fmt, "JSON")
        self.assertEqual(data[0]["question"], "Q")

    def test_python_input_is_not_executed(self):
        with self.assertRaises(fta.FlashcardError):
            fta.parse_flashcards("[__import__('os').getcwd()]")

    def test_top_level_dict_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.txt"
            p.write_text(str(card()), encoding="utf-8")
            with self.assertRaises(fta.FlashcardError):
                fta.load_flashcards(p)


class Validation(unittest.TestCase):
    def errors(self, c):
        return fta.validate_card(c, 1).errors

    def test_valid(self):
        self.assertEqual(self.errors(card()), [])

    def test_failures(self):
        self.assertTrue(self.errors(card(options=["A", "B", "C"])))
        self.assertTrue(self.errors(card(options=["A", "B", "B", "D"])))
        self.assertTrue(self.errors(card(options=["A", "B", "b ", "D"])))
        self.assertTrue(self.errors(card(answer="E")))
        self.assertTrue(self.errors(card(answer="b")))  # no case-insensitive guessing
        self.assertTrue(self.errors(card(question="   ")))
        self.assertTrue(self.errors(card(question="<b> </b>")))
        self.assertTrue(self.errors(card(options=["A", "B", "", "D"])))
        self.assertTrue(self.errors(card(options=("A", "B", "C", "D"))))
        self.assertTrue(self.errors({"question": "Q", "answer": "A"}))
        self.assertTrue(self.errors(card(extra_key=1)))
        self.assertTrue(self.errors("not a dict"))

    def test_explanation_is_allowed(self):
        self.assertEqual(self.errors(card(explanation="Because.")), [])

    def test_duplicates(self):
        deck = fta.validate_deck([
            card(),
            card(options=["D", "C", "B", "A"]),       # same card, options reordered
            card(answer="C"),                         # same question, different answer
        ])
        self.assertEqual([r.number for r in deck.duplicates], [2])
        self.assertEqual(deck.duplicate_question_groups, [[1, 3]])
        self.assertEqual(len(deck.valid), 2)


class AnswerPosition(unittest.TestCase):
    def test_letters(self):
        for i, letter in enumerate("ABCD"):
            opts = ["w", "x", "y", "z"]
            self.assertEqual(fta.correct_letter(card(answer=opts[i], options=opts)), letter)

    def test_unbalanced_warning(self):
        cards = [card(answer="A", question=f"Q{i}") for i in range(40)]
        self.assertTrue(fta.analyze_answer_distribution(cards).warnings)

    def test_balanced_no_warning(self):
        cards = [card(answer="ABCD"[i % 4], question=f"Q{i}") for i in range(40)]
        self.assertEqual(fta.analyze_answer_distribution(cards).warnings, [])

    def test_shuffle_preserves_answer_and_is_reproducible(self):
        cards = [card(question=f"Q{i}", answer=f"opt{i % 4}", options=[f"opt{j}" for j in range(4)]) for i in range(50)]
        rng1, rng2 = random.Random(42), random.Random(42)
        s1 = [fta.shuffle_card_options(c, rng1) for c in cards]
        s2 = [fta.shuffle_card_options(c, rng2) for c in cards]
        self.assertEqual(s1, s2)
        for orig, new in zip(cards, s1):
            self.assertEqual(new["answer"], orig["answer"])
            self.assertEqual(sorted(new["options"]), sorted(orig["options"]))
            self.assertEqual(new["options"]["ABCD".index(fta.correct_letter(new))], orig["answer"])
            self.assertEqual(orig["options"], [f"opt{j}" for j in range(4)])  # input untouched
        self.assertNotEqual([c["options"] for c in s1], [c["options"] for c in cards])


class TsvExport(unittest.TestCase):
    def test_round_trip_of_awkward_text(self):
        tricky = card(
            question='<b>Bold</b>, "quoted", tab\there & <i>html</i>',
            answer="line1\nline2",
            options=["# starts with hash", "line1\nline2", "comma, here", "café – ü"],
            explanation="",
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "out.tsv"
            fta.export_tsv([fta.card_to_row(tricky)], p, deck_name="Deck::Sub", tags=["t1", "t2"])
            raw = p.read_text(encoding="utf-8")
            self.assertTrue(raw.startswith("#separator:tab\n#html:true\n"))
            self.assertIn("#deck:Deck::Sub", raw)
            rows = read_tsv(p)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(len(row), 8)
        self.assertEqual(row[0], tricky["question"])          # HTML, quotes, tab preserved
        self.assertEqual(row[1], "# starts with hash")
        self.assertEqual(row[2], "line1<br>line2")
        self.assertEqual(row[4], "café – ü")
        self.assertEqual(row[5], "line1<br>line2")
        self.assertEqual(row[6], "B")


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_sample(self):
        out = self.dir / "valid.tsv"
        code, stdout, _ = run_cli(SAMPLES / "sample_valid.txt", "-o", out)
        self.assertEqual(code, 0)
        rows = read_tsv(out)
        self.assertEqual([r[6] for r in rows], ["B", "C", "A", "D"])
        self.assertEqual(rows[0][0], "<b>What is a foreign key?</b>")
        self.assertIn("Valid cards:         4", stdout)

    def test_default_output_name(self):
        src = self.dir / "database_chapter_3.txt"
        src.write_text(str([card()]), encoding="utf-8")
        self.assertEqual(run_cli(src)[0], 0)
        self.assertTrue((self.dir / "database_chapter_3_anki.tsv").exists())

    def test_malformed_sample_non_strict_exports_valid_cards(self):
        out = self.dir / "mal.tsv"
        report = self.dir / "report.txt"
        code, stdout, _ = run_cli(SAMPLES / "sample_malformed.txt", "-o", out, "--report", report)
        self.assertEqual(code, 0)
        self.assertEqual(len(read_tsv(out)), 3)
        self.assertIn("Invalid cards:       8", stdout)
        self.assertIn("Card 5", report.read_text(encoding="utf-8"))

    def test_strict_exports_nothing(self):
        out = self.dir / "strict.tsv"
        code, _, err = run_cli(SAMPLES / "sample_malformed.txt", "-o", out, "--strict")
        self.assertEqual(code, 3)
        self.assertFalse(out.exists())
        self.assertIn("Strict mode", err)

    def test_strict_passes_clean_file(self):
        out = self.dir / "ok.tsv"
        self.assertEqual(run_cli(SAMPLES / "sample_valid.txt", "-o", out, "--strict")[0], 0)

    def test_shuffle_cli_is_reproducible(self):
        a, b = self.dir / "a.tsv", self.dir / "b.tsv"
        run_cli(SAMPLES / "sample_valid.txt", "-o", a, "--shuffle-options", "--seed", 7)
        run_cli(SAMPLES / "sample_valid.txt", "-o", b, "--shuffle-options", "--seed", 7)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        for row in read_tsv(a):
            self.assertEqual(row[1 + "ABCD".index(row[6])], row[5])

    def test_errors_are_friendly(self):
        code, _, err = run_cli(self.dir / "missing.txt")
        self.assertEqual(code, 1)
        self.assertIn("not found", err)
        self.assertNotIn("Traceback", err)
        code, _, err = run_cli(SAMPLES / "sample_broken_syntax.txt", "-o", self.dir / "x.tsv")
        self.assertEqual(code, 1)
        self.assertIn("apostrophe", err)

    def test_unwritable_output(self):
        code, _, err = run_cli(SAMPLES / "sample_valid.txt", "-o", self.dir / "no_such_dir" / "x.tsv")
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)

    def test_write_templates(self):
        code, _, _ = run_cli("--write-templates", self.dir / "tpl")
        self.assertEqual(code, 0)
        self.assertEqual((self.dir / "tpl" / "front_template.html").read_text(encoding="utf-8"), fta.FRONT_TEMPLATE)

    def test_shipped_templates_match_script(self):
        tpl = HERE / "anki_templates"
        self.assertEqual((tpl / "front_template.html").read_text(encoding="utf-8"), fta.FRONT_TEMPLATE)
        self.assertEqual((tpl / "back_template.html").read_text(encoding="utf-8"), fta.BACK_TEMPLATE)
        self.assertEqual((tpl / "styling.css").read_text(encoding="utf-8"), fta.CARD_CSS)


try:
    import genanki  # noqa: F401
    HAVE_GENANKI = True
except ImportError:
    HAVE_GENANKI = False


class ApkgExport(unittest.TestCase):
    @unittest.skipUnless(HAVE_GENANKI, "genanki not installed")
    def test_apkg(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "valid.tsv"
            code, _, err = run_cli(SAMPLES / "sample_valid.txt", "-o", out, "--apkg", "--deck-name", "DB::Midterm 1")
            self.assertEqual(code, 0, err)
            apkg = Path(d) / "sample_valid.apkg"
            with zipfile.ZipFile(apkg) as z:
                (Path(d) / "c.anki2").write_bytes(z.read("collection.anki2"))
            db = sqlite3.connect(Path(d) / "c.anki2")
            notes = [r[0].split("\x1f") for r in db.execute("select flds from notes order by id")]
            decks = db.execute("select decks from col").fetchone()[0]
            db.close()
        self.assertEqual(len(notes), 4)
        self.assertEqual(notes[0][0], "<b>What is a foreign key?</b>")
        self.assertEqual(sorted(n[6] for n in notes), ["A", "B", "C", "D"])
        self.assertIn("DB::Midterm 1", decks)

    def test_ids_are_deterministic(self):
        self.assertEqual(fta.stable_id("deck:X"), fta.stable_id("deck:X"))
        self.assertNotEqual(fta.stable_id("deck:X"), fta.stable_id("deck:Y"))
        self.assertTrue((1 << 30) <= fta.stable_id("anything") < (1 << 31))


if __name__ == "__main__":
    unittest.main()
