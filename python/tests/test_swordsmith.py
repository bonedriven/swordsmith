"""Test suite for the Swordsmith crossword engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
SWORDSMITH_DIR = (TEST_DIR / ".." / "swordsmith").resolve()
if str(SWORDSMITH_DIR) not in sys.path:
    sys.path.insert(0, str(SWORDSMITH_DIR))

import swordsmith as sw  # noqa: E402


GRID_5X = SWORDSMITH_DIR / "grid" / "5x.txt"
GRID_15X = SWORDSMITH_DIR / "grid" / "15xcommon.txt"
GRID_15X_QUAD = SWORDSMITH_DIR / "grid" / "15xquadstack.txt"
GRID_7X_OPEN_PLUS = SWORDSMITH_DIR / "grid" / "7xopenplus.txt"
WORDLIST = SWORDSMITH_DIR / "wordlist" / "spreadthewordlist.dict"


QUADSTACK_WORDS = [
    "ABCDEFGHIJKLMNO",
    "BALOLULABONAOEN",
    "LRIVASURELIVRSC",
    "EDPENEEEXTTANTE",
    "ABLE",
    "BARD",
    "CLIP",
    "DOVE",
    "ELAN",
    "FUSE",
    "GLUE",
    "HARE",
    "IBEX",
    "JOLT",
    "KNIT",
    "LAVA",
    "MORN",
    "NEST",
    "ONCE",
]

QUADSTACK_ACROSS = QUADSTACK_WORDS[:4]


class Test5xDFS(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_5X)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.DFSFiller()

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())


class Test5xDFSBackjump(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_5X)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.DFSBackjumpFiller()

        result, _ = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())


class Test5xMinlook(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_5X)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookFiller(5)

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())


class Test5xMinlookBackjump(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_5X)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookBackjumpFiller(5)

        result, _ = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())


class Test15xDFS(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_15X)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.DFSFiller()

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())


class Test15xDFSBackjump(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_15X)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.DFSBackjumpFiller()

        result, _ = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())


class Test15xMinlook(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_15X)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookFiller(5)

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())


class Test15xMinlookBackjump(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_15X)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookBackjumpFiller(5)

        result, _ = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())


class TestWildcardLayoutRegression(unittest.TestCase):
    def runTest(self) -> None:
        raw_grid = sw.read_grid(GRID_7X_OPEN_PLUS)
        layouts = list(sw.generate_wildcard_layouts(raw_grid))
        self.assertTrue(layouts)
        for layout_rows, _ in layouts:
            self.assertTrue(all("+" not in row for row in layout_rows))

        target_blocks = set(sw.WILDCARD_DEFAULT_BLOCKS)

        chosen_layout = None
        for layout_rows, assignments in layouts:
            assigned_blocks = {
                coord for coord, value in assignments.items() if value == sw.BLOCK
            }
            if assigned_blocks == target_blocks:
                chosen_layout = layout_rows
                break

        self.assertIsNotNone(chosen_layout, "Expected wildcard assignment missing")

        total_cells = len(chosen_layout) * len(chosen_layout[0])
        block_count = sum(cell == sw.BLOCK for row in chosen_layout for cell in row)
        self.assertLessEqual(block_count / total_cells, 0.2)

        wordlist = sw.read_wordlist(WORDLIST)
        for word in sw.WILDCARD_DEFAULT_WORDS:
            wordlist.add_word(word)

        prioritized_words = list(sw.WILDCARD_DEFAULT_WORDS)
        prioritized_set = set(prioritized_words)
        original_get_matches = wordlist.get_matches

        def prioritized_get_matches(self: sw.Wordlist, pattern: str) -> list[str]:
            matches = original_get_matches(pattern)
            if not matches:
                return matches
            priority = [word for word in prioritized_words if word in matches]
            if not priority:
                return matches
            remainder = [word for word in matches if word not in prioritized_set]
            return priority + remainder

        wordlist.get_matches = prioritized_get_matches.__get__(wordlist, sw.Wordlist)

        original_shuffle = sw.shuffle
        sw.shuffle = lambda seq: None

        try:
            crossword = sw.AmericanCrossword.from_grid(chosen_layout)
            filler = sw.DFSFiller()
            result = filler.fill(crossword, wordlist, animate=False)
        finally:
            wordlist.get_matches = original_get_matches
            sw.shuffle = original_shuffle

        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())

        filled_words = {
            crossword.words[slot]
            for slot in crossword.slots
            if sw.Crossword.is_word_filled(crossword.words[slot])
        }
        self.assertSetEqual(filled_words, set(prioritized_words))


class TestQuadStackDFS(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_15X_QUAD)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.Wordlist(QUADSTACK_WORDS)
        filler = sw.DFSFiller()

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())

        stack_slots = sw.Filler.find_quad_stack_slots(crossword)
        filled_stack_words = [crossword.words[slot] for slot in stack_slots]
        self.assertCountEqual(filled_stack_words, QUADSTACK_ACROSS)


class TestQuadStackDFSBackjump(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_15X_QUAD)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.Wordlist(QUADSTACK_WORDS)
        filler = sw.DFSBackjumpFiller()

        result, _ = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())

        stack_slots = sw.Filler.find_quad_stack_slots(crossword)
        filled_stack_words = [crossword.words[slot] for slot in stack_slots]
        self.assertCountEqual(filled_stack_words, QUADSTACK_ACROSS)


class TestQuadStackMinlook(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_15X_QUAD)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.Wordlist(QUADSTACK_WORDS)
        filler = sw.MinlookFiller(5)

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())

        stack_slots = sw.Filler.find_quad_stack_slots(crossword)
        filled_stack_words = [crossword.words[slot] for slot in stack_slots]
        self.assertCountEqual(filled_stack_words, QUADSTACK_ACROSS)


class TestQuadStackMinlookBackjump(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_15X_QUAD)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.Wordlist(QUADSTACK_WORDS)
        filler = sw.MinlookBackjumpFiller(5)

        result, _ = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())

        stack_slots = sw.Filler.find_quad_stack_slots(crossword)
        filled_stack_words = [crossword.words[slot] for slot in stack_slots]
        self.assertCountEqual(filled_stack_words, QUADSTACK_ACROSS)


class TestDLXMatrixEncoding(unittest.TestCase):
    def runTest(self) -> None:
        grid = ["...", "...", "..."]
        crossword = sw.AmericanCrossword.from_grid(grid)
        words = ["ACE", "GUT", "ERA", "AGE", "CUR", "ETA"]
        wordlist = sw.Wordlist(words)
        filler = sw.DLXFiller()

        matrix = filler._build_exact_cover(crossword, wordlist)

        slot_columns = {
            key.value for key in matrix.columns if key.kind == filler.SLOT_KIND
        }
        self.assertEqual(slot_columns, set(crossword.slots))

        across_slots = {
            slot[0][0]: slot
            for slot in crossword.slots
            if sw.AmericanCrossword.is_across_slot(slot)
        }
        slot_row0 = across_slots[0]
        row_data = matrix.row_data[(slot_row0, "ACE")]
        letter_values = {key.value for key in row_data.letter_columns}
        self.assertSetEqual(
            letter_values,
            {((0, 0), "A"), ((0, 1), "C"), ((0, 2), "E")},
        )

        letter_columns = {
            key.value for key in matrix.columns if key.kind == filler.LETTER_KIND
        }
        self.assertIn(((0, 0), "A"), letter_columns)
        self.assertIn(((1, 0), "G"), letter_columns)


class TestDLXSimpleFill(unittest.TestCase):
    def runTest(self) -> None:
        grid = ["...", "...", "..."]
        crossword = sw.AmericanCrossword.from_grid(grid)
        words = ["ACE", "GUT", "ERA", "AGE", "CUR", "ETA"]
        wordlist = sw.Wordlist(words)
        filler = sw.DLXFiller()

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)
        self.assertTrue(crossword.is_filled())

        across_words = {
            crossword.words[slot]
            for slot in crossword.slots
            if sw.AmericanCrossword.is_across_slot(slot)
        }
        down_words = {
            crossword.words[slot]
            for slot in crossword.slots
            if sw.AmericanCrossword.is_down_slot(slot)
        }

        self.assertSetEqual(across_words, {"ACE", "GUT", "ERA"})
        self.assertSetEqual(down_words, {"AGE", "CUR", "ETA"})


class TestDLXRejectsDuplicateSolution(unittest.TestCase):
    def runTest(self) -> None:
        grid = ["..", ".."]
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.Wordlist(["AA"])
        filler = sw.DLXFiller()

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertFalse(result)
        self.assertFalse(crossword.is_filled())


if __name__ == "__main__":
    unittest.main()
