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
GRID_5X_OPEN_PLUS = SWORDSMITH_DIR / "grid" / "5xopenplus.txt"
GRID_15X = SWORDSMITH_DIR / "grid" / "15xcommon.txt"
GRID_15X_QUAD = SWORDSMITH_DIR / "grid" / "15xquadstack.txt"
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


class TestWildcardOpenPlus(unittest.TestCase):
    def runTest(self) -> None:
        grid = sw.read_grid(GRID_5X_OPEN_PLUS)
        plus_coords = {
            (row_index, col_index)
            for row_index, row in enumerate(grid)
            for col_index, value in enumerate(row)
            if value == "+"
        }

        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookBackjumpFiller(5)

        result, _ = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result)

        # Force the internal grid representation to reflect the filled slots.
        str(crossword)

        # All wildcard squares should be either blocks or filled letters.
        for row_index, col_index in plus_coords:
            value = crossword.grid[row_index][col_index]
            self.assertNotEqual(value, "+")
            self.assertTrue(
                value == sw.BLOCK or value.isalpha(),
                msg=f"Wildcard at {(row_index, col_index)} unresolved: {value!r}",
            )

        total_cells = crossword.rows * crossword.cols
        block_count = sum(
            1 for row in crossword.grid for value in row if value == sw.BLOCK
        )
        self.assertLessEqual(block_count * 100, total_cells * 20)

        rows, cols = crossword.rows, crossword.cols
        for row_index in range(rows):
            for col_index in range(cols):
                if crossword.grid[row_index][col_index] == sw.BLOCK:
                    mirror_row = rows - 1 - row_index
                    mirror_col = cols - 1 - col_index
                    self.assertEqual(
                        crossword.grid[mirror_row][mirror_col],
                        sw.BLOCK,
                        msg=(
                            "Black square symmetry violated at "
                            f"{(row_index, col_index)} vs {(mirror_row, mirror_col)}"
                        ),
                    )


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
