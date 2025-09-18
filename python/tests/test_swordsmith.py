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
WORDLIST = SWORDSMITH_DIR / "wordlist" / "spreadthewordlist.dict"


class TestMinWordLengthEnforcement(unittest.TestCase):
    def test_two_letter_run_raises(self) -> None:
        grid = [
            ".. ",
            ".. ",
            "   ",
        ]

        with self.assertRaises(ValueError) as exc_info:
            sw.AmericanCrossword.from_grid(grid, min_word_length=3)

        self.assertIn("(0, 0)", str(exc_info.exception))

    def test_wildcard_respects_min_length(self) -> None:
        grid = [
            "+. ",
            "...",
            " .+",
        ]

        all_layouts = list(sw.iterate_wildcard_layouts(grid, min_word_length=1))
        filtered_layouts = list(sw.iterate_wildcard_layouts(grid, min_word_length=3))

        self.assertGreater(len(all_layouts), len(filtered_layouts))

        for layout in filtered_layouts:
            sw.AmericanCrossword.from_grid(layout, min_word_length=3)

    def test_valid_grid_constructs(self) -> None:
        grid = [
            "...",
            "...",
            "...",
        ]

        crossword = sw.AmericanCrossword.from_grid(grid, min_word_length=3)
        self.assertEqual(crossword.rows, 3)
        self.assertTrue(all(len(slot) >= 3 for slot in crossword.slots))


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

        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookBackjumpFiller(5)

        crossword = None
        result = False
        for candidate_grid in sw.iterate_wildcard_layouts(grid):
            candidate_crossword = sw.AmericanCrossword.from_grid(candidate_grid)
            result, _ = filler.fill(
                candidate_crossword, wordlist, animate=False
            )
            if result:
                crossword = candidate_crossword
                break

        self.assertTrue(result)
        self.assertIsNotNone(crossword)

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


if __name__ == "__main__":
    unittest.main()
