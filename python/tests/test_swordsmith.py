import sys
import unittest
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
SWORDSMITH_DIR = (TEST_DIR / ".." / "swordsmith").resolve()
sys.path.append(str(SWORDSMITH_DIR))

import swordsmith as sw

GRID_5x = SWORDSMITH_DIR / "grid" / "5x.txt"
GRID_15x = SWORDSMITH_DIR / "grid" / "15xcommon.txt"
WORDLIST = SWORDSMITH_DIR / "wordlist" / "spreadthewordlist.dict"

class Test5xDFS(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_5x)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.DFSFiller()

        filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(crossword.is_filled())

class Test5xDFSBackjump(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_5x)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.DFSBackjumpFiller()

        filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(crossword.is_filled())

class Test5xMinlook(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_5x)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookFiller(5)

        filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(crossword.is_filled())

class Test5xMinlookBackjump(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_5x)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookBackjumpFiller(5)

        filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(crossword.is_filled())

class Test15xDFS(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_15x)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.DFSFiller()

        filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(crossword.is_filled())

class Test15xDFSBackjump(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_15x)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.DFSBackjumpFiller()

        filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(crossword.is_filled())

class Test15xMinlook(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_15x)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookFiller(5)

        filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(crossword.is_filled())

class Test15xMinlookBackjump(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_15x)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.read_wordlist(WORDLIST)
        filler = sw.MinlookBackjumpFiller(5)

        filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(crossword.is_filled())

if __name__ == "__main__":
    unittest.main()
