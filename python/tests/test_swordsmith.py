import sys
import unittest

sys.path.append('../swordsmith')

import swordsmith as sw

GRID_5x = '../swordsmith/grid/5x.txt'
GRID_15x = '../swordsmith/grid/15xcommon.txt'
WORDLIST = '../swordsmith/wordlist/spreadthewordlist.dict'

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

class TestQuadStackTemplate(unittest.TestCase):
    def runTest(self):
        stack_rows = 4
        grid = sw.read_grid(GRID_15x)
        quad_grid = sw.apply_quadruple_stack(grid, stack_rows=stack_rows)
        crossword = sw.AmericanCrossword.from_grid(quad_grid)

        rows = len(quad_grid)
        start_row = rows // 2 - stack_rows // 2
        target_rows = set(range(start_row, start_row + stack_rows))

        across_slots = [
            slot for slot in crossword.slots
            if sw.AmericanCrossword.is_across_slot(slot) and slot[0][0] in target_rows
        ]

        self.assertEqual(len(target_rows), len({slot[0][0] for slot in across_slots}))
        self.assertTrue(all(len(slot) == crossword.cols for slot in across_slots))

        down_slots = [
            slot for slot in crossword.slots
            if sw.AmericanCrossword.is_down_slot(slot)
        ]

        self.assertTrue(all(len(slot) > 1 for slot in down_slots))

unittest.main()