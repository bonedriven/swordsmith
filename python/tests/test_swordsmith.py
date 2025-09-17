import sys
import unittest

sys.path.append('../swordsmith')

import swordsmith as sw

GRID_5x = '../swordsmith/grid/5x.txt'
GRID_15x = '../swordsmith/grid/15xcommon.txt'
GRID_15x_QUAD = '../swordsmith/grid/15xquadstack.txt'
WORDLIST = '../swordsmith/wordlist/spreadthewordlist.dict'

QUADSTACK_WORDS = [
    'ABCDEFGHIJKLMNO',
    'BALOLULABONAOEN',
    'LRIVASURELIVRSC',
    'EDPENEEEXTTANTE',
    'ABLE',
    'BARD',
    'CLIP',
    'DOVE',
    'ELAN',
    'FUSE',
    'GLUE',
    'HARE',
    'IBEX',
    'JOLT',
    'KNIT',
    'LAVA',
    'MORN',
    'NEST',
    'ONCE',
]

QUADSTACK_ACROSS = QUADSTACK_WORDS[:4]

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


class TestQuadStackDFS(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_15x_QUAD)
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
    def runTest(self):
        grid = sw.read_grid(GRID_15x_QUAD)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.Wordlist(QUADSTACK_WORDS)
        filler = sw.DFSBackjumpFiller()

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result[0])
        self.assertTrue(crossword.is_filled())

        stack_slots = sw.Filler.find_quad_stack_slots(crossword)
        filled_stack_words = [crossword.words[slot] for slot in stack_slots]
        self.assertCountEqual(filled_stack_words, QUADSTACK_ACROSS)


class TestQuadStackMinlook(unittest.TestCase):
    def runTest(self):
        grid = sw.read_grid(GRID_15x_QUAD)
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
    def runTest(self):
        grid = sw.read_grid(GRID_15x_QUAD)
        crossword = sw.AmericanCrossword.from_grid(grid)
        wordlist = sw.Wordlist(QUADSTACK_WORDS)
        filler = sw.MinlookBackjumpFiller(5)

        result = filler.fill(crossword, wordlist, animate=False)
        self.assertTrue(result[0])
        self.assertTrue(crossword.is_filled())

        stack_slots = sw.Filler.find_quad_stack_slots(crossword)
        filled_stack_words = [crossword.words[slot] for slot in stack_slots]
        self.assertCountEqual(filled_stack_words, QUADSTACK_ACROSS)

unittest.main()
