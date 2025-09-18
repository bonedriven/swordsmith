import utils

import math
import argparse
import itertools
import time
import os

from abc import ABC, abstractmethod
from random import shuffle
from collections import defaultdict

EMPTY = '.'
BLOCK = ' '

class Crossword:
    def __init__(self):
        self.slots = set()                                          # set of slots in the puzzle
        self.squares = defaultdict(lambda: defaultdict(int))        # square => slots that contain it => index of square in slot
        self.crossings = defaultdict(lambda: defaultdict(tuple))    # slot => slots that cross it => tuple of squares where they cross
        self.words = {}                                             # slot => word in that slot
        self.wordset = set()                                        # set of filled words in puzzle
    
    def __str__(self):
        return '\n'.join(', '.join(str(square) for square in slot) + ': ' + self.words[slot] for slot in self.slots)
    
    def clear(self):
        """Resets the crossword by clearing all fields"""
        self.slots.clear()
        self.squares.clear()
        self.crossings.clear()
        self.words.clear()
        self.wordset.clear()
    
    def generate_crossings(self):
        for square in self.squares:
            for slot in self.squares[square]:
                for crossing_slot in self.squares[square]:
                    if slot != crossing_slot:
                        if self.crossings[slot][crossing_slot]:
                            self.crossings[slot][crossing_slot] = (*self.crossings[slot][crossing_slot], square)
                        else:
                            self.crossings[slot][crossing_slot] = (square,)

    def __put_letter_in_slot(self, letter, slot, i):
        """Sets letter at the given index of the given slot"""
        old_word = self.words[slot]
        if i >= len(slot):
            raise IndexError('Index greater than word length!')

        if old_word[i] == letter:
            # no change
            return
        
        new_word = old_word[0:i] + letter + old_word[i+1:]

        # update wordset
        if old_word in self.wordset:
            self.wordset.remove(old_word)
        if self.is_word_filled(new_word):
            self.wordset.add(new_word)
        
        # update words for just this slot, not crossing slots
        self.words[slot] = new_word
    
    def put_word(self, word, slot, wordlist_to_update=None):
        """Places word in the given slot, optionally adding it to the given wordlist"""
        if wordlist_to_update:
            wordlist_to_update.add_word(word)
        
        prev_word = self.words[slot]
        
        # place word in words map and wordset
        self.words[slot] = word
        if self.is_word_filled(prev_word) and prev_word in self.wordset:
            self.wordset.remove(prev_word)
        if self.is_word_filled(word):
            self.wordset.add(word)
        
        # update crossing words
        for crossing_slot in self.crossings[slot]:
            for square in self.crossings[slot][crossing_slot]:
                index = self.squares[square][slot]
                crossing_index = self.squares[square][crossing_slot]
                self.__put_letter_in_slot(word[index], crossing_slot, crossing_index)

    def is_dupe(self, word):
        """Returns whether or not a given word is already in the grid"""
        return word in self.wordset

    def is_filled(self):
        """Returns whether or not the whole crossword is filled"""
        return all(Crossword.is_word_filled(word) for word in self.words.values())
    
    def is_validly_filled(self, wordlist):
        """Returns whether the crossword is filled with words in the wordlist with no dupes"""
        if not self.is_filled():
            return False # some unfilled words
        if not all(word in wordlist.words for word in self.words.values()):
            return False # some invalid words
        if not len(self.wordset) == len(self.words.values()):
            return False # some dupes
        return True
    
    @staticmethod
    def is_word_filled(word):
        """Returns whether word is completely filled"""
        return EMPTY not in word


class AmericanCrossword(Crossword):
    def __init__(self, rows, cols, *, min_word_length: int = 3, require_rotational_symmetry: bool = True):
        super(AmericanCrossword, self).__init__()

        self.rows = rows
        self.cols = cols
        self.min_word_length = max(1, int(min_word_length))
        self.require_rotational_symmetry = bool(require_rotational_symmetry)
        self.grid = [[EMPTY for _ in range(cols)] for _ in range(rows)] # 2D array of squares

        self.__generate_slots_from_grid()

    def _rotational_partner(self, row, col):
        """Return the 180° rotationally symmetric coordinate for the given cell."""
        return self.rows - 1 - row, self.cols - 1 - col

    def _set_block_pair(self, row, col):
        """Place a block at the given coordinate and any required rotational partner."""
        partner_row, partner_col = self._rotational_partner(row, col)

        coords = {(row, col)}
        if self.require_rotational_symmetry:
            coords.add((partner_row, partner_col))

        # validate we’re not overwriting letters
        for target_row, target_col in coords:
            value = self.grid[target_row][target_col]
            if value not in (EMPTY, BLOCK):
                raise ValueError(
                    "Cannot place block at "
                    f"{(row, col)}; letter '{value}' already present at "
                    f"{(target_row, target_col)}."
                )

        # place blocks
        for target_row, target_col in coords:
            self.grid[target_row][target_col] = BLOCK

    @classmethod
    def from_grid(cls, grid, *, min_word_length: int = 3, all_checked: bool = True, require_rotational_symmetry: bool = True):
        """Generates AmericanCrossword from 2D array of characters"""
        grid = [row for row in grid if len(row) > 0]
        if not grid:
            raise ValueError('Grid must contain at least one non-empty row.')
        rows = len(grid)
        cols = len(grid[0])
        if any(len(row) != cols for row in grid):
            raise ValueError('All rows in the grid must have the same length')

        blocks = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == BLOCK:
                    blocks.add((r, c))

        if require_rotational_symmetry:
            for row, col in blocks:
                partner = (rows - 1 - row, cols - 1 - col)
                if partner not in blocks:
                    raise ValueError(
                        "Grid violates rotational symmetry: block at "
                        f"{(row, col)} lacks partner at {partner}."
                    )

        xw = cls(rows, cols, min_word_length=min_word_length, require_rotational_symmetry=require_rotational_symmetry)
        if blocks:
            xw.put_blocks(blocks)

        # copy letters, validating against symmetry conflicts
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != BLOCK and grid[r][c] != EMPTY:
                    if require_rotational_symmetry and xw.grid[r][c] == BLOCK:
                        partner = xw._rotational_partner(r, c)
                        raise ValueError(
                            "Grid violates rotational symmetry: letter "
                            f"'{grid[r][c]}' at {(r, c)} conflicts with block at "
                            f"{partner}."
                        )
                    xw.grid[r][c] = grid[r][c]

        if require_rotational_symmetry:
            for r in range(rows):
                for c in range(cols):
                    if xw.grid[r][c] != BLOCK and xw.grid[r][c] != EMPTY:
                        partner_row, partner_col = xw._rotational_partner(r, c)
                        if xw.grid[partner_row][partner_col] == BLOCK:
                            raise ValueError(
                                "Grid violates rotational symmetry after letter copy: "
                                f"letter '{xw.grid[r][c]}' at {(r, c)} conflicts with block at "
                                f"{(partner_row, partner_col)}."
                            )

        xw.__generate_slots_from_grid(all_checked)

        return xw
    
    @staticmethod
    def is_across_slot(slot):
        return len({row for row, col in slot}) == 1
    
    @staticmethod
    def is_down_slot(slot):
        return len({col for row, col in slot}) == 1
    
    def get_clue_numbers_and_words(self):
        """Returns across words and down words and their numbers a la newspaper crosswords"""
        square_index = 1

        across_slots = set()
        down_slots = set()

        across_words = {} # square index => slot
        down_words = {} # square index => slot

        for row in range(self.rows):
            for col in range(self.cols):
                increment_index = False
                for slot in self.squares[(row, col)]:
                    if self.is_across_slot(slot) and slot not in across_slots:
                        across_slots.add(slot)
                        across_words[square_index] = self.words[slot]
                        increment_index = True
                    if self.is_down_slot(slot) and slot not in down_slots:
                        down_slots.add(slot)
                        down_words[square_index] = self.words[slot]
                        increment_index = True
                if increment_index:
                    square_index += 1
        
        return across_words, down_words
    
    def __generate_grid_from_slots(self):
        for slot in self.slots:
            for i, square in enumerate(slot):
                row, col = square
                self.grid[row][col] = self.words[slot][i]
    
    def __str__(self):
        self.__generate_grid_from_slots()
        return '\n'.join(' '.join([letter for letter in row]) for row in self.grid)
    
    def put_block(self, row, col):
        """Places block in certain square"""
        previous_values = {}
        coords = {(row, col)}
        if self.require_rotational_symmetry:
            coords.add(self._rotational_partner(row, col))
        try:
            for r, c in coords:
                previous_values[(r, c)] = self.grid[r][c]
            self._set_block_pair(row, col)
            self.__generate_slots_from_grid()
        except Exception as error:
            # rollback and regenerate cleanly (or clear on failure-to-regenerate)
            for (r, c), val in previous_values.items():
                self.grid[r][c] = val
            try:
                self.__generate_slots_from_grid()
            except Exception:
                self.clear()
            raise error

    def put_blocks(self, coords):
        """Places list of blocks in specified squares"""
        previous_values = {}
        try:
            for row, col in coords:
                if (row, col) not in previous_values:
                    previous_values[(row, col)] = self.grid[row][col]
                self._set_block_pair(row, col)
            self.__generate_slots_from_grid()
        except Exception as error:
            for (r, c), val in previous_values.items():
                self.grid[r][c] = val
            try:
                self.__generate_slots_from_grid()
            except Exception:
                self.clear()
            raise error
    
    def add_slot(self, squares, word):
        slot = tuple(squares)
        self.slots.add(slot)

        for i, square in enumerate(squares):
            self.squares[square][slot] = i
        
        if Crossword.is_word_filled(word):
            self.wordset.add(word)
        
        self.words[slot] = word

    def __generate_slots_from_grid(self, all_checked: bool = False):
        # When all_checked=True, accept slots of length 1+ regardless of min_word_length;
        # useful when loading pre-checked grids. Otherwise enforce self.min_word_length.
        min_length = 1 if all_checked else self.min_word_length
        min_length = max(1, min_length)

        slots_to_add = []

        def finalize_run(squares, word):
            if not squares:
                return
            if len(squares) < min_length:
                coords = ', '.join(f'({row}, {col})' for row, col in squares)
                raise ValueError(
                    f'Slot of length {len(squares)} shorter than minimum {min_length} encountered at {coords}'
                )
            slots_to_add.append((squares[:], word))

        self.clear()

        # generate across words
        for r in range(self.rows):
            word = ''
            squares = []
            for c in range(self.cols):
                letter = self.grid[r][c]
                if letter != BLOCK:
                    word += letter
                    squares.append((r, c))
                else:
                    finalize_run(squares, word)
                    word = ''
                    squares = []
            finalize_run(squares, word)

        # generate down words
        for c in range(self.cols):
            word = ''
            squares = []
            for r in range(self.rows):
                letter = self.grid[r][c]
                if letter != BLOCK:
                    word += letter
                    squares.append((r, c))
                else:
                    finalize_run(squares, word)
                    word = ''
                    squares = []
            finalize_run(squares, word)

        for squares, word in slots_to_add:
            self.add_slot(squares, word)

        self.generate_crossings()


class Wordlist:
    """Collection of words to be used for filling a crossword"""
    def __init__(self, words):
        self.words = set(words)
        self.added_words = set()

        # mapping from wildcard patterns to lists of matching words, used for memoization
        self.pattern_matches = {}

        # mapping from length to index to letter to wordset
        # this stores an n-letter word n times, so might be memory intensive but we'll see
        self.indices = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

        # mapping from length to wordset
        self.lengths = defaultdict(set)

        self.__init_indices()
    
    def __init_indices(self):
        for word in self.words:
            self.__add_word_to_indices(word)
    
    def __add_word_to_indices(self, word):
        length = len(word)
        self.lengths[length].add(word)
        for i, letter in enumerate(word):
            self.indices[length][i][letter].add(word)
    
    def __remove_word_from_indices(self, word):
        length = len(word)
        self.lengths[length].remove(word)
        for i, letter in enumerate(word):
            self.indices[length][i][letter].remove(word)
    
    def add_word(self, word):
        if word not in self.words:
            self.words.add(word)
            self.added_words.add(word)
            self.__add_word_to_indices(word)
    
    def remove_word(self, word):
        if word in self.words:
            self.words.remove(word)
            self.__remove_word_from_indices(word)
        if word in self.added_words:
            self.added_words.remove(word)
    
    def get_matches(self, pattern):
        if pattern in self.pattern_matches:
            return self.pattern_matches[pattern]
        
        length = len(pattern)
        indices = [self.indices[length][i][letter] for i, letter in enumerate(pattern) if letter != EMPTY]
        if indices:
            matches = set.intersection(*indices)
        else:
            matches = self.lengths[length]

        return matches


class Filler(ABC):
    """Abstract base class containing useful methods for filling crosswords"""

    @abstractmethod
    def fill(self, crossword, wordlist, animate):
        """Fills the given crossword using some strategy"""
    
    @staticmethod
    def get_new_crossing_words(crossword, slot, word):
        """Returns list of new words that cross the given slot, given a word to theoretically put
