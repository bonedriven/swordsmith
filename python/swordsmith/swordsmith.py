import utils

import math
import argparse
import time
import os

from abc import ABC, abstractmethod
from random import shuffle
from collections import defaultdict

EMPTY = '.'
BLOCK = ' '
WILDCARD = '+'  # flippable cell that can become BLOCK or EMPTY


# =============================
# Core data structures
# =============================
class Crossword:
    def __init__(self):
        self.slots = set()                                          # set of slots in the puzzle
        self.squares = defaultdict(lambda: defaultdict(int))        # square => slots that contain it => index of square in slot
        self.crossings = defaultdict(lambda: defaultdict(tuple))    # slot => slots that cross it => tuple of squares where they cross
        self.words = {}                                             # slot => word in that slot
        self.wordset = set()                                        # set of filled words in puzzle

    def __str__(self):
        return ''.join(', '.join(str(square) for square in slot) + ': ' + self.words[slot] for slot in self.slots)

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
            return False  # some unfilled words
        if not all(word in wordlist.words for word in self.words.values()):
            return False  # some invalid words
        if not len(self.wordset) == len(self.words.values()):
            return False  # some dupes
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
        self.grid = [[EMPTY for _ in range(cols)] for _ in range(rows)]  # 2D array of squares

        # Flexible (wildcard) cells: set of (r,c) that are allowed to flip to BLOCK
        self.wildcards = set()

        self.__generate_slots_from_grid(all_checked=True)

    # ---------- symmetry helpers ----------
    def _rotational_partner(self, row, col):
        """Return the 180° rotationally symmetric coordinate for the given cell."""
        return self.rows - 1 - row, self.cols - 1 - col

    def _set_block_pair(self, row, col):
        """Place a block at the given coordinate and any required rotational partner.
        This low-level method assumes the placement is allowed and does not check wildcards."""
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

    # ---------- construction ----------
    @classmethod
    def from_grid(cls, grid, *, min_word_length: int = 3, all_checked: bool = True, require_rotational_symmetry: bool = True):
        """Generates AmericanCrossword from 2D array of characters (no wildcards)."""
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

    @classmethod
    def from_grid_with_wildcards(cls, grid, *, wildcard: str = WILDCARD, min_word_length: int = 3, require_rotational_symmetry: bool = True):
        """Constructs a crossword where any `wildcard` cells are *flexible* and may be flipped to BLOCK by the solver.
        Wildcards are treated as EMPTY at load time; min-length checks are deferred until completion.
        """
        grid = [row for row in grid if len(row) > 0]
        if not grid:
            raise ValueError('Grid must contain at least one non-empty row.')
        rows = len(grid)
        cols = len(grid[0])
        if any(len(row) != cols for row in grid):
            raise ValueError('All rows in the grid must have the same length')

        # Validate fixed-block symmetry only (wildcards are flexible and can satisfy symmetry later)
        blocks = set()
        wilds = set()
        for r in range(rows):
            for c in range(cols):
                ch = grid[r][c]
                if ch == BLOCK:
                    blocks.add((r, c))
                elif ch == wildcard:
                    wilds.add((r, c))

        if require_rotational_symmetry:
            for (r, c) in blocks:
                pr, pc = (rows - 1 - r, cols - 1 - c)
                if (pr, pc) not in blocks:
                    raise ValueError(
                        "Grid violates rotational symmetry: fixed block at "
                        f"{(r, c)} lacks partner at {(pr, pc)}."
                    )

        xw = cls(rows, cols, min_word_length=min_word_length, require_rotational_symmetry=require_rotational_symmetry)
        xw.wildcards = wilds
        if blocks:
            xw.put_blocks(blocks)

        # copy letters (wildcards load as EMPTY)
        for r in range(rows):
            for c in range(cols):
                ch = grid[r][c]
                if ch not in (BLOCK, EMPTY, wildcard):
                    if require_rotational_symmetry and xw.grid[r][c] == BLOCK:
                        partner = xw._rotational_partner(r, c)
                        raise ValueError(
                            "Grid violates rotational symmetry: letter "
                            f"'{ch}' at {(r, c)} conflicts with block at {partner}."
                        )
                    xw.grid[r][c] = ch

        # Generate slots but *do not* enforce min length yet (wildcards might split later)
        xw.__generate_slots_from_grid(all_checked=True)
        return xw

    # ---------- slot helpers ----------
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

        across_words = {}  # square index => slot
        down_words = {}  # square index => slot

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
        return '
'.join(' '.join([letter for letter in row]) for row in self.grid)

    # ---------- block placement (fixed) ----------
    def put_block(self, row, col):
        """Places block in certain square (and its symmetric partner if required)."""
        previous_values = {}
        coords = {(row, col)}
        if self.require_rotational_symmetry:
            coords.add(self._rotational_partner(row, col))
        try:
            for r, c in coords:
                previous_values[(r, c)] = self.grid[r][c]
            self._set_block_pair(row, col)
            self.__generate_slots_from_grid(all_checked=True)
        except Exception as error:
            # rollback and regenerate cleanly (or clear on failure-to-regenerate)
            for (r, c), val in previous_values.items():
                self.grid[r][c] = val
            try:
                self.__generate_slots_from_grid(all_checked=True)
            except Exception:
                self.clear()
            raise error

    def put_blocks(self, coords):
        """Places list of blocks in specified squares."""
        previous_values = {}
        try:
            for row, col in coords:
                if (row, col) not in previous_values:
                    previous_values[(row, col)] = self.grid[row][col]
                self._set_block_pair(row, col)
            self.__generate_slots_from_grid(all_checked=True)
        except Exception as error:
            for (r, c), val in previous_values.items():
                self.grid[r][c] = val
            try:
                self.__generate_slots_from_grid(all_checked=True)
            except Exception:
                self.clear()
            raise error

    # ---------- wildcard flipping (integrated search) ----------
    def can_flip_wildcard(self, row, col):
        return (row, col) in self.wildcards and self.grid[row][col] != BLOCK

    def flip_wildcard(self, row, col):
        """Flip a wildcard to BLOCK, respecting rotational symmetry. Returns a reversible token."""
        if (row, col) not in self.wildcards:
            raise ValueError(f"Cell {(row, col)} is not a wildcard")
        changes = []
        partner = self._rotational_partner(row, col)
        coords = {(row, col)}
        if self.require_rotational_symmetry:
            coords.add(partner)
        for (r, c) in coords:
            if (r, c) not in self.wildcards and self.grid[r][c] != BLOCK:
                # trying to force a non-wildcard to BLOCK would violate input assumptions
                raise ValueError(f"Symmetry requires {(r, c)} to be BLOCK but it's not a wildcard")
        for (r, c) in coords:
            if self.grid[r][c] != BLOCK:
                changes.append((r, c, self.grid[r][c]))
                self.grid[r][c] = BLOCK
        # rebuild slots (min length deferred in integrated mode)
        self.__generate_slots_from_grid(all_checked=True)
        return tuple(changes)

    def undo_flip(self, changes):
        for (r, c, prev) in changes:
            self.grid[r][c] = prev
        self.__generate_slots_from_grid(all_checked=True)

    # ---------- validation helpers ----------
    def count_blocks(self):
        return sum(1 for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] == BLOCK)

    def has_short_run(self):
        m = max(1, int(self.min_word_length))
        if m <= 1:
            return False
        # rows
        for r in range(self.rows):
            run = 0
            for c in range(self.cols):
                v = self.grid[r][c]
                if v != BLOCK:
                    run += 1
                else:
                    if 0 < run < m:
                        return True
                    run = 0
            if 0 < run < m:
                return True
        # cols
        for c in range(self.cols):
            run = 0
            for r in range(self.rows):
                v = self.grid[r][c]
                if v != BLOCK:
                    run += 1
                else:
                    if 0 < run < m:
                        return True
                    run = 0
            if 0 < run < m:
                return True
        return False


# =============================
# Wordlist and matching
# =============================
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


# =============================
# Fillers (search strategies)
# =============================
class Filler(ABC):
    """Abstract base class containing useful methods for filling crosswords"""

    @abstractmethod
    def fill(self, crossword, wordlist, animate):
        """Fills the given crossword using some strategy"""

    @staticmethod
    def get_new_crossing_words(crossword, slot, word):
        """Returns list of new words that cross the given slot, given a word to theoretically put in the slot. Excludes slots that were already filled"""
        new_crossing_words = []

        for crossing_slot in crossword.crossings[slot]:
            new_crossing_word = crossword.words[crossing_slot]
            for square in crossword.crossings[slot][crossing_slot]:
                index = crossword.squares[square][slot]
                letter = word[index]

                crossing_index = crossword.squares[square][crossing_slot]
                crossing_word = crossword.words[crossing_slot]

                new_crossing_word = new_crossing_word[:crossing_index] + letter + new_crossing_word[crossing_index + 1:]

            if Crossword.is_word_filled(crossing_word) and crossing_word == new_crossing_word:
                # this word was already there, ignore
                continue

            new_crossing_words.append(new_crossing_word)

        return new_crossing_words

    @staticmethod
    def is_valid_match(crossword, wordlist, slot, match):
        """Returns whether the match can be placed in the slot without creating a dupe or invalid word."""

        if match not in wordlist.words:
            return False  # match is invalid word
        if crossword.is_dupe(match):
            return False  # match is dupe

        new_crossing_words = Filler.get_new_crossing_words(crossword, slot, match)

        # make sure crossing words are valid
        for crossing_word in new_crossing_words:
            if Crossword.is_word_filled(crossing_word) and crossing_word not in wordlist.words:
                return False  # created invalid word
            if crossword.is_dupe(crossing_word):
                return False  # created dupe

        # make sure crossing words don't dupe each other
        if len(set(new_crossing_words)) != len(new_crossing_words):
            return False

        return True

    @staticmethod
    def fewest_matches(crossword, wordlist):
        """Finds the slot that has the fewest possible matches, this is probably the best next place to look."""
        fewest_matches_slot = None
        fewest_matches = len(wordlist.words) + 1

        for slot in crossword.words:
            word = crossword.words[slot]
            if Crossword.is_word_filled(word):
                continue
            matches = len(wordlist.get_matches(word))
            if matches < fewest_matches:
                fewest_matches = matches
                fewest_matches_slot = slot
        return fewest_matches_slot, fewest_matches

    @staticmethod
    def minlook(crossword, wordlist, slot, matches, k):
        """Considers given matches, returns index of the one that offers the most possible crossing words. If there are none, returns -1"""
        match_indices = range(min(k, len(matches)))  # just take first k matches
        failed_indices = set()

        best_match_index = -1
        best_cross_product = -1

        for match_index in match_indices:
            cross_product = 0

            for crossing_word in Filler.get_new_crossing_words(crossword, slot, matches[match_index]):
                num_matches = len(wordlist.get_matches(crossing_word))

                # if no matches for some crossing slot, give up and move on
                # this is basically "arc-consistency lookahead"
                if num_matches == 0:
                    failed_indices.add(match_index)
                    cross_product = float('-inf')
                    break

                # use log product to avoid explosions
                cross_product += math.log(num_matches)

            if cross_product > best_cross_product:
                best_match_index = match_index
                best_cross_product = cross_product

        return best_match_index, failed_indices


class DFSFiller(Filler):
    """Naive DFS word filler."""

    def fill(self, crossword, wordlist, animate):
        if animate:
            utils.clear_terminal()
            print(crossword)

        if crossword.is_filled():
            return True

        slot, num_matches = Filler.fewest_matches(crossword, wordlist)
        if num_matches == 0:
            return False

        previous_word = crossword.words[slot]
        matches = list(wordlist.get_matches(crossword.words[slot]))
        shuffle(matches)

        for match in matches:
            if not Filler.is_valid_match(crossword, wordlist, slot, match):
                continue
            crossword.put_word(match, slot)
            if self.fill(crossword, wordlist, animate):
                return True
        crossword.put_word(previous_word, slot)
        return False


class DFSBackjumpFiller(Filler):
    """DFS with conflict-directed backjumping."""

    def fill(self, crossword, wordlist, animate):
        if animate:
            utils.clear_terminal()
            print(crossword)

        if crossword.is_filled():
            return True, None

        slot, num_matches = Filler.fewest_matches(crossword, wordlist)
        if num_matches == 0:
            return False, slot

        previous_word = crossword.words[slot]
        matches = list(wordlist.get_matches(crossword.words[slot]))
        shuffle(matches)

        for match in matches:
            if not Filler.is_valid_match(crossword, wordlist, slot, match):
                continue
            crossword.put_word(match, slot)
            is_filled, failed_slot = self.fill(crossword, wordlist, animate)
            if is_filled:
                return True, None
            if failed_slot not in crossword.crossings[slot]:
                crossword.put_word(previous_word, slot)
                return False, failed_slot
        crossword.put_word(previous_word, slot)
        return False, slot


class MinlookFiller(Filler):
    """DFS with minlook heuristic."""

    def __init__(self, k):
        self.k = k

    def fill(self, crossword, wordlist, animate):
        if animate:
            utils.clear_terminal()
            print(crossword)
        if crossword.is_filled():
            return True

        slot, num_matches = Filler.fewest_matches(crossword, wordlist)
        if num_matches == 0:
            return False

        previous_word = crossword.words[slot]
        matches = list(wordlist.get_matches(crossword.words[slot]))
        shuffle(matches)

        while matches:
            match_index, failed_indices = Filler.minlook(crossword, wordlist, slot, matches, self.k)
            if match_index != -1:
                match = matches[match_index]
            matches = [matches[i] for i in range(len(matches)) if i != match_index and i not in failed_indices]
            if match_index == -1:
                continue
            if not Filler.is_valid_match(crossword, wordlist, slot, match):
                continue
            crossword.put_word(match, slot)
            if self.fill(crossword, wordlist, animate):
                return True
        crossword.put_word(previous_word, slot)
        return False


class MinlookBackjumpFiller(Filler):
    """Minlook with backjumping."""

    def __init__(self, k):
        self.k = k

    def fill(self, crossword, wordlist, animate):
        if animate:
            utils.clear_terminal()
            print(crossword)
        if crossword.is_filled():
            return True, None

        slot, num_matches = Filler.fewest_matches(crossword, wordlist)
        if num_matches == 0:
            return False, slot

        previous_word = crossword.words[slot]
        matches = list(wordlist.get_matches(crossword.words[slot]))
        shuffle(matches)

        while matches:
            match_index, failed_indices = Filler.minlook(crossword, wordlist, slot, matches, self.k)
            if match_index != -1:
                match = matches[match_index]
            matches = [matches[i] for i in range(len(matches)) if i != match_index and i not in failed_indices]
            if match_index == -1:
                continue
            if not Filler.is_valid_match(crossword, wordlist, slot, match):
                continue
            crossword.put_word(match, slot)
            is_filled, failed_slot = self.fill(crossword, wordlist, animate)
            if is_filled:
                return True, None
            if failed_slot not in crossword.crossings[slot]:
                crossword.put_word(previous_word, slot)
                return False, failed_slot
        crossword.put_word(previous_word, slot)
        return False, slot


# =============================
# Hybrid / Integrated CSP filler
# =============================
class HybridWildcardFiller(Filler):
    """Integrated CSP-style search that interleaves word placement with wildcard (block) decisions.

    Strategy:
      1) Choose the unfilled slot with fewest matches.
      2) If it has matches, try them (word-first branch).
      3) If it has ZERO matches, try flipping one of the wildcard cells *inside that slot* to BLOCK
         (and its rotational partner if required), then re-run search.

    Success requires: grid is filled, no dupes, all entries valid, and no run shorter than min_word_length.
    Optional block ratio is enforced if provided.
    """

    def __init__(self, max_block_ratio: float | None = None):
        self.max_block_ratio = max_block_ratio

    def _within_block_budget(self, crossword: AmericanCrossword) -> bool:
        if self.max_block_ratio is None:
            return True
        max_blocks = int(self.max_block_ratio * crossword.rows * crossword.cols)
        return crossword.count_blocks() <= max_blocks

    def _valid_completion(self, crossword: AmericanCrossword, wordlist: Wordlist) -> bool:
        if not crossword.is_filled():
            return False
        if crossword.has_short_run():
            return False
        if not crossword.is_validly_filled(wordlist):
            return False
        return self._within_block_budget(crossword)

    def _wildcards_in_slot(self, crossword: AmericanCrossword, slot):
        return [(r, c) for (r, c) in slot if (r, c) in crossword.wildcards and crossword.grid[r][c] != BLOCK]

    def fill(self, crossword: AmericanCrossword, wordlist: Wordlist, animate: bool):
        if animate:
            utils.clear_terminal()
            print(crossword)

        # completion check (stronger than base fillers)
        if self._valid_completion(crossword, wordlist):
            return True

        # Choose slot with fewest matches
        slot, num_matches = Filler.fewest_matches(crossword, wordlist)

        if slot is None:
            # No unfilled slots but not valid completion (e.g., short runs exist)
            return False

        previous_word = crossword.words[slot]

        if num_matches > 0:
            # Word-first branch
            matches = list(wordlist.get_matches(crossword.words[slot]))
            shuffle(matches)
            for match in matches:
                if not Filler.is_valid_match(crossword, wordlist, slot, match):
                    continue
                crossword.put_word(match, slot)
                if self.fill(crossword, wordlist, animate):
                    return True
            # restore and fall through to wildcard decisions as a last resort
            crossword.put_word(previous_word, slot)

        # If zero matches (or all matches failed), consider flipping a wildcard inside this slot
        wcs = self._wildcards_in_slot(crossword, slot)
        if not wcs:
            return False

        # heuristic: try wildcards that split the slot more evenly first (center-out)
        # we approximate by distance from slot midpoint
        def dist_from_mid(rc):
            rs = [r for r, _ in slot]
            cs = [c for _, c in slot]
            if AmericanCrossword.is_across_slot(slot):
                mid = (min(cs) + max(cs)) / 2.0
                return abs(rc[1] - mid)
            else:
                mid = (min(rs) + max(rs)) / 2.0
                return abs(rc[0] - mid)

        wcs.sort(key=dist_from_mid)

        for (wr, wc) in wcs:
            try:
                token = crossword.flip_wildcard(wr, wc)
            except ValueError:
                continue
            if not self._within_block_budget(crossword):
                crossword.undo_flip(token)
                continue
            # After flip, the previously referenced `slot` object is stale; recurse fresh
            if self.fill(crossword, wordlist, animate):
                return True
            crossword.undo_flip(token)

        # restore last tried slot word (in case we changed it earlier)
        if crossword.words.get(slot) != previous_word:
            try:
                crossword.put_word(previous_word, slot)
            except Exception:
                pass
        return False


# =============================
# I/O helpers and orchestration
# =============================
WORDLIST_FOLDER = 'wordlist/'
GRID_FOLDER = 'grid/'
GRID_SUFFIX = '.txt'


def read_grid(filepath):
    with open(filepath, 'r') as f:
        return f.read().splitlines()


def read_wordlist(filepath, scored=True, min_score=50):
    with open(filepath, 'r') as f:
        words = f.readlines()

    words = [w.strip().upper() for w in words if w.strip()]

    if scored:
        temp = []
        for w in words:
            parts = w.split(';')
            if len(parts) == 1:
                temp.append(parts[0])
            else:
                try:
                    if int(parts[1]) >= min_score:
                        temp.append(parts[0])
                except ValueError:
                    # ignore malformed scores, keep the word
                    temp.append(parts[0])
        words = temp

    return Wordlist(words)


def log_times(times, strategy):
    print(f'Filled {len(times)} crosswords using {strategy}')
    print(f'Min time: {min(times):.4f} seconds')
    print(f'Avg time: {sum(times) / len(times):.4f} seconds')
    print(f'Max time: {max(times):.4f} seconds')


def get_filler(args):
    if args.strategy == 'dfs':
        return DFSFiller()
    elif args.strategy == 'dfsb':
        return DFSBackjumpFiller()
    elif args.strategy == 'minlook':
        return MinlookFiller(args.k)
    elif args.strategy == 'mlb':
        return MinlookBackjumpFiller(args.k)
    elif args.strategy == 'hybrid':
        return HybridWildcardFiller(max_block_ratio=args.max_block_ratio if args.max_block_ratio > 0 else None)
    else:
        return None


def run_test(args):
    dirname = os.path.dirname(__file__)
    wordlist_path_prefix = os.path.join(dirname, WORDLIST_FOLDER)
    grid_path_prefix = os.path.join(dirname, GRID_FOLDER)

    wordlist = read_wordlist(wordlist_path_prefix + args.wordlist_path)

    grid_path = grid_path_prefix + args.grid_path
    if not grid_path.endswith(GRID_SUFFIX):
        grid_path = grid_path + GRID_SUFFIX

    grid = read_grid(grid_path)

    times = []

    for _ in range(args.num_trials):
        tic = time.time()

        # Build crossword depending on mode
        if args.mode == 'preresolve':
            # Pre-resolve wildcards to concrete layouts, then fill
            wildcard_present = any(args.wildcard in row for row in grid)
            require_rotational_symmetry = getattr(args, 'require_rotational_symmetry', True)

            def iterate_wildcard_layouts(
                grid_lines,
                *,
                min_word_length: int = 3,
                require_rotational_symmetry: bool = True,
                wildcard: str = WILDCARD,
                max_block_ratio: float = 0.20,
            ):
                # Lightweight resolver using backtracking with early pruning
                grid_lines = [row for row in grid_lines if row]
                rows = len(grid_lines)
                if rows == 0:
                    yield []
                    return
                cols = len(grid_lines[0])
                base = [list(row) for row in grid_lines]

                def partner(r, c):
                    return rows - 1 - r, cols - 1 - c

                def is_block_symmetric(G):
                    if not require_rotational_symmetry:
                        return True
                    for r in range(rows):
                        for c in range(cols):
                            if G[r][c] == BLOCK:
                                pr, pc = partner(r, c)
                                if G[pr][pc] != BLOCK:
                                    return False
                    return True

                if require_rotational_symmetry:
                    for r in range(rows):
                        for c in range(cols):
                            if base[r][c] == BLOCK:
                                pr, pc = partner(r, c)
                                if base[pr][pc] != BLOCK:
                                    return

                wilds, seen = [], set()
                for r in range(rows):
                    for c in range(cols):
                        if base[r][c] == wildcard and (r, c) not in seen:
                            pr, pc = partner(r, c)
                            group = [(r, c)]
                            seen.add((r, c))
                            if (pr, pc) != (r, c) and base[pr][pc] == wildcard:
                                group.append((pr, pc))
                                seen.add((pr, pc))
                            wilds.append(tuple(sorted(group)))

                max_blocks = int(max_block_ratio * rows * cols)

                def short_run_exists(G):
                    m = max(1, int(min_word_length))
                    if m <= 1:
                        return False
                    for r in range(rows):
                        run = 0
                        for c in range(cols):
                            v = G[r][c]
                            if v != BLOCK:
                                run += 1
                            else:
                                if 0 < run < m:
                                    return True
                                run = 0
                        if 0 < run < m:
                            return True
                    for c in range(cols):
                        run = 0
                        for r in range(rows):
                            v = G[r][c]
                            if v != BLOCK:
                                run += 1
                            else:
                                if 0 < run < m:
                                    return True
                                run = 0
                        if 0 < run < m:
                            return True
                    return False

                fixed_blocks = sum(1 for r in range(rows) for c in range(cols) if base[r][c] == BLOCK)

                def group_weight(g):
                    # prioritize corners/edges
                    return -min(
                        abs(r - 0) + abs(c - 0),
                        abs(r - 0) + abs(c - (cols - 1)),
                        abs(r - (rows - 1)) + abs(c - 0),
                        abs(r - (rows - 1)) + abs(c - (cols - 1)),
                        for (r, c) in g
                    )

                wilds.sort(key=group_weight)
                G = [row[:] for row in base]

                def decide(i, blocks_used):
                    if i == len(wilds):
                        if not short_run_exists(G) and is_block_symmetric(G):
                            yield [''.join(row) for row in G]
                        return
                    group = wilds[i]
                    b_add = len(group)
                    # Try BLOCK
                    if fixed_blocks + blocks_used + b_add <= max_blocks:
                        ok = True
                        changed = []
                        for (r, c) in group:
                            if G[r][c] in (EMPTY, wildcard):
                                G[r][c] = BLOCK
                                changed.append((r, c))
                            elif G[r][c] != BLOCK:
                                ok = False
                                break
                            if require_rotational_symmetry:
                                pr, pc = partner(r, c)
                                if base[pr][pc] == wildcard:
                                    pass
                                elif G[pr][pc] != BLOCK:
                                    ok = False
                                    break
                        if ok and is_block_symmetric(G) and not short_run_exists(G):
                            yield from decide(i + 1, blocks_used + b_add)
                        for (r, c) in changed:
                            G[r][c] = EMPTY
                    # Try EMPTY
                    changed = []
                    ok = True
                    for (r, c) in group:
                        if G[r][c] in (BLOCK, wildcard):
                            G[r][c] = EMPTY
                            changed.append((r, c))
                        elif G[r][c] != EMPTY:
                            ok = False
                            break
                        if require_rotational_symmetry:
                            pr, pc = partner(r, c)
                            if base[pr][pc] == BLOCK:
                                ok = False
                                break
                    if ok and (not require_rotational_symmetry or is_block_symmetric(G)) and not short_run_exists(G):
                        yield from decide(i + 1, blocks_used)
                    for (r, c) in changed:
                        G[r][c] = EMPTY

                if not wilds:
                    if (not require_rotational_symmetry or is_block_symmetric(G)) and not short_run_exists(G):
                        yield [''.join(row) for row in G]
                    return
                yield from decide(0, 0)

            if wildcard_present:
                crossword = None
                for candidate_grid in iterate_wildcard_layouts(
                    grid,
                    min_word_length=args.min_word_length,
                    require_rotational_symmetry=args.require_rotational_symmetry,
                    wildcard=args.wildcard,
                    max_block_ratio=args.max_block_ratio,
                ):
                    try:
                        candidate_crossword = AmericanCrossword.from_grid(
                            candidate_grid,
                            min_word_length=args.min_word_length,
                            all_checked=True,
                            require_rotational_symmetry=args.require_rotational_symmetry,
                        )
                    except ValueError:
                        continue
                    filler = get_filler(args)
                    result = filler.fill(candidate_crossword, wordlist, args.animate)
                    is_filled = result[0] if isinstance(result, tuple) else result
                    if is_filled:
                        crossword = candidate_crossword
                        break
                if crossword is None:
                    raise RuntimeError('Unable to fill crossword for any wildcard configuration.')
            else:
                crossword = AmericanCrossword.from_grid(
                    grid,
                    min_word_length=args.min_word_length,
                    all_checked=True,
                    require_rotational_symmetry=args.require_rotational_symmetry,
                )
                filler = get_filler(args)
                filler.fill(crossword, wordlist, args.animate)
        else:  # args.mode == 'hybrid'
            # Load with flexible wildcards and use the integrated solver
            crossword = AmericanCrossword.from_grid_with_wildcards(
                grid,
                wildcard=args.wildcard,
                min_word_length=args.min_word_length,
                require_rotational_symmetry=args.require_rotational_symmetry,
            )
            filler = get_filler(args)
            if filler is None:
                raise ValueError('Invalid strategy for hybrid mode')
            filler.fill(crossword, wordlist, args.animate)

        duration = time.time() - tic
        times.append(duration)

        if not args.animate:
            print(crossword)

        print(f'
Filled {crossword.cols}x{crossword.rows} crossword in {duration:.4f} seconds
')

    log_times(times, args.strategy)


# =============================
# CLI
# =============================

def main():
    parser = argparse.ArgumentParser(description='Crossword filler with flexible blocks (wildcards) — integrated CSP and pre-resolve modes')

    parser.add_argument('-w', '--wordlist', dest='wordlist_path', type=str,
                        default='spreadthewordlist.dict', help='filepath for wordlist')
    parser.add_argument('-g', '--grid', dest='grid_path', type=str,
                        default='15xcommon.txt', help='filepath for grid (in grid/)')
    parser.add_argument('-t', '--num_trials', dest='num_trials', type=int,
                        default=5, help='number of grids to try filling')
    parser.add_argument('-a', '--animate',
                        default=False, action='store_true', help='whether to animate grid filling')

    # Strategies (word assignment)
    parser.add_argument('-s', '--strategy', dest='strategy', type=str,
                        default='hybrid', help='dfs, dfsb, minlook, mlb, hybrid (integrated wildcard)')
    parser.add_argument('-k', '--k', dest='k', type=int,
                        default=5, help='k constant for minlook')

    # Grid constraints
    parser.add_argument('--min-word-length', dest='min_word_length', type=int,
                        default=3, help='minimum allowed slot length in the final grid')

    parser.set_defaults(require_rotational_symmetry=True)
    parser.add_argument('--require-rotational-symmetry', dest='require_rotational_symmetry', action='store_true',
                        help='Enforce 180-degree rotational symmetry for blocks (default).')
    parser.add_argument('--no-rotational-symmetry', dest='require_rotational_symmetry', action='store_false',
                        help='Allow asymmetric block placement when loading/resolving grids.')

    parser.add_argument('--wildcard', dest='wildcard', type=str, default=WILDCARD,
                        help='Character in the grid that can flip to block or empty (default "+")')
    parser.add_argument('--max-block-ratio', dest='max_block_ratio', type=float, default=0.20,
                        help='Maximum fraction of grid squares allowed to be blocks (budget). Use 0 or negative to disable.')

    # Mode: pre-resolve wildcards first or integrate wildcard choices into the solver
    parser.add_argument('--mode', dest='mode', type=str, choices=['preresolve', 'hybrid'], default='hybrid',
                        help='Wildcard handling mode: preresolve (expand layouts first) or hybrid (integrated CSP).')

    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
