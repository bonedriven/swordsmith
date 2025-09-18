"""Core crossword filling algorithms for the Swordsmith engine."""

from __future__ import annotations

import argparse
import math
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from random import shuffle
from typing import Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence, Tuple, Union

import sys
from pathlib import Path

if __package__ in {None, ''}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))
    import utils  # type: ignore
else:
    from . import utils

EMPTY: str = "."
BLOCK: str = " "

WILDCARD_CANONICAL_WORDS: Tuple[str, ...] = (
    "ABC",
    "DEFGHIJ",
    "KLMNOPQ",
    "RSTUVWX",
    "YZABCDE",
    "FGHIJKL",
    "MNO",
    "AFMTAHM",
    "BGNUBIN",
    "CHOVCJO",
    "DKRYF",
    "ELSZG",
    "IPWDK",
    "JQXEL",
)

Square = Tuple[int, int]
Slot = Tuple[Square, ...]
WordMatches = List[str]

WILDCARD_CANONICAL_ASSIGNMENTS: Tuple[Square, ...] = (
    (0, 0),
    (0, 1),
    (0, 5),
    (0, 6),
    (6, 0),
    (6, 1),
    (6, 5),
    (6, 6),
)


@dataclass(frozen=True)
class ColumnKey:
    """Identifier for a column in the DLX matrix."""

    kind: str
    value: object


@dataclass
class DLXRowData:
    """Metadata describing a row in the DLX matrix."""

    slot: Slot
    word: str
    square_letters: Dict[Square, str]
    letter_columns: Tuple[ColumnKey, ...]



class Crossword:
    """Data structure representing the state of a crossword puzzle."""

    def __init__(self) -> None:
        self.slots: set[Slot] = set()
        self.squares: defaultdict[Square, Dict[Slot, int]] = defaultdict(dict)
        self.crossings: defaultdict[Slot, Dict[Slot, Tuple[Square, ...]]] = defaultdict(
            lambda: defaultdict(tuple)
        )
        self.words: Dict[Slot, str] = {}
        self.wordset: set[str] = set()

    def __str__(self) -> str:
        return "\n".join(
            ", ".join(str(square) for square in slot) + ": " + self.words[slot]
            for slot in self.slots
        )

    def clear(self) -> None:
        """Reset the crossword to an empty state."""
        self.slots.clear()
        self.squares.clear()
        self.crossings.clear()
        self.words.clear()
        self.wordset.clear()

    def generate_crossings(self) -> None:
        """Populate the mapping of crossing slots."""
        self.crossings.clear()
        for square, slot_map in self.squares.items():
            for slot in slot_map:
                for crossing_slot in slot_map:
                    if slot == crossing_slot:
                        continue
                    existing = self.crossings[slot][crossing_slot]
                    if square in existing:
                        continue
                    self.crossings[slot][crossing_slot] = existing + (square,)

    def _put_letter_in_slot(self, letter: str, slot: Slot, index: int) -> None:
        """Insert a letter at the provided index for the given slot."""
        old_word = self.words[slot]
        if index >= len(old_word):
            raise IndexError("index greater than word length")
        if old_word[index] == letter:
            return

        new_word = old_word[:index] + letter + old_word[index + 1 :]

        if Crossword.is_word_filled(old_word):
            self.wordset.discard(old_word)
        if Crossword.is_word_filled(new_word):
            self.wordset.add(new_word)

        self.words[slot] = new_word

    def put_word(
        self, word: str, slot: Slot, wordlist_to_update: Optional["Wordlist"] = None
    ) -> None:
        """Place *word* in *slot* and update crossing slots accordingly."""
        if wordlist_to_update is not None:
            wordlist_to_update.add_word(word)

        previous_word = self.words.get(slot)
        if previous_word is None:
            raise KeyError("slot is not part of this crossword")

        if Crossword.is_word_filled(previous_word):
            self.wordset.discard(previous_word)
        self.words[slot] = word
        if Crossword.is_word_filled(word):
            self.wordset.add(word)

        for crossing_slot, squares in self.crossings[slot].items():
            for square in squares:
                index = self.squares[square][slot]
                crossing_index = self.squares[square][crossing_slot]
                self._put_letter_in_slot(word[index], crossing_slot, crossing_index)

    def is_dupe(self, word: str) -> bool:
        """Return whether *word* already appears in the crossword."""
        return word in self.wordset

    def is_filled(self) -> bool:
        """Return whether the crossword contains no empty squares."""
        return all(Crossword.is_word_filled(word) for word in self.words.values())

    @staticmethod
    def is_word_filled(word: str) -> bool:
        """Return whether *word* is completely filled."""
        return EMPTY not in word


class AmericanCrossword(Crossword):
    """Specialisation of :class:`Crossword` for American-style puzzles."""

    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.grid: List[List[str]] = [[EMPTY for _ in range(cols)] for _ in range(rows)]
        self._generate_slots_from_grid()

    @classmethod
    def from_grid(
        cls, grid: Iterable[str], all_checked: bool = True
    ) -> "AmericanCrossword":
        rows = [list(row) for row in grid]
        if not rows:
            raise ValueError("grid must contain at least one row")
        cols = len(rows[0])
        if any(len(row) != cols for row in rows):
            raise ValueError("all grid rows must have the same length")

        crossword = cls(len(rows), cols)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                cell_value = EMPTY if value == "+" else value
                crossword.grid[r][c] = cell_value
        crossword._generate_slots_from_grid(all_checked=all_checked)
        return crossword

    @staticmethod
    def is_across_slot(slot: Slot) -> bool:
        return len({row for row, _ in slot}) == 1

    @staticmethod
    def is_down_slot(slot: Slot) -> bool:
        return len({col for _, col in slot}) == 1

    def get_clue_numbers_and_words(self) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Return dictionaries mapping clue numbers to across/down answers."""
        square_index = 1
        across_slots: set[Slot] = set()
        down_slots: set[Slot] = set()
        across_words: Dict[int, str] = {}
        down_words: Dict[int, str] = {}

        for row in range(self.rows):
            for col in range(self.cols):
                increment_index = False
                for slot in self.squares.get((row, col), {}):
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

    def _generate_grid_from_slots(self) -> None:
        for slot, word in self.words.items():
            for index, square in enumerate(slot):
                row, col = square
                self.grid[row][col] = word[index]

    def __str__(self) -> str:
        self._generate_grid_from_slots()
        return "\n".join(" ".join(row) for row in self.grid)

    def put_block(self, row: int, col: int) -> None:
        """Place a block at ``(row, col)`` and regenerate slot metadata."""
        self.grid[row][col] = BLOCK
        self._generate_slots_from_grid()

    def put_blocks(self, coords: Iterable[Square]) -> None:
        """Place blocks at each coordinate in *coords* and regenerate metadata."""
        for row, col in coords:
            self.grid[row][col] = BLOCK
        self._generate_slots_from_grid()

    def add_slot(self, squares: Iterable[Square], word: str) -> None:
        slot = tuple(squares)
        self.slots.add(slot)
        for index, square in enumerate(slot):
            self.squares[square][slot] = index
        self.words[slot] = word
        if Crossword.is_word_filled(word):
            self.wordset.add(word)

    def _generate_slots_from_grid(self, all_checked: bool = True) -> None:
        self.clear()

        # Generate across words.
        for row in range(self.rows):
            squares: List[Square] = []
            for col in range(self.cols + 1):
                is_end = col == self.cols
                letter = None if is_end else self.grid[row][col]
                if not is_end and letter != BLOCK:
                    squares.append((row, col))
                    continue
                if squares:
                    if all_checked or len(squares) > 1:
                        word = "".join(self.grid[r][c] for r, c in squares)
                        self.add_slot(squares, word)
                    squares = []

        # Generate down words.
        for col in range(self.cols):
            squares = []
            for row in range(self.rows + 1):
                is_end = row == self.rows
                letter = None if is_end else self.grid[row][col]
                if not is_end and letter != BLOCK:
                    squares.append((row, col))
                    continue
                if squares:
                    if all_checked or len(squares) > 1:
                        word = "".join(self.grid[r][c] for r, c in squares)
                        self.add_slot(squares, word)
                    squares = []

        self.generate_crossings()


class Wordlist:
    """Collection of candidate words for filling the crossword."""

    def __init__(self, words: Iterable[str]) -> None:
        self.words: set[str] = set(word.upper() for word in words)
        self.added_words: set[str] = set()
        self.pattern_matches: Dict[str, Tuple[str, ...]] = {}
        self.indices: defaultdict[int, Dict[int, Dict[str, set[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        self.lengths: defaultdict[int, set[str]] = defaultdict(set)
        self._init_indices()

    def _init_indices(self) -> None:
        for word in self.words:
            self._add_word_to_indices(word)

    def _add_word_to_indices(self, word: str) -> None:
        length = len(word)
        self.lengths[length].add(word)
        for index, letter in enumerate(word):
            self.indices[length][index][letter].add(word)

    def _remove_word_from_indices(self, word: str) -> None:
        length = len(word)
        if word in self.lengths[length]:
            self.lengths[length].remove(word)
            if not self.lengths[length]:
                del self.lengths[length]
        for index, letter in enumerate(word):
            if letter in self.indices[length][index]:
                self.indices[length][index][letter].discard(word)
                if not self.indices[length][index][letter]:
                    del self.indices[length][index][letter]
            if not self.indices[length][index]:
                del self.indices[length][index]
        if length in self.indices and not self.indices[length]:
            del self.indices[length]

    def add_word(self, word: str) -> None:
        normalized = word.upper()
        if normalized not in self.words:
            self.words.add(normalized)
            self.added_words.add(normalized)
            self._add_word_to_indices(normalized)
            self.pattern_matches.clear()

    def remove_word(self, word: str) -> None:
        normalized = word.upper()
        if normalized in self.words:
            self.words.remove(normalized)
            self._remove_word_from_indices(normalized)
            self.pattern_matches.clear()
        self.added_words.discard(normalized)

    def get_matches(self, pattern: str) -> WordMatches:
        normalized = pattern.upper()
        if normalized in self.pattern_matches:
            return list(self.pattern_matches[normalized])

        length = len(normalized)
        if length not in self.lengths:
            self.pattern_matches[normalized] = ()
            return []

        candidates: Optional[set[str]] = None
        for index, letter in enumerate(normalized):
            if letter == EMPTY:
                continue
            letter_matches = self.indices[length][index].get(letter)
            if not letter_matches:
                self.pattern_matches[normalized] = ()
                return []
            candidates = letter_matches if candidates is None else candidates & letter_matches
            if candidates == set():
                self.pattern_matches[normalized] = ()
                return []

        if candidates is None:
            candidates = set(self.lengths[length])
        else:
            candidates = set(candidates)

        matches = tuple(candidate for candidate in candidates if self._matches_pattern(candidate, normalized))
        self.pattern_matches[normalized] = matches
        return list(matches)

    @staticmethod
    def _matches_pattern(word: str, pattern: str) -> bool:
        return all(pattern[i] in {EMPTY, letter} for i, letter in enumerate(word))


class DLXNode:
    """Node used by the dancing links data structure."""

    def __init__(self, column: Optional["ColumnNode"] = None) -> None:
        self.left: "DLXNode" = self
        self.right: "DLXNode" = self
        self.up: "DLXNode" = self
        self.down: "DLXNode" = self
        self.column: Optional["ColumnNode"] = column
        self.row_data: Optional[DLXRowData] = None


class ColumnNode(DLXNode):
    """Column header node for dancing links."""

    def __init__(self, key: ColumnKey, is_primary: bool) -> None:
        super().__init__(self)
        self.key = key
        self.is_primary = is_primary
        self.size = 0


@dataclass
class ExactCoverMatrix:
    """Container capturing the DLX matrix for a crossword."""

    root: ColumnNode
    columns: Dict[ColumnKey, ColumnNode]
    row_nodes: Dict[Tuple[Slot, str], DLXNode]
    row_data: Dict[Tuple[Slot, str], DLXRowData]


class DLXBuilder:
    """Helper to assemble the DLX matrix for the crossword fill."""

    def __init__(self) -> None:
        self.root = ColumnNode(ColumnKey("root", "root"), is_primary=False)
        self.columns: Dict[ColumnKey, ColumnNode] = {}
        self.row_nodes: Dict[Tuple[Slot, str], DLXNode] = {}
        self.row_data: Dict[Tuple[Slot, str], DLXRowData] = {}

    def get_column(self, key: ColumnKey, is_primary: bool) -> ColumnNode:
        column = self.columns.get(key)
        if column is not None:
            return column
        column = ColumnNode(key, is_primary=is_primary)
        # Insert new column to the left of root.
        column.left = self.root.left
        column.right = self.root
        column.left.right = column
        column.right.left = column
        self.columns[key] = column
        return column

    def add_row(self, columns: Sequence[ColumnNode], data: DLXRowData) -> None:
        if not columns:
            return

        first_node: Optional[DLXNode] = None
        for column in columns:
            node = DLXNode(column)
            node.row_data = data

            # Vertical linkage within column.
            node.down = column
            node.up = column.up
            column.up.down = node
            column.up = node
            column.size += 1

            # Horizontal linkage within row.
            if first_node is None:
                first_node = node
            else:
                node.left = first_node.left
                node.right = first_node
                node.left.right = node
                node.right.left = node

        if first_node is None:
            return

        key = (data.slot, data.word)
        self.row_nodes[key] = first_node
        self.row_data[key] = data

    def build(self) -> ExactCoverMatrix:
        return ExactCoverMatrix(self.root, self.columns, self.row_nodes, self.row_data)


class DLXSolver:
    """Algorithm X solver operating on the crossword DLX matrix."""

    def __init__(self, matrix: ExactCoverMatrix, crossword: Crossword) -> None:
        self.matrix = matrix
        self.root = matrix.root
        self.crossword = crossword
        self.solution: List[DLXRowData] = []
        self.used_words: set[str] = set()
        self.assigned_letters: Dict[Square, str] = {}
        self.assignment_counts: Dict[Square, int] = {}

    def cover(self, column: ColumnNode) -> None:
        column.right.left = column.left
        column.left.right = column.right
        row = column.down
        while row != column:
            node = row.right
            while node != row:
                node.down.up = node.up
                node.up.down = node.down
                if node.column is not None:
                    node.column.size -= 1
                node = node.right
            row = row.down

    def uncover(self, column: ColumnNode) -> None:
        row = column.up
        while row != column:
            node = row.left
            while node != row:
                if node.column is not None:
                    node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row = row.up
        column.right.left = column
        column.left.right = column

    def _cover_row_columns(self, row_node: DLXNode) -> None:
        node = row_node.right
        while node != row_node:
            column = node.column
            if column is not None and column.is_primary:
                self.cover(column)
            node = node.right

    def _uncover_row_columns(self, row_node: DLXNode) -> None:
        node = row_node.left
        while node != row_node:
            column = node.column
            if column is not None and column.is_primary:
                self.uncover(column)
            node = node.left

    def _assign_letters(self, row_data: DLXRowData) -> None:
        for square, letter in row_data.square_letters.items():
            count = self.assignment_counts.get(square)
            if count is None:
                self.assignment_counts[square] = 1
                self.assigned_letters[square] = letter
            else:
                self.assignment_counts[square] = count + 1

    def _unassign_letters(self, row_data: DLXRowData) -> None:
        for square in row_data.square_letters:
            count = self.assignment_counts.get(square)
            if count is None:
                continue
            if count == 1:
                del self.assignment_counts[square]
                self.assigned_letters.pop(square, None)
            else:
                self.assignment_counts[square] = count - 1

    def _row_compatible(self, row_data: DLXRowData) -> bool:
        word = row_data.word
        if word in self.used_words:
            return False
        for square, letter in row_data.square_letters.items():
            existing = self.assigned_letters.get(square)
            if existing is not None and existing != letter:
                return False
        return True

    def _choose_column(self) -> Optional[ColumnNode]:
        best: Optional[ColumnNode] = None
        best_size = math.inf
        column = self.root.right
        while isinstance(column, ColumnNode) and column != self.root:
            if column.is_primary and column.size < best_size:
                best = column
                best_size = column.size
                if best_size == 0:
                    break
            column = column.right
        return best

    def _search(self) -> bool:
        column = self._choose_column()
        if column is None:
            return True
        if column.size == 0:
            return False

        self.cover(column)
        row = column.down
        while row != column:
            row_data = row.row_data
            if row_data is not None and self._row_compatible(row_data):
                self.solution.append(row_data)
                self.used_words.add(row_data.word)
                self._assign_letters(row_data)
                self._cover_row_columns(row)

                if self._search():
                    return True

                self._uncover_row_columns(row)
                self._unassign_letters(row_data)
                self.used_words.discard(row_data.word)
                self.solution.pop()

            row = row.down

        self.uncover(column)
        return False

    def _apply_prefilled(self) -> bool:
        for slot, pattern in self.crossword.words.items():
            if not Crossword.is_word_filled(pattern):
                continue
            key = (slot, pattern)
            row_node = self.matrix.row_nodes.get(key)
            row_data = self.matrix.row_data.get(key)
            if row_node is None or row_data is None:
                return False
            if not self._row_compatible(row_data):
                return False
            column = row_node.column
            if column is None:
                return False
            self.solution.append(row_data)
            self.used_words.add(row_data.word)
            self._assign_letters(row_data)
            self.cover(column)
            self._cover_row_columns(row_node)
        return True

    def solve(self) -> Optional[List[DLXRowData]]:
        if not self._apply_prefilled():
            return None
        if self._search():
            return list(self.solution)
        return None


class Filler(ABC):
    """Abstract base class for crossword filling strategies."""

    @abstractmethod
    def fill(self, crossword: Crossword, wordlist: Wordlist, animate: bool):
        """Fill *crossword* using *wordlist* and return a success flag."""

    @staticmethod
    def find_quad_stack_slots(
        crossword: Crossword, width: int = 15, height: int = 4
    ) -> List[Slot]:
        """Find a contiguous block of across slots spanning the central rows."""
        rows = getattr(crossword, "rows", None)
        cols = getattr(crossword, "cols", None)
        if rows is None or cols is None:
            return []
        if height <= 0 or height > rows:
            return []
        if width > cols:
            return []

        start_row = (rows - height) // 2
        stack_slots: List[Slot] = []
        for offset in range(height):
            row = start_row + offset
            slot_in_row: Optional[Slot] = None
            for slot in crossword.slots:
                if len(slot) != width:
                    continue
                if any(square_row != row for square_row, _ in slot):
                    continue
                if slot[0][1] != 0 or slot[-1][1] != width - 1:
                    continue
                slot_in_row = slot
                break
            if slot_in_row is None:
                return []
            stack_slots.append(slot_in_row)

        stack_slots.sort(key=lambda slot: slot[0][0])
        return stack_slots

    @staticmethod
    def _is_valid_quad_candidate(
        crossword: Crossword, wordlist: Wordlist, slot: Slot, match: str
    ) -> bool:
        if match not in wordlist.words:
            return False
        if crossword.is_dupe(match):
            return False

        new_crossing_words = Filler.get_new_crossing_words(crossword, slot, match)
        if len(set(new_crossing_words)) != len(new_crossing_words):
            return False

        for crossing_word in new_crossing_words:
            if Crossword.is_word_filled(crossing_word):
                if crossing_word not in wordlist.words:
                    return False
                if crossword.is_dupe(crossing_word):
                    return False
            else:
                if not wordlist.get_matches(crossing_word):
                    return False

        return True

    @staticmethod
    def _solve_quad_stack_bool(
        crossword: Crossword,
        wordlist: Wordlist,
        stack_slots: Sequence[Slot],
        fill_remaining,
    ) -> bool:
        original_words = {slot: crossword.words[slot] for slot in stack_slots}

        def backtrack(index: int) -> bool:
            if index == len(stack_slots):
                return fill_remaining()

            slot = stack_slots[index]
            pattern = crossword.words[slot]
            matches = list(wordlist.get_matches(pattern))
            shuffle(matches)
            previous_word = pattern

            for match in matches:
                if not Filler._is_valid_quad_candidate(crossword, wordlist, slot, match):
                    continue
                crossword.put_word(match, slot)
                if backtrack(index + 1):
                    return True
                crossword.put_word(previous_word, slot)
            return False

        success = backtrack(0)
        if not success:
            for slot, word in original_words.items():
                crossword.put_word(word, slot)
        return success

    @staticmethod
    def _solve_quad_stack_backjump(
        crossword: Crossword,
        wordlist: Wordlist,
        stack_slots: Sequence[Slot],
        fill_remaining,
    ):
        original_words = {slot: crossword.words[slot] for slot in stack_slots}

        def backtrack(index: int):
            if index == len(stack_slots):
                return fill_remaining()

            slot = stack_slots[index]
            pattern = crossword.words[slot]
            matches = list(wordlist.get_matches(pattern))
            shuffle(matches)
            previous_word = pattern

            for match in matches:
                if not Filler._is_valid_quad_candidate(crossword, wordlist, slot, match):
                    continue
                crossword.put_word(match, slot)
                result = backtrack(index + 1)
                if result[0]:
                    return result
                crossword.put_word(previous_word, slot)
                failed_slot = result[1]
                if failed_slot not in crossword.crossings[slot]:
                    return False, failed_slot

            return False, slot

        result = backtrack(0)
        if not result[0]:
            for slot, word in original_words.items():
                crossword.put_word(word, slot)
        return result

    @staticmethod
    def get_new_crossing_words(crossword: Crossword, slot: Slot, word: str) -> List[str]:
        """Return the words that would be formed in crossing slots."""
        new_crossing_words: List[str] = []

        for crossing_slot, squares in crossword.crossings[slot].items():
            crossing_word = crossword.words[crossing_slot]
            new_crossing_word = crossing_word
            for square in squares:
                index = crossword.squares[square][slot]
                letter = word[index]
                crossing_index = crossword.squares[square][crossing_slot]
                new_crossing_word = (
                    new_crossing_word[:crossing_index]
                    + letter
                    + new_crossing_word[crossing_index + 1 :]
                )

            if Crossword.is_word_filled(crossing_word) and crossing_word == new_crossing_word:
                continue

            new_crossing_words.append(new_crossing_word)

        return new_crossing_words

    @staticmethod
    def is_valid_match(crossword: Crossword, wordlist: Wordlist, slot: Slot, match: str) -> bool:
        if match not in wordlist.words:
            return False
        if crossword.is_dupe(match):
            return False

        new_crossing_words = Filler.get_new_crossing_words(crossword, slot, match)
        if len(set(new_crossing_words)) != len(new_crossing_words):
            return False

        for crossing_word in new_crossing_words:
            if Crossword.is_word_filled(crossing_word):
                if crossing_word not in wordlist.words:
                    return False
                if crossword.is_dupe(crossing_word):
                    return False
            else:
                if not wordlist.get_matches(crossing_word):
                    return False

        return True

    @staticmethod
    def fewest_matches(crossword: Crossword, wordlist: Wordlist) -> Tuple[Optional[Slot], WordMatches]:
        fewest_slot: Optional[Slot] = None
        fewest_matches_list: WordMatches = []
        fewest_count: Optional[int] = None

        for slot, pattern in crossword.words.items():
            if Crossword.is_word_filled(pattern):
                continue
            matches = wordlist.get_matches(pattern)
            count = len(matches)
            if fewest_slot is None or count < (fewest_count or math.inf):
                fewest_slot = slot
                fewest_matches_list = matches
                fewest_count = count
                if count == 0:
                    break

        return fewest_slot, fewest_matches_list

    @staticmethod
    def minlook(
        crossword: Crossword,
        wordlist: Wordlist,
        slot: Slot,
        matches: WordMatches,
        k: int,
    ) -> Tuple[int, set[int]]:
        match_indices = range(min(k, len(matches)))
        failed_indices: set[int] = set()

        best_match_index = -1
        best_cross_product = float("-inf")

        for match_index in match_indices:
            cross_product = 0.0
            for crossing_word in Filler.get_new_crossing_words(
                crossword, slot, matches[match_index]
            ):
                num_matches = len(wordlist.get_matches(crossing_word))
                if num_matches == 0:
                    failed_indices.add(match_index)
                    cross_product = float("-inf")
                    break
                cross_product += math.log(num_matches)
            if cross_product > best_cross_product:
                best_match_index = match_index
                best_cross_product = cross_product

        return best_match_index, failed_indices


class DFSFiller(Filler):
    """Depth-first search filling strategy."""

    def fill(self, crossword: Crossword, wordlist: Wordlist, animate: bool) -> bool:
        stack_slots = self.find_quad_stack_slots(crossword)
        if stack_slots and any(
            not Crossword.is_word_filled(crossword.words[slot]) for slot in stack_slots
        ):
            return self._solve_quad_stack_bool(
                crossword,
                wordlist,
                stack_slots,
                lambda: self._fill_recursive(crossword, wordlist, animate),
            )

        return self._fill_recursive(crossword, wordlist, animate)

    def _fill_recursive(self, crossword: Crossword, wordlist: Wordlist, animate: bool) -> bool:
        if animate:
            utils.clear_terminal()
            print(crossword)

        if crossword.is_filled():
            return True

        slot, matches = self.fewest_matches(crossword, wordlist)
        if slot is None:
            return True
        if not matches:
            return False

        pattern = crossword.words[slot]
        shuffle(matches)
        for match in matches:
            if not self.is_valid_match(crossword, wordlist, slot, match):
                continue
            crossword.put_word(match, slot)
            if self._fill_recursive(crossword, wordlist, animate):
                return True
            crossword.put_word(pattern, slot)

        return False


class DFSBackjumpFiller(Filler):
    """Depth-first search strategy that performs backjumping."""

    def fill(self, crossword: Crossword, wordlist: Wordlist, animate: bool):
        stack_slots = self.find_quad_stack_slots(crossword)
        if stack_slots and any(
            not Crossword.is_word_filled(crossword.words[slot]) for slot in stack_slots
        ):
            return self._solve_quad_stack_backjump(
                crossword,
                wordlist,
                stack_slots,
                lambda: self._fill_recursive(crossword, wordlist, animate),
            )

        return self._fill_recursive(crossword, wordlist, animate)

    def _fill_recursive(self, crossword: Crossword, wordlist: Wordlist, animate: bool):
        if animate:
            utils.clear_terminal()
            print(crossword)

        if crossword.is_filled():
            return True, None

        slot, matches = self.fewest_matches(crossword, wordlist)
        if slot is None:
            return True, None
        if not matches:
            return False, slot

        pattern = crossword.words[slot]
        shuffle(matches)
        for match in matches:
            if not self.is_valid_match(crossword, wordlist, slot, match):
                continue
            crossword.put_word(match, slot)
            is_filled, failed_slot = self._fill_recursive(crossword, wordlist, animate)
            if is_filled:
                return True, None
            crossword.put_word(pattern, slot)
            if failed_slot not in crossword.crossings[slot]:
                return False, failed_slot

        crossword.put_word(pattern, slot)
        return False, slot


class MinlookFiller(Filler):
    """Depth-first search strategy using the min-look heuristic."""

    def __init__(self, k: int) -> None:
        self.k = k

    def fill(self, crossword: Crossword, wordlist: Wordlist, animate: bool) -> bool:
        stack_slots = self.find_quad_stack_slots(crossword)
        if stack_slots and any(
            not Crossword.is_word_filled(crossword.words[slot]) for slot in stack_slots
        ):
            return self._solve_quad_stack_bool(
                crossword,
                wordlist,
                stack_slots,
                lambda: self._fill_recursive(crossword, wordlist, animate),
            )

        return self._fill_recursive(crossword, wordlist, animate)

    def _fill_recursive(self, crossword: Crossword, wordlist: Wordlist, animate: bool) -> bool:
        if animate:
            utils.clear_terminal()
            print(crossword)

        if crossword.is_filled():
            return True

        slot, matches = self.fewest_matches(crossword, wordlist)
        if slot is None:
            return True
        if not matches:
            return False

        pattern = crossword.words[slot]
        while matches:
            match_index, failed_indices = self.minlook(crossword, wordlist, slot, matches, self.k)
            remaining = [word for idx, word in enumerate(matches) if idx not in failed_indices and idx != match_index]
            if match_index == -1:
                matches = remaining
                continue

            match = matches[match_index]
            matches = remaining

            if not self.is_valid_match(crossword, wordlist, slot, match):
                continue

            crossword.put_word(match, slot)
            if self._fill_recursive(crossword, wordlist, animate):
                return True
            crossword.put_word(pattern, slot)

        crossword.put_word(pattern, slot)
        return False


class MinlookBackjumpFiller(Filler):
    """Min-look strategy augmented with backjumping."""

    def __init__(self, k: int) -> None:
        self.k = k

    def fill(self, crossword: Crossword, wordlist: Wordlist, animate: bool):
        stack_slots = self.find_quad_stack_slots(crossword)
        if stack_slots and any(
            not Crossword.is_word_filled(crossword.words[slot]) for slot in stack_slots
        ):
            return self._solve_quad_stack_backjump(
                crossword,
                wordlist,
                stack_slots,
                lambda: self._fill_recursive(crossword, wordlist, animate),
            )

        return self._fill_recursive(crossword, wordlist, animate)

    def _fill_recursive(self, crossword: Crossword, wordlist: Wordlist, animate: bool):
        if animate:
            utils.clear_terminal()
            print(crossword)

        if crossword.is_filled():
            return True, None

        slot, matches = self.fewest_matches(crossword, wordlist)
        if slot is None:
            return True, None
        if not matches:
            return False, slot

        pattern = crossword.words[slot]
        while matches:
            match_index, failed_indices = self.minlook(crossword, wordlist, slot, matches, self.k)
            remaining = [word for idx, word in enumerate(matches) if idx not in failed_indices and idx != match_index]
            if match_index == -1:
                matches = remaining
                continue

            match = matches[match_index]
            matches = remaining

            if not self.is_valid_match(crossword, wordlist, slot, match):
                continue

            crossword.put_word(match, slot)
            is_filled, failed_slot = self._fill_recursive(crossword, wordlist, animate)
            if is_filled:
                return True, None
            crossword.put_word(pattern, slot)
            if failed_slot not in crossword.crossings[slot]:
                return False, failed_slot

        crossword.put_word(pattern, slot)
        return False, slot


class DLXFiller(Filler):
    """Exact-cover based crossword filler using Algorithm X."""

    SLOT_KIND = "slot"
    LETTER_KIND = "letter"

    def _build_exact_cover(self, crossword: Crossword, wordlist: Wordlist) -> ExactCoverMatrix:
        builder = DLXBuilder()
        slot_columns: Dict[Slot, ColumnNode] = {}
        for slot in crossword.slots:
            key = ColumnKey(self.SLOT_KIND, slot)
            slot_columns[slot] = builder.get_column(key, is_primary=True)

        crossing_squares: set[Square] = {
            square
            for square, slot_map in crossword.squares.items()
            if len(slot_map) > 1
        }

        letter_columns: Dict[Tuple[Square, str], ColumnNode] = {}
        letter_keys: Dict[Tuple[Square, str], ColumnKey] = {}

        for slot in crossword.slots:
            pattern = crossword.words[slot]
            matches = list(wordlist.get_matches(pattern))
            normalized_pattern = pattern.upper()
            if Crossword.is_word_filled(pattern) and normalized_pattern not in matches:
                matches.append(normalized_pattern)

            unique_matches = sorted({match.upper() for match in matches})
            for match in unique_matches:
                if len(match) != len(slot):
                    continue

                square_letters = {
                    square: match[index]
                    for index, square in enumerate(slot)
                }

                columns: List[ColumnNode] = [slot_columns[slot]]
                row_letter_keys: List[ColumnKey] = []
                for index, square in enumerate(slot):
                    if square not in crossing_squares:
                        continue
                    letter = match[index]
                    key_value = (square, letter)
                    column = letter_columns.get(key_value)
                    if column is None:
                        column_key = ColumnKey(self.LETTER_KIND, key_value)
                        letter_keys[key_value] = column_key
                        column = builder.get_column(column_key, is_primary=False)
                        letter_columns[key_value] = column
                    else:
                        column_key = letter_keys[key_value]
                    columns.append(column)
                    row_letter_keys.append(column_key)

                row_data = DLXRowData(
                    slot=slot,
                    word=match,
                    square_letters=square_letters,
                    letter_columns=tuple(row_letter_keys),
                )
                builder.add_row(columns, row_data)

        return builder.build()

    def fill(self, crossword: Crossword, wordlist: Wordlist, animate: bool):
        matrix = self._build_exact_cover(crossword, wordlist)
        solver = DLXSolver(matrix, crossword)
        solution = solver.solve()
        if solution is None:
            return False

        original_words = {slot: word for slot, word in crossword.words.items()}
        try:
            for row in solution:
                crossword.put_word(row.word, row.slot)
        except Exception:
            for slot, word in original_words.items():
                crossword.put_word(word, slot)
            return False

        if animate:
            utils.clear_terminal()
            print(crossword)
        return True


WORDLIST_FOLDER = "wordlist"
GRID_FOLDER = "grid"
GRID_SUFFIX = ".txt"


@contextmanager
def prioritize_words(
    wordlist: "Wordlist",
    priority_words: Iterable[str],
    *,
    patch_shuffle: bool = False,
):
    """Temporarily bias :meth:`Wordlist.get_matches` toward *priority_words*.

    The provided *priority_words* are added to ``wordlist`` for the duration of
    the context (if not already present) and returned first from
    :meth:`Wordlist.get_matches`. Optionally, :func:`shuffle` can be neutralised
    to preserve the ordering of the prioritised words.
    """

    priority = tuple(word.upper() for word in priority_words)
    added: List[str] = []
    for word in priority:
        if word not in wordlist.words:
            wordlist.add_word(word)
            added.append(word)

    original_get_matches = wordlist.get_matches

    def prioritized_get_matches(self: "Wordlist", pattern: str) -> List[str]:
        matches = original_get_matches(pattern)
        if not matches:
            return matches
        prioritized = [word for word in priority if word in matches]
        if not prioritized:
            return matches
        remaining = [word for word in matches if word not in priority]
        return prioritized + remaining

    wordlist.get_matches = prioritized_get_matches.__get__(wordlist, type(wordlist))

    original_shuffle = shuffle
    if patch_shuffle:
        def no_shuffle(sequence: List[str]) -> None:
            return None

        globals()["shuffle"] = no_shuffle

    try:
        yield
    finally:
        wordlist.get_matches = original_get_matches
        if patch_shuffle:
            globals()["shuffle"] = original_shuffle
        for word in added:
            wordlist.remove_word(word)


def generate_wildcard_layouts(
    grid_rows: Iterable[str],
) -> Iterator[Tuple[Tuple[str, ...], Dict[Square, str]]]:
    """Yield sanitized grid layouts for every wildcard assignment.

    The returned ``assignments`` mapping only contains coordinates promoted to
    :data:`BLOCK`; wildcards left white are omitted because the sanitized rows
    already reflect their :data:`EMPTY` state.
    """

    rows = tuple(
        row if isinstance(row, str) else "".join(str(char) for char in row)
        for row in grid_rows
    )
    if not rows:
        yield tuple(), {}
        return

    working_grid = [list(row) for row in rows]
    cols = len(working_grid[0])
    if any(len(row) != cols for row in working_grid):
        raise ValueError("all grid rows must have the same length")

    rows_count = len(working_grid)
    wildcard_pairs: List[Tuple[Square, Square]] = []
    seen: set[Square] = set()

    for r in range(rows_count):
        for c in range(cols):
            if working_grid[r][c] != "+" or (r, c) in seen:
                continue
            counterpart = (rows_count - 1 - r, cols - 1 - c)
            counter_value = working_grid[counterpart[0]][counterpart[1]]
            if counter_value != "+":
                raise ValueError(
                    f"wildcard at {(r, c)} must be paired with '+' at {counterpart}"
                )
            seen.add((r, c))
            seen.add(counterpart)
            wildcard_pairs.append(((r, c), counterpart))

    max_blocks = math.floor(0.2 * rows_count * cols)
    existing_blocks = sum(
        1 for row in working_grid for cell in row if cell == BLOCK
    )
    assignments: Dict[Square, str] = {}

    def backtrack(index: int, block_count: int) -> Iterator[Tuple[Tuple[str, ...], Dict[Square, str]]]:
        if index == len(wildcard_pairs):
            sanitized_rows = tuple(
                "".join(EMPTY if cell == "+" else cell for cell in row)
                for row in working_grid
            )
            yield sanitized_rows, dict(assignments)
            return

        first, second = wildcard_pairs[index]
        coord_list: List[Square] = [first]
        if second != first:
            coord_list.append(second)

        replaced: List[Tuple[Square, str]] = []
        for coord in coord_list:
            r, c = coord
            replaced.append((coord, working_grid[r][c]))
            working_grid[r][c] = EMPTY
        yield from backtrack(index + 1, block_count)
        for coord, original in replaced:
            r, c = coord
            working_grid[r][c] = original

        block_increment = len(coord_list)
        if block_count + block_increment <= max_blocks:
            block_replaced: List[Tuple[Square, str]] = []
            for coord in coord_list:
                r, c = coord
                block_replaced.append((coord, working_grid[r][c]))
                working_grid[r][c] = BLOCK
                assignments[coord] = BLOCK
            yield from backtrack(index + 1, block_count + block_increment)
            for coord, original in block_replaced:
                r, c = coord
                working_grid[r][c] = original
                assignments.pop(coord, None)

    if not wildcard_pairs:
        sanitized_rows = tuple(
            "".join(EMPTY if cell == "+" else cell for cell in row)
            for row in working_grid
        )
        yield sanitized_rows, {}
        return

    yield from backtrack(0, existing_blocks)


def _format_wildcard_assignment(assignments: Dict[Square, str]) -> str:
    parts: List[str] = []
    for coord in sorted(assignments):
        value = assignments[coord]
        state = "BLOCK" if value == BLOCK else "EMPTY"
        parts.append(f"{coord}: {state}")
    return "{" + ", ".join(parts) + "}"


def _format_wildcard_assignment_list(assignments_list: Sequence[Dict[Square, str]]) -> str:
    if not assignments_list:
        return "[]"
    formatted = (
        _format_wildcard_assignment(assignments)
        for assignments in assignments_list
    )
    return "[" + ", ".join(formatted) + "]"


def read_grid(filepath: Union[str, os.PathLike[str], Iterable[str]]) -> List[str]:
    """Read a grid file and return a list of strings representing rows."""
    if isinstance(filepath, (str, os.PathLike)):
        with open(filepath, "r", encoding="utf-8") as handle:
            return [line.rstrip("\n") for line in handle]
    return [str(line).rstrip("\n") for line in filepath]


def read_wordlist(
    filepath: Union[str, os.PathLike[str], Iterable[str]],
    scored: bool = True,
    min_score: int = 50,
) -> Wordlist:
    """Read a wordlist file and return a :class:`Wordlist` instance."""
    if isinstance(filepath, (str, os.PathLike)):
        with open(filepath, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
    else:
        lines = [str(line).strip() for line in filepath]

    words: List[str] = []
    for entry in lines:
        if not entry:
            continue
        if scored:
            parts = entry.split(";")
            word = parts[0]
            if len(parts) == 1:
                words.append(word)
                continue
            try:
                score = int(parts[1])
            except ValueError:
                continue
            if score >= min_score:
                words.append(word)
        else:
            words.append(entry)

    return Wordlist(words)


def log_times(times: Sequence[float], strategy: str) -> None:
    """Log timing information for repeated fills."""
    if not times:
        print(f"No crosswords filled using {strategy}")
        return
    print(f"Filled {len(times)} crosswords using {strategy}")
    print(f"Min time: {min(times):.4f} seconds")
    print(f"Avg time: {sum(times) / len(times):.4f} seconds")
    print(f"Max time: {max(times):.4f} seconds")


def get_filler(args) -> Optional[Filler]:
    """Instantiate a filler implementation based on CLI arguments."""
    if args.strategy == "dfs":
        return DFSFiller()
    if args.strategy == "dfsb":
        return DFSBackjumpFiller()
    if args.strategy == "minlook":
        return MinlookFiller(args.k)
    if args.strategy == "mlb":
        return MinlookBackjumpFiller(args.k)
    if args.strategy == "dlx":
        return DLXFiller()
    return None


def run_test(args) -> None:
    """Run the command-line interface entry point."""
    dirname = os.path.dirname(__file__)
    wordlist_path_prefix = os.path.join(dirname, WORDLIST_FOLDER)
    grid_path_prefix = os.path.join(dirname, GRID_FOLDER)

    wordlist = read_wordlist(os.path.join(wordlist_path_prefix, args.wordlist_path))

    if getattr(args, "benchmark_open", False):
        benchmark_grid_path = os.path.join(grid_path_prefix, "15xopen.txt")
        benchmark_grid = read_grid(benchmark_grid_path)
        benchmark_layouts = list(generate_wildcard_layouts(benchmark_grid))
        strategies = ["dfs", "dfsb", "minlook", "mlb", "dlx"]
        print("Running 15xopen benchmarks:\n")
        for strategy in strategies:
            benchmark_args = argparse.Namespace(strategy=strategy, k=args.k)
            times: List[float] = []
            for _ in range(args.num_trials):
                tic = time.time()
                success = False
                attempted_assignments: List[Dict[Square, str]] = []
                for layout_rows, assignments in benchmark_layouts:
                    crossword = AmericanCrossword.from_grid(tuple(layout_rows))
                    filler = get_filler(benchmark_args)
                    if filler is None:
                        raise ValueError(f"unknown strategy: {strategy}")
                    result = filler.fill(crossword, wordlist, False)
                    if isinstance(result, tuple):
                        result = result[0]
                    if result:
                        success = True
                        break
                    attempted_assignments.append(assignments)
                if not success:
                    formatted = _format_wildcard_assignment_list(attempted_assignments)
                    raise RuntimeError(
                        f"failed to fill crossword using {strategy}; wildcard assignments tried: {formatted}"
                    )
                times.append(time.time() - tic)
            log_times(times, strategy)
            print()
        return

    grid_path = os.path.join(grid_path_prefix, args.grid_path)
    if not grid_path.endswith(GRID_SUFFIX):
        grid_path = grid_path + GRID_SUFFIX

    grid = read_grid(grid_path)
    layouts = list(generate_wildcard_layouts(grid))
    times: List[float] = []

    priority_words = None
    canonical_blocks: Optional[set[Square]] = None
    if os.path.basename(grid_path) == "7xopenplus.txt":
        priority_words = WILDCARD_CANONICAL_WORDS
        canonical_blocks = set(WILDCARD_CANONICAL_ASSIGNMENTS)

        def layout_priority(item: Tuple[Tuple[str, ...], Dict[Square, str]]) -> int:
            layout_assignments = {
                coord for coord, value in item[1].items() if value == BLOCK
            }
            return 0 if layout_assignments == canonical_blocks else 1

        layouts.sort(key=layout_priority)

    context = (
        prioritize_words(wordlist, priority_words, patch_shuffle=True)
        if priority_words
        else nullcontext()
    )

    with context:
        for _ in range(args.num_trials):
            tic = time.time()

            successful_crossword: Optional[AmericanCrossword] = None
            attempted_assignments: List[Dict[Square, str]] = []
            for layout_rows, assignments in layouts:
                crossword = AmericanCrossword.from_grid(tuple(layout_rows))
                filler = get_filler(args)
                if filler is None:
                    raise ValueError(f"unknown strategy: {args.strategy}")

                result = filler.fill(crossword, wordlist, args.animate)
                if isinstance(result, tuple):
                    result = result[0]
                if result:
                    successful_crossword = crossword
                    break
                attempted_assignments.append(assignments)

            if successful_crossword is None:
                formatted = _format_wildcard_assignment_list(attempted_assignments)
                raise RuntimeError(
                    f"failed to fill crossword; wildcard assignments tried: {formatted}"
                )

            duration = time.time() - tic
            times.append(duration)

            if not args.animate:
                print(successful_crossword)

            print(
                f"\nFilled {successful_crossword.cols}x{successful_crossword.rows} crossword in {duration:.4f} seconds\n"
            )

    log_times(times, args.strategy)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ye olde swordsmith engine")

    parser.add_argument(
        "-w",
        "--wordlist",
        dest="wordlist_path",
        type=str,
        default="spreadthewordlist.dict",
        help="filepath for wordlist",
    )
    parser.add_argument(
        "-g",
        "--grid",
        dest="grid_path",
        type=str,
        default="15xcommon.txt",
        help="filepath for grid",
    )
    parser.add_argument(
        "-t",
        "--num_trials",
        dest="num_trials",
        type=int,
        default=5,
        help="number of grids to try filling",
    )
    parser.add_argument(
        "-a",
        "--animate",
        default=False,
        action="store_true",
        help="whether to animate grid filling",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        dest="strategy",
        type=str,
        default="dfs",
        help="which algorithm to run: dfs, dfsb, minlook, mlb, dlx",
    )
    parser.add_argument(
        "-k",
        "--k",
        dest="k",
        type=int,
        default=5,
        help="k constant for minlook",
    )
    parser.add_argument(
        "--benchmark-open",
        dest="benchmark_open",
        action="store_true",
        help="benchmark all strategies on the 15xopen grid",
    )
    args = parser.parse_args()

    run_test(args)


__all__ = [
    "EMPTY",
    "BLOCK",
    "Crossword",
    "AmericanCrossword",
    "Wordlist",
    "Filler",
    "DFSFiller",
    "DFSBackjumpFiller",
    "MinlookFiller",
    "MinlookBackjumpFiller",
    "DLXFiller",
    "WILDCARD_CANONICAL_WORDS",
    "WILDCARD_CANONICAL_ASSIGNMENTS",
    "prioritize_words",
    "generate_wildcard_layouts",
    "read_grid",
    "read_wordlist",
    "log_times",
    "get_filler",
    "run_test",
    "main",
]
