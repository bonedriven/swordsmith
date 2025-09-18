"""Core crossword filling algorithms for the Swordsmith engine."""

from __future__ import annotations

import argparse
import math
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
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

Square = Tuple[int, int]
Slot = Tuple[Square, ...]
WordMatches = List[str]


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
                crossword.grid[r][c] = value
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


WORDLIST_FOLDER = "wordlist"
GRID_FOLDER = "grid"
GRID_SUFFIX = ".txt"


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
    return None


def run_test(args) -> None:
    """Run the command-line interface entry point."""
    dirname = os.path.dirname(__file__)
    wordlist_path_prefix = os.path.join(dirname, WORDLIST_FOLDER)
    grid_path_prefix = os.path.join(dirname, GRID_FOLDER)

    wordlist = read_wordlist(os.path.join(wordlist_path_prefix, args.wordlist_path))

    grid_path = os.path.join(grid_path_prefix, args.grid_path)
    if not grid_path.endswith(GRID_SUFFIX):
        grid_path = grid_path + GRID_SUFFIX

    grid = read_grid(grid_path)
    times: List[float] = []

    for _ in range(args.num_trials):
        tic = time.time()

        crossword = AmericanCrossword.from_grid(grid)
        filler = get_filler(args)
        if filler is None:
            raise ValueError(f"unknown strategy: {args.strategy}")

        result = filler.fill(crossword, wordlist, args.animate)
        if isinstance(result, tuple):
            result = result[0]
        if not result:
            raise RuntimeError("failed to fill crossword")

        duration = time.time() - tic
        times.append(duration)

        if not args.animate:
            print(crossword)

        print(f"\nFilled {crossword.cols}x{crossword.rows} crossword in {duration:.4f} seconds\n")

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
        help="which algorithm to run: dfs, dfsb, minlook, mlb",
    )
    parser.add_argument(
        "-k",
        "--k",
        dest="k",
        type=int,
        default=5,
        help="k constant for minlook",
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
    "read_grid",
    "read_wordlist",
    "log_times",
    "get_filler",
    "run_test",
    "main",
]
