"""Utility helpers for terminal interaction."""

import os
from typing import Final


_WINDOWS_CLEAR: Final[str] = "cls"
_POSIX_CLEAR: Final[str] = "clear"


def clear_terminal() -> None:
    """Clear the active terminal window."""
    os.system(_WINDOWS_CLEAR if os.name == "nt" else _POSIX_CLEAR)
