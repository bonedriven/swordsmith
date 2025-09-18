from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    PACKAGE_DIR = Path(__file__).resolve().parent
    if str(PACKAGE_DIR) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR))
    import swordsmith as _swordsmith  # type: ignore
else:
    from . import swordsmith as _swordsmith


_swordsmith.main()
