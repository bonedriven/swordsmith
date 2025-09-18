# Swordsmith Contribution Guidance

## Scope
These instructions apply to the entire repository. Additional nested `AGENTS.md` files may add constraints for specific folders; obey the most specific guidance available.

## Code Style Expectations
- **General**
  - Keep files formatted with Unix newlines and include a trailing newline at the end of each file.
  - Prefer descriptive, lowercase-with-dashes branchless commit messages (imperative mood) and avoid introducing new dependencies without a clear justification.
  - Maintain existing naming conventions and avoid large refactors unrelated to the current change.
- **Python (`python/swordsmith`)**
  - Follow PEP 8 conventions: four-space indentation, `snake_case` for functions/variables, and `UPPER_SNAKE_CASE` for module-level constants.
  - Group imports in the order: standard library, third-party, then local modules, separated by blank lines.
  - When modifying or adding public functions/classes, include concise docstrings describing their purpose and key parameters/return values.
  - Favor readability over micro-optimizations; keep helper functions small and single-purpose.
  - Preserve the existing module structure (e.g., keep shared helpers in `utils.py` and core logic in `swordsmith.py`).

## Required Verification Commands
Run the following and include the results in the PR description:
- `cd python && python3 swordsmith -g 15xcommon -s mlb`
- `cd python/tests && PYTHONPATH=../swordsmith python3 test_swordsmith.py`

If a command is not relevant to your change, explain why in the PR.

## PR Summary Citation Rules
- Reference all substantive code or documentation changes using inline citations in the form ``【F:path/to/file†Lstart-Lend】`` with paths relative to the repository root.
- Cite test or command output with terminal chunk references ``【chunk_id†Lstart-Lend】`` when summarizing verification results.
- Ensure every bullet in the PR summary cites at least one relevant change, and every listed test cites its corresponding command output.
