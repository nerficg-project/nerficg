"""utils.py: Utility code for script execution."""

import sys
from pathlib import Path


class DiscoverSourcePath:
    """A context class adding the source code location to the current python path."""

    def __enter__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

    def __exit__(self, *_):
        sys.path.pop(0)
