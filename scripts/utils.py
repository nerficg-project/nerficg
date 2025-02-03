# -- coding: utf-8 --

"""utils.py: utility code for script execution."""

import sys
from pathlib import Path


class discoverSourcePath:
    """A context class adding the source code location to the current python path."""

    def __enter__(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

    def __exit__(self, *_):
        sys.path.pop(0)


def getCachePath() -> Path:
    """Returns the path to the framework .cache directory."""
    return Path(__file__).resolve().parents[1] / '.cache'
