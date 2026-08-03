"""Import repo-local easysteer modules without permanently mutating sys.path."""

import os
import sys
from contextlib import contextmanager

# Repository root (the directory containing easysteer/ and frontend/).
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@contextmanager
def project_root_on_path():
    """Temporarily put the repository root on sys.path.

    Use as::

        with project_root_on_path():
            from easysteer.steer.lat import LATExtractor

    sys.path is restored afterwards so pip-installed packages are not
    shadowed by repo-local directories.
    """
    original_path = sys.path.copy()
    try:
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        yield PROJECT_ROOT
    finally:
        sys.path[:] = original_path
