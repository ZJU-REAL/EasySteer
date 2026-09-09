"""Import repo-local easysteer modules without permanently mutating sys.path."""

import os
import sys
from contextlib import contextmanager

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@contextmanager
def project_root_on_path():
    """Temporarily put the repository root on sys.path.

    Restore the original path after the import so repository directories do not
    shadow installed packages in later imports.

    Yields:
        The repository root directory.
    """
    original_path = sys.path.copy()
    try:
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        yield PROJECT_ROOT
    finally:
        sys.path[:] = original_path
