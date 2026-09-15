# SPDX-License-Identifier: Apache-2.0
"""Compatibility alias for :mod:`easysteer.extraction.iti`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("easysteer.extraction.iti")
