#!/usr/bin/env python
"""Regenerate downstream artifacts. Thin wrapper around `bolt generate`.

Examples:
    python generate.py all
    python generate.py documentation
    python generate.py airflow notebook
"""
from __future__ import annotations

import pathlib
import sys

_VENDOR = pathlib.Path(__file__).resolve().parent / "_boltpipeliner"
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))

from bolt_pipeliner.cli.app import app

if __name__ == "__main__":
    app(["generate", *sys.argv[1:]])
