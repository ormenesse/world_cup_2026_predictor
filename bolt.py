#!/usr/bin/env python
"""Self-contained `bolt` entry point.

Uses the vendored copy of bolt_pipeliner under ./_boltpipeliner/ when present,
falling back to a pip-installed copy otherwise. Forwards all CLI args.
"""
from __future__ import annotations

import pathlib
import sys

_VENDOR = pathlib.Path(__file__).resolve().parent / "_boltpipeliner"
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))

from bolt_pipeliner.cli.app import main

if __name__ == "__main__":
    main()
