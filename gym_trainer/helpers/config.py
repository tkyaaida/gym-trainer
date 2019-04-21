#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from pathlib import Path


class Config:
    PKG_ROOT_PATH = Path(__file__).parent.parent

    EXP_OUT_DIR = PKG_ROOT_PATH / 'results'  # type: Path
