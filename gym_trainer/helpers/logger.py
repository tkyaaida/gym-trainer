#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataclasses import dataclass
from typing import List
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG


@dataclass
class Module:
    name: str
    level: int = DEBUG
    stream: bool = True
    file: bool = True


class Logger:
    def __init__(self, name: str, fname: str = None, modules: List[Module] = None):
        if fname is None:
            fname = name + '.log'
        self.logger = getLogger(name)
        self.logger.setLevel(DEBUG)

        fmt = Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

        # stdout
        sh = StreamHandler()
        sh.setLevel(DEBUG)
        sh.setFormatter(fmt)

        # file
        fh = FileHandler(filename=fname)
        fh.setLevel(DEBUG)
        fh.setFormatter(fmt)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

        # add module logger
        if modules:
            for module in modules:
                logger = getLogger(module.name)
                logger.setLevel(module.level)
                if module.stream:
                    logger.addHandler(sh)
                if module.file:
                    logger.addHandler(fh)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.debug(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
