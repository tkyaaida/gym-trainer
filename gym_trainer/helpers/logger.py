#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG


class Logger:
    def __init__(self, name, fname=None):
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
