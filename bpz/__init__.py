#!/usr/bin/env python

"""
bpz: Bayesian Photometric Redshifts

Fork of bpz-1.99.3 (http://www.stsci.edu/~dcoe/BPZ/)

.. moduleauthor:: N. Chotard <nchotard@in2p3.fr>

"""

import os
import glob

# Automatically import all modules (python files)
__all__ = [os.path.basename(m).replace('.py', '') for m in glob.glob("bpz/*.py")
           if '__init__' not in m]

# Set to True if you want to import all previous modules directly
importAll = True

if importAll:
    for pkg in __all__:
        __import__(__name__ + '.' + pkg)
