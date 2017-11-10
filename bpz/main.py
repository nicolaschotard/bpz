"""Main entry points for scripts."""


from __future__ import print_function
from argparse import ArgumentParser


def bpz(argv=None):
    """Run BPZ."""
    description = """Run BPZ."""
    prog = "bpz.py"

    parser = ArgumentParser(prog=prog, description=description)
    args = parser.parse_args(argv)

    print("This will soon run BPZ")
