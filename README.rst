bpz: Bayesian Photometric Redshifts
===================================

.. image:: https://landscape.io/github/nicolaschotard/bpz/master/landscape.svg?style=flat
   :target: https://landscape.io/github/nicolaschotard/bpz/master
   :alt: Code Health

Introduction
------------

This repository initialy contained a fork of `bpz-1.99.3
<http://www.stsci.edu/~dcoe/BPZ/bpz-1.99.3.tar.gz>`_. This is now a
personal version of BPZ. Not intended to be shared (yet).

Documentation about BPZ can be found on the main BPZ webpage::

  http://www.stsci.edu/~dcoe/BPZ/

Installation
------------

To install this package, do::
    git clone https://github.com/nicolaschotard/bpz.git
    pip install bpz/

Usage
-----

To run ``bpz``, do::
    bpz_run.py UDFtest.cat -INTERP 2
    bpz_finalize.py UDFtest

