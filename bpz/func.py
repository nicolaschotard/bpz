# Automatically adapted for numpy Jun 08, 2006 by convertcode.py

#
# func.py: general function objects
# Author: Johann Hibschman <johann@physics.berkeley.edu>
#
# Copyright (C) Johann Hibschman 1997
#

# Enhanced for use with Numeric functions (ufuncs)

# All of the functions are combined with Numeric ufuncs; this loses
# some performance when the functions are used on scalar arguments,
# but should give a big win on vectors.


import numpy as np
import operator
import math
from types import *

ArrayType = type(np.asarray(1.0))
UfuncType = type(np.add)

# unary function objects (maybe rename to UN_FUNC?)


class FuncOps(object):
    """
    Common mix-in operations for function objects.
    Normal function classes are assumed to implement a call routine,
    which will be chained to in the __call__ method.
    """

    def compose(self, f):
        return UnCompose(self, f)

    def __add__(self, f):
        return BinCompose(np.add, self, f)

    def __sub__(self, f):
        return BinCompose(np.subtract, self, f)

    def __mul__(self, f):
        return BinCompose(np.multiply, self, f)

    def __div__(self, f):
        return BinCompose(np.divide, self, f)

    def __neg__(self):
        return UnCompose(np.negative, self)

    def __pow__(self, f):
        return BinCompose(np.power, self, f)

    def __coerce__(self, x):
        # if type(x) in [IntType, FloatType, LongType, ComplexType]:
        if type(x) in [int, float, int, complex]:
            return (self, UnConstant(x))
        else:
            return (self, x)

    def __call__(self, arg):
        "Default call routine, used for ordinary functions."
        if type(arg) == ArrayType:
            return array_map(self.call, arg)
        else:
            return self.call(arg)

    def exp(self):
        return UnCompose(np.exp, self)

    def log(self):
        return UnCompose(np.log, self)


# Bind a normal function
# Should check if the argument is a function.
class FuncBinder(FuncOps):
    def __init__(self, a_f):
        if ((type(a_f) == UfuncType)
            or
            (type(a_f) == InstanceType and
             FuncOps in a_f.__class__.__bases__)):
            self.__call__ = a_f        # overwrite the existing call method
        self.call = a_f


# wrap a constant in a Function class
class UnConstant(FuncOps):
    def __init__(self, x):
        self.constant = x

    def __call__(self, x):
        return self.constant

# just return the argument: f(x) = x
# This is used to build up more complex expressions.


class Identity(FuncOps):
    def __init__(self):
        pass

    def __call__(self, arg):
        return arg


# compose two unary functions
class UnCompose(FuncOps):
    def __init__(self, a_f, a_g):
        self.f = a_f
        self.g = a_g

    def __call__(self, arg):
        return self.f(self.g(arg))


# -------------------------------------------------
# binary function objects

# classes of composition:
#    a,b,c,d: binary functions m,n,o: unary functions
#    d=c.compose(a,b) - c(a(x,y),b(x,y)) - used for a/b, a*b, etc.
#    m=c.compose(n,o) - c(n(x), o(x))
#    d=c.compose(n,o) - c(n(x), o(y))
#    d=m.compose(c)   - m(c(x,y))


class BinFuncOps(object):
    # returns self(f(x), g(x)), a unary function
    def compose(self, f, g):
        return BinCompose(self, f, g)

    # returns self(f(x), g(y)), a binary function
    def compose2(self, f, g):
        return BinUnCompose(self, f, g)

    # returns f(self(x,y)), a binary function
    def compose_by(self, f):
        return UnBinCompose(f, self)

    def __add__(self, f):
        return BinBinCompose(operator.add, self, f)

    def __sub__(self, f):
        return BinBinCompose(operator.sub, self, f)

    def __mul__(self, f):
        return BinBinCompose(operator.mul, self, f)

    def __div__(self, f):
        return BinBinCompose(operator.div, self, f)

    def __pow__(self, f):
        return BinBinCompose(pow, self, f)

    def __neg__(self):
        return UnBinCompose(operator.neg, self)

    def reduce(self, a, axis=0):
        result = np.take(a, [0], axis)
        for i in range(1, a.shape[axis]):
            result = self(result, np.take(a, [i], axis))
        return result

    def accumulate(self, a, axis=0):
        n = len(a.shape)
        sum = np.take(a, [0], axis)
        out = np.zeros(a.shape, a.dtype.char)
        for i in range(1, a.shape[axis]):
            out[all_but_axis(i, axis, n)] = self(sum, take(a, [i], axis))
        return out

    def outer(self, a, b):
        n_a = len(a.shape)
        n_b = len(b.shape)
        a2 = np.reshape(a, a.shape + (1,) * n_b)
        b2 = np.reshape(b, (1,) * n_a + b.shape)

        # duplicate each array in the appropriate directions
        a3 = a2
        for i in range(n_b):
            a3 = np.repeat(a3, (b.shape[i],), n_a + i)
        b3 = b2
        for i in range(n_a):
            b3 = np.repeat(b3, (a.shape[i],), i)

        answer = array_map_2(self, a3, b3)
        return answer


def all_but_axis(i, axis, num_axes):
    """
    Return a slice covering all combinations with coordinate i along
    axis.  (Effectively the hyperplane perpendicular to axis at i.)
    """
    the_slice = ()
    for j in range(num_axes):
        if j == axis:
            the_slice = the_slice + (i,)
        else:
            the_slice = the_slice + (slice(None),)
    return the_slice


class BinCompose(FuncOps):
    def __init__(self, a_binop, a_f, a_g):
        self.binop = a_binop
        self.f = a_f
        self.g = a_g
        self.temp = lambda x, op=a_binop, f=a_f, g=a_g: op(f(x), g(x))

    def __call__(self, arg):
        return self.temp(arg)


class UnBinCompose(BinFuncOps):
    def __init__(self, a_f, a_g):
        self.f = a_f
        self.g = a_g

    def __call__(self, arg1, arg2):
        return self.f(self.g(arg1, arg2))


# compose a two unary functions with a binary function to get a binary
# function: f(g(x), h(y))
class BinUnCompose(BinFuncOps):
    def __init__(self, a_f, a_g, a_h):
        self.f = a_f
        self.g = a_g
        self.h = a_h

    def __call__(self, arg1, arg2):
        return self.f(self.g(arg1), self.h(arg2))


# compose two binary functions together, using a third binary function
# to make the composition: h(f(x,y), g(x,y))
class BinBinCompose(BinFuncOps):
    def __init__(self, a_h, a_f, a_g):
        self.f = a_f
        self.g = a_g
        self.h = a_h

    def __call__(self, arg1, arg2):
        return self.h(self.f(arg1, arg2), self.g(arg1, arg2))

# ----------------------------------------------------
# Array mapping routines


def array_map(f, ar):
    "Apply an ordinary function to all values in an array."
    flat_ar = np.ravel(ar)
    out = np.zeros(len(flat_ar), flat_ar.dtype.char)
    for i in range(len(flat_ar)):
        out[i] = f(flat_ar[i])
    out.shape = ar.shape
    return out


def array_map_2(f, a, b):
    if a.shape != b.shape:
        raise ShapeError
    flat_a = np.ravel(a)
    flat_b = np.ravel(b)
    out = np.zeros(len(flat_a), a.dtype.char)
    for i in range(len(flat_a)):
        out[i] = f(flat_a[i], flat_b[i])
    return np.reshape(out, a.shape)
