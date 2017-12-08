# Automatically adapted for numpy Jun 08, 2006 by convertcode.py

"""Matlab(tm) compatibility functions.

This will hopefully become a complete set of the basic functions available in
matlab.  The syntax is kept as close to the matlab syntax as possible.  One 
fundamental change is that the first index in matlab varies the fastest (as in 
FORTRAN).  That means that it will usually perform reductions over columns, 
whereas with this object the most natural reductions are over rows.  It's perfectly
possible to make this work the way it does in matlab if that's desired.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from past.utils import old_div
import numpy as np
from scipy.special import erf
import numpy.random as RandomArray


def multiples(lo, hi, x=1, eps=1e-7):
    """Returns an array of the multiples of x between [lo,hi] inclusive"""
    l = np.ceil(old_div((lo - eps), x)) * x
    return np.arange(l, hi + eps, x)


def base(b, nums):
    """base(10, [1, 2, 3]) RETURNS 123"""
    if not isinstance(nums, list):
        nums = nums.tolist()
    nums.reverse()
    x = 0
    for i, num in enumerate(nums):
        x += np.array(num) * b**i
    return x


def strbegin(stri, phr):  # coetools.py
    return stri[:len(phr)] == phr


def prange(x, xinclude=None, margin=0.05):
    """RETURNS GOOD RANGE FOR DATA x TO BE PLOTTED IN.
    xinclude = VALUE YOU WANT TO BE INCLUDED IN RANGE.
    margin = FRACTIONAL MARGIN ON EITHER SIDE OF DATA."""
    xmin = min(x)
    xmax = max(x)
    if xinclude is not None:
        xmin = min([xmin, xinclude])
        xmax = max([xmax, xinclude])

    dx = xmax - xmin
    if dx:
        xmin = xmin - dx * margin
        xmax = xmax + dx * margin
    else:
        xmin = xmin - margin
        xmax = xmax + margin
    return [xmin, xmax]


def gaussin(nsigma=1):
    """FRACTION WITHIN nsigma"""
    return erf(old_div(nsigma, np.sqrt(2)))


sigma = gaussin


def floatin(x, l, ndec=3):
    """IS x IN THE LIST l?
    WHO KNOWS WITH FLOATING POINTS!"""
    x = int(x * 10**ndec + 0.1)
    l = (np.array(l) * 10**ndec + 0.1).astype(int).tolist()
    return x in l


# Elementary Matrices


def listo(x):
    if singlevalue(x):
        x = [x]
    return x


def insidepoly1(xp, yp, x, y):
    """DETERMINES WHETHER THE POINT (x, y)
    IS INSIDE THE CONVEX POLYGON DELIMITED BY (xp, yp)"""
    xp, yp = CCWsort(xp, yp)
    xp = xp.tolist()
    yp = yp.tolist()
    if xp[-1] != xp[0]:
        xp.append(xp[0])
        yp.append(yp[0])

    xo = mean(xp)
    yo = mean(yp)
    inpoly = 1
    xa = [xo, x]
    ya = [yo, y]
    for j in range(len(xp) - 1):
        xb = xp[j:j + 2]
        yb = yp[j:j + 2]
        if linescross2(xa, ya, xb, yb):
            inpoly = 0
            break

    return inpoly


def insidepoly(xp, yp, xx, yy):
    """DETERMINES WHETHER THE POINTS (xx, yy)
    ARE INSIDE THE CONVEX POLYGON DELIMITED BY (xp, yp)"""
    xp, yp = CCWsort(xp, yp)
    xx = np.ravel(listo(xx))
    yy = np.ravel(listo(yy))
    inhull = []
    for i in range(len(xx)):
        if i and not (i % 10000):
            print('%d / %d' % (i, len(xx)))
        inhull1 = insidepoly1(xp, yp, xx[i], yy[i])
        inhull.append(inhull1)

    return np.array(inhull).astype(int)


def p2p(x):  # DEFINED AS ptp IN MLab (BELOW)
    return max(x) - min(x)


def rotate(x, y, ang):
    """ROTATES (x, y) BY ang RADIANS CCW"""
    x2 = x * np.cos(ang) - y * np.sin(ang)
    y2 = y * np.cos(ang) + x * np.sin(ang)
    return x2, y2


def rotdeg(x, y, ang):
    """ROTATES (x, y) BY ang DEGREES CCW"""
    return np.rotate(x, y, ang / 180. * np.pi)


def linefit(x1, y1, x2, y2):
    """y = mx + b FIT TO TWO POINTS"""
    if x2 == x1:
        m = np.Inf
        b = np.NaN
    else:
        m = old_div((y2 - y1), (x2 - x1))
        b = y1 - m * x1
    return m, b


def linescross(xa, ya, xb, yb):
    """
    DO THE LINES CONNECTING A TO B CROSS?
    A: TWO POINTS: (xa[0], ya[0]), (xa[1], ya[1])
    B: TWO POINTS: (xb[0], yb[0]), (xb[1], yb[1])
    DRAW LINE FROM A0 TO B0 
    IF A1 & B1 ARE ON OPPOSITE SIDES OF THIS LINE, 
    AND THE SAME IS TRUE VICE VERSA,
    THEN THE LINES CROSS
    """
    if xa[0] == xb[0]:
        xb = list(xb)
        xb[0] = xb[0] + 1e-10

    if xa[1] == xb[1]:
        xb = list(xb)
        xb[1] = xb[1] + 1e-10

    m0, b0 = linefit(xa[0], ya[0], xb[0], yb[0])
    ya1 = m0 * xa[1] + b0
    yb1 = m0 * xb[1] + b0
    cross1 = (ya1 > ya[1]) != (yb1 > yb[1])

    m1, b1 = linefit(xa[1], ya[1], xb[1], yb[1])
    ya0 = m1 * xa[0] + b1
    yb0 = m1 * xb[0] + b1
    cross0 = (ya0 > ya[0]) != (yb0 > yb[0])

    return cross0 and cross1


def linescross2(xa, ya, xb, yb):
    """
    DO THE LINES A & B CROSS?
    DIFFERENT NOTATION:
    LINE A: (xa[0], ya[0]) -> (xa[1], ya[1])
    LINE B: (xb[0], yb[0]) -> (xb[1], yb[1])
    DRAW LINE A
    IF THE B POINTS ARE ON OPPOSITE SIDES OF THIS LINE, 
    AND THE SAME IS TRUE VICE VERSA,
    THEN THE LINES CROSS
    """
    if xa[0] == xa[1]:
        xa = list(xa)
        xa[1] = xa[1] + 1e-10

    if xb[0] == xb[1]:
        xb = list(xb)
        xb[1] = xb[1] + 1e-10

    ma, ba = linefit(xa[0], ya[0], xa[1], ya[1])
    yb0 = ma * xb[0] + ba
    yb1 = ma * xb[1] + ba
    crossb = (yb0 > yb[0]) != (yb1 > yb[1])

    mb, bb = linefit(xb[0], yb[0], xb[1], yb[1])
    ya0 = mb * xa[0] + bb
    ya1 = mb * xa[1] + bb
    crossa = (ya0 > ya[0]) != (ya1 > ya[1])

    return crossa and crossb


def gauss(r, sig=1., normsum=1):
    """GAUSSIAN NORMALIZED SUCH THAT AREA=1"""
    r = np.clip(old_div(r, float(sig)), 0, 10)
    G = np.exp(-0.5 * r**2)
    G = np.where(np.less(r, 10), G, 0)
    if normsum:
        G = G * 0.5 / (np.pi * sig**2)
    return G


def gauss1(r, sig=1.):
    """GAUSSIAN NORMALIZED SUCH THAT PEAK AMPLITUDE = 1"""
    return gauss(r, sig, 0)


def atanxy(x, y, degrees=0):
    """ANGLE CCW FROM x-axis"""
    theta = np.arctan(divsafe(y, x, inf=1e30, nan=0))
    theta = np.where(np.less(x, 0), theta + np.pi, theta)
    theta = np.where(np.logical_and(np.greater(x, 0), np.less(y, 0)),
                     theta + 2 * np.pi, theta)
    if degrees:
        theta = theta * 180. / np.pi
    return theta


def CCWsort(x, y):
    """FOR A CONVEX SET OF POINTS, 
    SORT THEM SUCH THAT THEY GO AROUND IN ORDER CCW FROM THE x-AXIS"""
    xc = mean(x)
    yc = mean(y)
    ang = atanxy(x - xc, y - yc)
    SI = np.array(np.argsort(ang))
    x2 = x.take(SI, 0)
    y2 = y.take(SI, 0)
    return x2, y2


def odd(n):
    """RETURNS WHETHER AN INTEGER IS ODD"""
    return n & 1


def divsafe(a, b, inf=np.Inf, nan=np.NaN):
    """a / b with a / 0 = inf and 0 / 0 = nan"""
    a = np.array(a).astype(float)
    b = np.array(b).astype(float)
    asgn = np.greater_equal(a, 0) * 2 - 1.
    bsgn = np.greater_equal(b, 0) * 2 - 1.
    xsgn = asgn * bsgn
    sgn = np.where(b, xsgn, asgn)
    sgn = np.where(a, xsgn, bsgn)
    babs = np.clip(abs(b), 1e-200, 1e9999)
    bb = bsgn * babs
    return np.where(b, old_div(a, bb), np.where(a, sgn * inf, nan))


def roundint(x):
    if singlevalue(x):
        return(int(round(x)))
    else:
        return np.asarray(x).round().astype(int)


intround = roundint


def singlevalue(x):
    """IS x A SINGLE VALUE?  (AS OPPOSED TO AN ARRAY OR LIST)"""
    return not isinstance(x, (list, np.ndarray))


def roundn(x, ndec=0):
    if singlevalue(x):
        fac = 10.**ndec
        return old_div(roundint(x * fac), fac)
    else:
        rr = []
        for xx in x:
            rr.append(roundn(xx, ndec))
        return np.array(rr)


def logical(x):
    return np.where(x, 1, 0)


def close(x, y, rtol=1.e-5, atol=1.e-8):
    """JUST LIKE THE Numeric FUNCTION allclose, BUT FOR SINGLE VALUES.  (WILL IT BE QUICKER?)"""
    return abs(y - x) < (atol + rtol * abs(y))


def count(a):
    """RETURNS A DICTIONARY WITH THE NUMBER OF TIMES EACH ID OCCURS"""
    bins = norep(a)
    h = histogram(a, bins)
    d = {}
    for i in range(len(h)):
        d[bins[i]] = h[i]
    return d


def rep(a):
    """RETURNS A DICTIONARY WITH THE NUMBER OF TIMES EACH ID IS REPEATED
    1 INDICATES THE VALUE APPEARED TWICE (WAS REPEATED ONCE)"""
    a = np.sort(a)
    d = a[1:] - a[:-1]
    c = np.compress(np.logical_not(d), a)
    if c.any():
        bins = norep(c)
        h = histogram(c, bins)
        d = {}
        for i in range(len(h)):
            d[bins[i]] = h[i]
        return d
    else:
        return {}


def norep(a):
    """RETURNS a w/o REPETITIONS, i.e. THE MEMBERS OF a"""
    a = np.sort(a)
    d = a[1:] - a[:-1]
    c = np.compress(d, a)
    x = np.concatenate((c, [a[-1]]))
    return x


def between(lo, x, hi):  # --DC
    # RETURNS 1 WHERE lo < x < hi
    # (can also use that syntax "lo < x < hi")
    if lo in [None, '']:
        try:
            good = np.ones(len(x)).astype(int)
        except:
            good = 1
    else:
        good = np.greater(x, lo)
    if hi not in [None, '']:
        good = good * np.less(x, hi)
    return good


def ndec(x, max=3):  # --DC
    """RETURNS # OF DECIMAL PLACES IN A NUMBER"""
    for n in range(max, 0, -1):
        if round(x, n) != round(x, n - 1):
            return n
    return 0  # IF ALL ELSE FAILS...  THERE'S NO DECIMALS


def interp(x, xdata, ydata, extrap=0):  # NEW VERSION!
    """DETERMINES y AS LINEAR INTERPOLATION OF 2 NEAREST ydata"""
    SI = np.argsort(xdata)
    xdata = xdata.take(SI, 0)
    ydata = ydata.take(SI, 0)
    ii = np.searchsorted(xdata, x)
    if singlevalue(ii):
        ii = np.array([ii])
    # 0 = before all
    # len(xdata) = after all
    n = len(xdata)
    if extrap:
        i2 = np.clip(ii,   1, n - 1)
        i1 = i2 - 1
    else:
        i2 = np.clip(ii,   0, n - 1)
        i1 = np.clip(ii - 1, 0, n - 1)

    x2 = np.take(xdata, i2)
    x1 = np.take(xdata, i1)
    y2 = np.take(ydata, i2)
    y1 = np.take(ydata, i1)
    # m = (y2 - y1) / (x2 - x1)
    m = divsafe(y2 - y1, x2 - x1, nan=0)
    b = y1 - m * x1
    y = m * x + b
    if len(y) == 0:
        y = y[0]
    return y


interpn = interp


def rand(*args):
    """rand(d1,...,dn) returns a matrix of the given dimensions
    which is initialized to random numbers from a uniform distribution
    in the range [0,1).
    """
    return RandomArray.random(args)


def eye(N, M=None, k=0):
    """eye(N, M=N, k=0, dtype=None) returns a N-by-M matrix where the 
    k-th diagonal is all ones, and everything else is zeros.
    """
    if M is None:
        M = N
    if isinstance(M, str):
        typecode = M
        M = N
    m = np.equal(np.subtract.outer(np.arange(N), np.arange(M)), -k)
    return np.asarray(m, dtype=typecode)


def tri(N, M=None, k=0):
    """tri(N, M=N, k=0, dtype=None) returns a N-by-M matrix where all
    the diagonals starting from lower left corner up to the k-th are all ones.
    """
    if M is None:
        M = N
    if type(M) == type('d'):
        typecode = M
        M = N
    m = np.greater_equal(np.subtract.outer(np.arange(N), np.arange(M)), -k)
    return m.astype(typecode)

# Matrix manipulation


def diag(v, k=0):
    """diag(v,k=0) returns the k-th diagonal if v is a matrix or
    returns a matrix with v as the k-th diagonal if v is a vector.
    """
    v = np.asarray(v)
    s = v.shape
    if len(s) == 1:
        n = s[0] + abs(k)
        if k > 0:
            v = np.concatenate((np.zeros(k, v.dtype.char), v))
        elif k < 0:
            v = np.concatenate((v, np.zeros(-k, v.dtype.char)))
        return eye(n, k=k) * v
    elif len(s) == 2:
        v = np.add.reduce(eye(s[0], s[1], k=k) * v)
        if k > 0:
            return v[k:]
        elif k < 0:
            return v[:k]
        else:
            return v
    else:
        raise ValueError("Input must be 1- or 2-D.")


def max(m):
    """max(m) returns the maximum along the first dimension of m.
    """
    return np.maximum.reduce(m)


def min(m):
    """min(m) returns the minimum along the first dimension of m.
    """
    return np.minimum.reduce(m)


def mean(m, axis=0):
    """mean(m) returns the mean along the first dimension of m.  Note:  if m is
    an integer array, integer division will occur.
    """
    m = np.asarray(m)
    return old_div(np.add.reduce(m, axis=axis), m.shape[axis])


def msort(m):
    """msort(m) returns a sort along the first dimension of m as in MATLAB.
    """
    return np.transpose(np.sort(np.transpose(m)))


def median(m):
    """median(m) returns the median of m along the first dimension of m.
    """
    m = np.asarray(m)
    if m.shape[0] & 1:
        return msort(m)[old_div(m.shape[0], 2)]  # ODD # OF ELEMENTS
    else:
        # EVEN # OF ELEMENTS
        return old_div((msort(m)[old_div(m.shape[0], 2)] + msort(m)[old_div(m.shape[0], 2) - 1]), 2.0)


def rms(m):
    """Root-Mean-Squared, as advertised.
    std (below) first subtracts by the mean
    and later divides by N-1 instead of N"""
    return np.sqrt(mean(m**2))


def std(m):
    """std(m) returns the standard deviation along the first
    dimension of m.  The result is unbiased meaning division by len(m)-1.
    """
    mu = mean(m)
    return old_div(np.sqrt(np.add.reduce(pow(m - mu, 2))), np.sqrt(len(m) - 1.0))


def clip2(m, m_min=None, m_max=None):
    if m_min is None:
        m_min = min(m)
    if m_max is None:
        m_max = max(m)
    return np.clip(m, m_min, m_max)


def total(m):
    """RETURNS THE TOTAL OF THE ENTIRE ARRAY --DC"""
    return np.add.reduce(np.ravel(m))


def size(m):
    """RETURNS THE TOTAL SIZE OF THE ARRAY --DC"""
    s = m.shape
    x = 1
    for n in s:
        x = x * n
    return x


def cumsum(m, axis=0):
    """cumsum(m) returns the cumulative sum of the elements along the
    first dimension of m.
    """
    return np.add.accumulate(m, axis=axis)


def prod(m):
    """prod(m) returns the product of the elements along the first
    dimension of m.
    """
    return np.multiply.reduce(m)


def trapz(y, x=None):
    """trapz(y,x=None) integrates y = f(x) using the trapezoidal rule.
    """
    if x is None:
        d = 1
    else:
        d = diff(x)
    return np.add.reduce(d * (y[1:] + y[0:-1]) / 2.0)


def xbins(x):
    """[-0.5, 0.5, 1] --> [-1, 0, 0.75, 1.25]"""
    d = shorten(x)
    da = x[1] - x[0]
    db = x[-1] - x[-2]
    d = np.concatenate(([x[0] - old_div(da, 2.)], d, [x[-1] + old_div(db, 2.)]))
    return d


def diff(x, n=1):
    """diff(x,n=1) calculates the first-order, discrete difference
    approximation to the derivative.
    """
    if n > 1:
        return diff(x[1:] - x[:-1], n - 1)
    else:
        return x[1:] - x[:-1]


def shorten(x, n=1):  # shrink
    """shorten(x,n=1) 
    SHORTENS x, TAKING AVG OF NEIGHBORS, RECURSIVELY IF n > 1
    """
    a = old_div((x[1:] + x[:-1]), 2.)
    if n > 1:
        return shorten(a, n - 1)
    else:
        return a


def histogram(a, bins):
    n = np.searchsorted(np.sort(a), bins)
    n = np.concatenate([n, [len(a)]])
    return n[1:] - n[:-1]


def histo(a, da=1., amin=None, amax=None):  # --DC
    """
    Histogram of 'a' defined on the bin grid 'bins'
       Usage: h=histogram(p,xp)
    """
    if amin is None:
        amin = min(a)
    if amax is None:
        amax = max(a)
    nnn = old_div((amax - amin), da)
    if np.less(nnn - int(nnn), 1e-4):
        amax = amax + da
    bins = np.arange(amin, amax + da, da)
    n = np.searchsorted(np.sort(a), bins)
    n = np.array(list(map(float, n)))
    return n[1:] - n[:-1]


def Histo(a, da=1., amin=[], amax=[], **other):  # --DC
    if amin == []:
        amin = min(a)
    if amax == []:
        amax = max(a)
    try:
        amin = amin[0]
    except:
        pass
    h = histo(a, da, amin, amax)
    return Histogram(h, amin, da, **other)


def isNaN(x):
    return not (x < 0) and not (x > 0) and (x != 0)


def isnan(x):
    l = np.less(x, 0)
    g = np.greater(x, 0)
    e = np.equal(x, 0)
    n = np.logical_and(np.logical_not(l), np.logical_not(g))
    n = np.logical_and(n, np.logical_not(e))
    return n
