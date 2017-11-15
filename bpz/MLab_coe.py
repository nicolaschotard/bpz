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
from scipy.optimize import golden
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


def strbegin(str, phr):  # coetools.py
    return str[:len(phr)] == phr


def prange(x, xinclude=None, margin=0.05):
    """RETURNS GOOD RANGE FOR DATA x TO BE PLOTTED IN.
    xinclude = VALUE YOU WANT TO BE INCLUDED IN RANGE.
    margin = FRACTIONAL MARGIN ON EITHER SIDE OF DATA."""
    xmin = min(x)
    xmax = max(x)
    if xinclude != None:
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


def minmax(x, range=None):
    if range:
        lo, hi = range
        good = between(lo, x, hi)
        x = np.compress(good, x)
    return min(x), max(x)


def Psig(P, nsigma=1):
    """(ir, il) bound central nsigma of P
    -- edges contain equal amounts of P"""
    Pn = old_div(P, total(P))
    g = gausst(nsigma)
    Pl = cumsum(Pn)
    Pr = cumsum(Pn[::-1])
    n = len(P)
    i = np.arange(n)
    il = interp(g, Pl, i)
    ir = interp(g, Pr, i)
    ir = n - ir
    return il, ir


def xsig(x, P, nsigma=1):
    print('xsigmom MUCH MORE ACCURATE THAN xsig IN MLab_coe')
    return old_div(p2p(np.take(x, Psig(P, nsigma))), 2.)


def gaussin(nsigma=1):
    """FRACTION WITHIN nsigma"""
    return erf(old_div(nsigma, np.sqrt(2)))


def gaussp(nsigma=1):
    """FRACTION INCLUDED UP TO nsigma"""
    return 0.5 + old_div(gaussin(nsigma), 2.)


def gaussbtw(nsig1, nsig2):
    """FRACTION BETWEEN nsig1, nsig2"""
    return abs(gaussp(nsig2) - gaussp(nsig1))


sigma = gaussin


def gausst(nsigma=1):
    """FRACTION IN TAIL TO ONE SIDE OF nsigma"""
    return 1 - gaussp(nsigma)


def mom2(x, y):
    return np.sqrt(old_div(total(x**2 * y), total(y)))


def mom2dx(dx, x, y):
    return mom2(x + dx, y)


def xsigmom(x, y):
    """1-sigma of y(x) calculated using moments"""
    dx = golden(mom2dx, (x, y))
    return mom2(x + dx, y)


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

# ~/glens/h0limits/scatterrea.py
# ALSO SEE matplotlib.nxutils.pnpoly & points_inside_poly()
# http://matplotlib.sourceforge.net/faq/howto_faq.html


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


def outside(x, y, xo, yo):
    """GIVEN 3 POINTS a, b, c OF A POLYGON 
    WITH CENTER xo, yo
    DETERMINE WHETHER b IS OUTSIDE ac,
    THAT IS, WHETHER abc IS CONVEX"""
    # DOES o--b CROSS a--c ?
    #      A--B       A--B
    xa, xb, xc = x
    ya, yb, yc = y
    xA = (xo, xa)
    yA = (yo, ya)
    xB = (xb, xc)
    yB = (yb, yc)
    return linescross(xA, yA, xB, yB)


def convexhull(x, y, rep=1, nprev=0):
    """RETURNS THE CONVEX HULL OF x, y
    THAT IS, THE EXTERIOR POINTS"""
    x = x.astype(float)
    y = y.astype(float)
    x, y = CCWsort(x, y)
    xo = mean(x)
    yo = mean(y)
    x = x.tolist()
    y = y.tolist()
    dmax = max([p2p(x), p2p(y)])
    ngood = 0
    while ngood < len(x) + 1:
        dx = x[1] - xo
        dy = y[1] - yo
        dr = np.hypot(dx, dy)
        dx = dx * dmax / dr
        dy = dy * dmax / dr
        x1 = xo - dx
        y1 = yo - dy
        if not outside(x[:3], y[:3], x1, y1):
            del x[1]
            del y[1]
        else:  # ROTATE THE COORD LISTS
            x.append(x.pop(0))
            y.append(y.pop(0))
            ngood += 1

    x = np.array(x)
    y = np.array(y)

    # REPEAT UNTIL CONVERGENCE
    if (nprev == 0) or (len(x) < nprev):
        x, y = convexhull(x, y, nprev=len(x))

    if rep:
        x = np.concatenate((x, [x[0]]))
        y = np.concatenate((y, [y[0]]))

    return x, y


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
    # return np.where(b, a / bb, np.where(a, Inf, NaN))
    return np.where(b, old_div(a, bb), np.where(a, sgn * inf, nan))


def floorint(x):
    return(int(np.floor(x)))


def ceilint(x):
    return(int(np.ceil(x)))


def roundint(x):
    if singlevalue(x):
        return(int(round(x)))
    else:
        return np.asarray(x).round().astype(int)


intround = roundint


def singlevalue(x):
    """IS x A SINGLE VALUE?  (AS OPPOSED TO AN ARRAY OR LIST)"""
    return not isinstance(x, (list, np.ndarray))
    #return type(x) in [type(None), float, float32, float64, int, int0, int8, int16, int32, int64]  # THERE ARE MORE TYPECODES IN Numpy


def roundn(x, ndec=0):
    if singlevalue(x):
        fac = 10.**ndec
        return old_div(roundint(x * fac), fac)
    else:
        rr = []
        for xx in x:
            rr.append(roundn(xx, ndec))
        return np.array(rr)


def percentile(p, x):
    x = np.sort(x)
    i = p * (len(x) - 1.)
    return interp(i, np.arange(len(x)), x)


def logical(x):
    return np.where(x, 1, 0)


def log10clip(x, loexp, hiexp=None):
    if hiexp == None:
        return np.log10(clip2(x, 10.**loexp, None))
    else:
        return np.log10(clip2(x, 10.**loexp, 10.**hiexp))


def linreg(X, Y):
    # written by William Park
    # http://www.python.org/topics/scicomp/recipes_in_python.html
    """ Returns coefficients to the regression line 'y=ax+b' from x[] and y[]. 
    Basically, it solves Sxx a + Sx b = Sxy Sx a + N b = Sy 
    where Sxy = \sum_i x_i y_i, Sx = \sum_i x_i, and Sy = \sum_i y_i. 
    The solution is a = (Sxy N - Sy Sx)/det b = (Sxx Sy - Sx Sxy)/det 
    where det = Sxx N - Sx^2. 
    In addition, 
    Var|a| = s^2 |Sxx Sx|^-1 
    = s^2 | N -Sx| / det |b| |Sx N | |-Sx Sxx| s^2
    = {\sum_i (y_i - \hat{y_i})^2 \over N-2} 
    = {\sum_i (y_i - ax_i - b)^2 \over N-2} 
    = residual / (N-2) R^2 
    = 1 - {\sum_i (y_i - \hat{y_i})^2 \over \sum_i (y_i - \mean{y})^2} 
    = 1 - residual/meanerror 
    It also prints to &lt;stdout&gt; 
    few other data, N, a, b, R^2, s^2, 
    which are useful in assessing the confidence of estimation. """
    if len(X) != len(Y):
        raise ValueError('unequal length')
    N = len(X)
    if N == 2:  # --DC
        a = old_div((Y[1] - Y[0]), (X[1] - X[0]))
        b = Y[0] - a * X[0]
    else:
        Sx = Sy = Sxx = Syy = Sxy = 0.0
        for x, y in map(None, X, Y):
            Sx = Sx + x
            Sy = Sy + y
            Sxx = Sxx + x * x
            Syy = Syy + y * y
            Sxy = Sxy + x * y
        det = Sxx * N - Sx * Sx
        a, b = old_div((Sxy * N - Sy * Sx),
                       det), old_div((Sxx * Sy - Sx * Sxy), det)
        meanerror = residual = 0.0
        for x, y in map(None, X, Y):
            meanerror = meanerror + (y - old_div(Sy, N))**2
            residual = residual + (y - a * x - b)**2
        RR = 1 - old_div(residual, meanerror)
        ss = old_div(residual, (N - 2))
        Var_a, Var_b = ss * N / det, ss * Sxx / det
    print("y=ax+b")
    print("N= %d" % N)
    if N == 2:
        print("a= ", a)
        print("b= ", b)
    else:
        print("a= %g \\pm t_{%d;\\alpha/2} %g" % (a, N - 2, np.sqrt(Var_a)))
        print("b= %g \\pm t_{%d;\\alpha/2} %g" % (b, N - 2, np.sqrt(Var_b)))
        print("R^2= %g" % RR)
        print("s^2= %g" % ss)
    return a, b


def linregrobust(x, y):
    n = len(x)
    a, b = linreg(x, y)
    dy = y - (a * x + b)
    #s = std2(dy)
    s = std(dy)
    good = np.less(abs(dy), 3 * s)
    x, y = np.compress(good, (x, y))
    ng = len(x)
    if ng < n:
        print('REMOVED %d OUTLIER(S), RECALCULATING linreg' % (n - ng))
        a, b = linreg(x, y)
    return a, b


def close(x, y, rtol=1.e-5, atol=1.e-8):
    """JUST LIKE THE Numeric FUNCTION allclose, BUT FOR SINGLE VALUES.  (WILL IT BE QUICKER?)"""
    return abs(y - x) < (atol + rtol * abs(y))


def wherein(x, vals):
    """RETURNS 1 WHERE x IS IN vals"""
    try:
        good = np.zeros(len(x), int)
    except:
        good = 0
    for val in vals:
        good = np.logical_or(good, close(x, val))
    return good


def wherenotin(x, vals):
    """RETURNS 1 WHERE x ISN'T IN vals"""
    return np.logical_not(wherein(x, vals))


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
##     l = []
# for x in ravel(a):
# if x not in l:
# l.append(x)
# return np.array(l)


def norepxy(x, y, tol=1e-8):
    """REMOVES REPEATS IN (x,y) LISTS -- WITHIN tol EQUALS MATCH"""
    if type(x) == type(np.array([])):
        x = x.tolist()
        y = y.tolist()
    else:  # DON'T MODIFY ORIGINAL INPUT LISTS
        x = x[:]
        y = y[:]
    i = 0
    while i < len(x) - 1:
        j = i + 1
        while j < len(x):
            dist = np.hypot(x[i] - x[j], y[i] - y[j])
            if dist < tol:
                del x[j]
                del y[j]
            else:
                j += 1
        i += 1
    return x, y


def isseq(a):
    """TELLS YOU IF a IS SEQUENTIAL, LIKE [3, 4, 5, 6]"""
    return (np.alltrue(a == np.arange(len(a)) + a[0]))


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


def divisible(x, n):  # --DC
    return (old_div(x, float(n)) - old_div(x, n)) < (old_div(0.2, n))


def ndec(x, max=3):  # --DC
    """RETURNS # OF DECIMAL PLACES IN A NUMBER"""
    for n in range(max, 0, -1):
        if round(x, n) != round(x, n - 1):
            return n
    return 0  # IF ALL ELSE FAILS...  THERE'S NO DECIMALS


def qkfmt(x, max=8):
    n = ndec(x, max=max)
    if n:
        fmt = '%%.%df' % n
    else:
        fmt = '%d'
    return fmt % x


def interp(x, xdata, ydata, silent=0, extrap=0):  # NEW VERSION!
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


def interp1(x, xdata, ydata, silent=0):  # --DC
    """DETERMINES y AS LINEAR INTERPOLATION OF 2 NEAREST ydata"""
    SI = np.argsort(xdata)
    # NEW numpy's take IS ACTING FUNNY
    # NO DEFAULT AXIS, MUST BE SET EXPLICITLY TO 0
    xdata = xdata.take(SI, 0).astype(float).tolist()
    ydata = ydata.take(SI, 0).astype(float).tolist()
    if x > xdata[-1]:
        if not silent:
            print(x, 'OUT OF RANGE in interp in MLab_coe.py')
        return ydata[-1]
    elif x < xdata[0]:
        if not silent:
            print(x, 'OUT OF RANGE in interp in MLab_coe.py')
        return ydata[0]
    else:
        i = np.searchsorted(xdata, x)
        if xdata[i] == x:
            return ydata[i]
        else:
            [xlo, xhi] = xdata[i - 1:i + 1]
            [ylo, yhi] = ydata[i - 1:i + 1]
            return old_div(((x - xlo) * yhi + (xhi - x) * ylo), (xhi - xlo))


def interpn1(x, xdata, ydata, silent=0):  # --DC
    """DETERMINES y AS LINEAR INTERPOLATION OF 2 NEAREST ydata
    interpn TAKES AN ARRAY AS INPUT"""
    yout = []
    for x1 in x:
        yout.append(interp(x1, xdata, ydata, silent=silent))
    return np.array(yout)


def interp2(x, xdata, ydata):  # --DC
    """LINEAR INTERPOLATION/EXTRAPOLATION GIVEN TWO DATA POINTS"""
    m = old_div((ydata[1] - ydata[0]), (xdata[1] - xdata[0]))
    b = ydata[1] - m * xdata[1]
    y = m * x + b
    return y


def bilin(x, y, data, datax, datay):  # --DC
    """ x, y ARE COORDS OF INTEREST
    data IS 2x2 ARRAY CONTAINING NEARBY DATA
    datax, datay CONTAINS x & y COORDS OF NEARBY DATA"""
    lavg = old_div(((y - datay[0]) * data[1, 0] +
                    (datay[1] - y) * data[0, 0]), (datay[1] - datay[0]))
    ravg = old_div(((y - datay[0]) * data[1, 1] +
                    (datay[1] - y) * data[0, 1]), (datay[1] - datay[0]))
    return old_div(((x - datax[0]) * ravg + (datax[1] - x) * lavg), (datax[1] - datax[0]))


def bilin2(x, y, data):  # --DC
    """ x, y ARE COORDS OF INTEREST, IN FRAME OF data - THE ENTIRE ARRAY"""
    # SHOULD BE CHECKS FOR IF x, y ARE AT EDGE OF data
    ny, nx = data.shape
    ix = int(x)
    iy = int(y)
    if ix == nx - 1:
        x -= 1e-7
        ix -= 1
    if iy == ny - 1:
        y -= 1e-7
        iy -= 1
    if not ((0 <= ix < nx - 1) and (0 <= iy < ny - 1)):
        val = 0
    else:
        stamp = data[iy:iy + 2, ix:ix + 2]
        datax = [ix, ix + 1]
        datay = [iy, iy + 1]
        # print x, y, stamp, datax, datay
        val = bilin(x, y, stamp, datax, datay)
    return val


def rand(*args):
    """rand(d1,...,dn) returns a matrix of the given dimensions
    which is initialized to random numbers from a uniform distribution
    in the range [0,1).
    """
    return RandomArray.random(args)


def eye(N, M=None, k=0, dtype=None):
    """eye(N, M=N, k=0, dtype=None) returns a N-by-M matrix where the 
    k-th diagonal is all ones, and everything else is zeros.
    """
    if M == None:
        M = N
    if type(M) == type('d'):
        typecode = M
        M = N
    m = np.equal(np.subtract.outer(np.arange(N), np.arange(M)), -k)
    return np.asarray(m, dtype=typecode)


def tri(N, M=None, k=0, dtype=None):
    """tri(N, M=N, k=0, dtype=None) returns a N-by-M matrix where all
    the diagonals starting from lower left corner up to the k-th are all ones.
    """
    if M == None:
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


def fliplr(m):
    """fliplr(m) returns a 2-D matrix m with the rows preserved and
    columns flipped in the left/right direction.  Only works with 2-D
    arrays.
    """
    m = np.asarray(m)
    if len(m.shape) != 2:
        raise ValueError("Input must be 2-D.")
    return m[:, ::-1]


def flipud(m):
    """flipud(m) returns a 2-D matrix with the columns preserved and
    rows flipped in the up/down direction.  Only works with 2-D arrays.
    """
    m = np.asarray(m)
    if len(m.shape) != 2:
        raise ValueError("Input must be 2-D.")
    return m[::-1]


def rot90(m, k=1):
    """rot90(m,k=1) returns the matrix found by rotating m by k*90 degrees
    in the counterclockwise direction.
    """
    m = np.asarray(m)
    if len(m.shape) != 2:
        raise ValueError("Input must be 2-D.")
    k = k % 4
    if k == 0:
        return m
    elif k == 1:
        return np.transpose(fliplr(m))
    elif k == 2:
        return fliplr(flipud(m))
    elif k == 3:
        return fliplr(np.transpose(m))


def rot180(m):
    return rot90(m, 2)


def rot270(m):
    return rot90(m, 3)


def tril(m, k=0):
    """tril(m,k=0) returns the elements on and below the k-th diagonal of
    m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
    diagonal.
    """
    m = np.asarray(m)
    return tri(m.shape[0], m.shape[1], k=k, dtype=m.dtype.char) * m


def triu(m, k=0):
    """triu(m,k=0) returns the elements on and above the k-th diagonal of
    m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
    diagonal.
    """
    m = np.asarray(m)
    return (1 - tri(m.shape[0], m.shape[1], k - 1, m.dtype.char)) * m

# Data analysis

# Basic operations


def max(m):
    """max(m) returns the maximum along the first dimension of m.
    """
    return np.maximum.reduce(m)


def min(m):
    """min(m) returns the minimum along the first dimension of m.
    """
    return np.minimum.reduce(m)

# Actually from BASIS, but it fits in so naturally here...


def ptp(m):
    """ptp(m) returns the maximum - minimum along the first dimension of m.
    """
    return np.max(m) - np.min(m)


def mean1(m):
    """mean(m) returns the mean along the first dimension of m.  Note:  if m is
    an integer array, integer division will occur.
    """
    return old_div(np.add.reduce(m), len(m))


def mean(m, axis=0):
    """mean(m) returns the mean along the first dimension of m.  Note:  if m is
    an integer array, integer division will occur.
    """
    m = np.asarray(m)
    return old_div(np.add.reduce(m, axis=axis), m.shape[axis])


def meangeom(m):
    return np.product(m) ** (old_div(1., len(m)))

# sort is done in C but is done row-wise rather than column-wise


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


stddev = std


def meanstd(m):
    """meanstd(m) returns the mean and uncertainty = std / np.sqrt(N-1)
    """
    mu = mean(m)
    dmu = old_div(np.sqrt(np.add.reduce(pow(m - mu, 2))), (len(m) - 1.0))
    return mu, dmu


def avgstd2(m):  # --DC
    """avgstd2(m) returns the average & standard deviation along the first dimension of m.
    avgstd2 ELIMINATES OUTLIERS
    The result is unbiased meaning division by len(m)-1.
    """
    done = ''
    while not done:
        n = len(m)
        mu = mean(m)
        sig = old_div(np.sqrt(np.add.reduce(pow(m - mu, 2))), np.sqrt(n - 1.0))
        good = np.greater(m, mu - 3 * sig) * np.less(m, mu + 3 * sig)
        m = np.compress(good, m)
        done = sum(good) == n

    return [mu, old_div(np.sqrt(np.add.reduce(pow(m - mu, 2))), np.sqrt(len(m) - 1.0))]


def std2(m):  # --DC
    """std2(m) returns the standard deviation along the first dimension of m.
    std2 ELIMINATES OUTLIERS
    The result is unbiased meaning division by len(m)-1.
    """
    [a, s] = avgstd2(m)
    return s


stddev = std


def weightedavg(x, w):
    return old_div(sum(x * w), sum(w))


weightedmean = weightedavg


def thetaavgstd(theta):
    """CALCULATES THE AVERAGE & STANDARD DEVIATION IN A LIST (OR 1-D ARRAY) OF THETA (ANGLE) MEASUREMENTS
    RETURNS THE LIST [avg, std]
    CAN HANDLE ANY RANGE OF theta
    USES INCREASING WEIGHTED AVERAGES (2 POINTS AT A TIME)"""
    n = len(theta)
    if n == 1:
        return([theta[0], 999])
    else:
        thavg = theta[0]
        for i in range(1, n):
            th = theta[i]
            if thavg - th > np.pi:
                thavg = thavg - 2 * np.pi
            elif th - thavg > np.pi:
                th = th - 2 * np.pi
            thavg = old_div((i * thavg + th), (i + 1))
        for i in range(n):
            if theta[i] > thavg + np.pi:
                theta[i] = theta[i] - 2 * np.pi
        thstd = std(theta)
        return([thavg, thstd])


def clip2(m, m_min=None, m_max=None):
    if m_min == None:
        m_min = min(m)
    if m_max == None:
        m_max = max(m)
    return np.clip(m, m_min, m_max)


sum = np.add.reduce  # ALLOWS FOR AXIS TO BE INPUT --DC


def total(m):
    """RETURNS THE TOTAL OF THE ENTIRE ARRAY --DC"""
    return sum(np.ravel(m))


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


def cumprod(m):
    """cumprod(m) returns the cumulative product of the elments along the
    first dimension of m.
    """
    return np.multiply.accumulate(m)


def trapz(y, x=None):
    """trapz(y,x=None) integrates y = f(x) using the trapezoidal rule.
    """
    if x is None:
        d = 1
    else:
        d = diff(x)
    return sum(d * (y[1:] + y[0:-1]) / 2.0)


def cumtrapz(y, x=None, axis=0):
    """trapz(y,x=None) integrates y = f(x) using the trapezoidal rule. --DC"""
    if x == None:
        d = 1
    else:
        d = diff(x)
    if axis == 0:
        return cumsum(d * (y[1:] + y[0:-1]) / 2.0)
    elif axis == 1:
        return cumsum(d * (y[:, 1:] + y[:, 0:-1]) / 2.0, axis=1)
    else:
        print('YOUR VALUE OF axis = %d IS NO GOOD IN MLab_coe.cumtrapz' % axis)


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


def corrcoef(x, y=None):
    """The correlation coefficients
    """
    c = cov(x, y)
    d = diag(c)
    return old_div(c, np.sqrt(np.multiply.outer(d, d)))


def cov(m, y=None):
    m = np.asarray(m)
    mu = mean(m)
    if y != None:
        m = np.concatenate((m, y))
    sum_cov = 0.0
    for v in m:
        sum_cov = sum_cov + np.multiply.outer(v, v)
    return old_div((sum_cov - len(m) * np.multiply.outer(mu, mu)), (len(m) - 1.0))


def histogram(a, bins):
    n = np.searchsorted(np.sort(a), bins)
    n = np.concatenate([n, [len(a)]])
    return n[1:] - n[:-1]


def histo(a, da=1., amin=[], amax=[]):  # --DC
    """
    Histogram of 'a' defined on the bin grid 'bins'
       Usage: h=histogram(p,xp)
    """
    if amin == []:
        amin = min(a)
    if amax == []:
        amax = max(a)
    nnn = old_div((amax - amin), da)
    if np.less(nnn - int(nnn), 1e-4):
        amax = amax + da
    bins = np.arange(amin, amax + da, da)
    n = np.searchsorted(sort(a), bins)
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
