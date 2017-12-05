from __future__ import division
from __future__ import print_function

from past.utils import old_div
import matplotlib
matplotlib.use('TkAgg')
import pylab

import os
import numpy as np
from bpz import coeio
from bpz import MLab_coe
from bpz import coetools

# p.14 postscript, native gtk and native wx do not support alpha or antialiasing.
# You can create an arbitrary number of axes images inside a single axes, and these will be
# composed via alpha blending. However, if you want to blend several images, you must make sure
# that the hold state is True and that the alpha of the layered images is less than 1.0; if
# alpha=1.0 then the image on top will totally obscure the images below. Because the image blending
# is done using antigrain (regardless of your backend choice), you can blend images even on backends
# which don't support alpha (eg, postscript). This is because the alpha blending is done in the
# frontend and the blended image is transferred directly to the backend as an RGB pixel array.
# See Recipe 9.4.2 for an example of how to layer images.


# Now handled in thick():
fontsize = 18  # 20
pparams = {'axes.labelsize': fontsize,
           'font.size': fontsize,
           'legend.fontsize': fontsize - 4,
           'figure.subplot.left': 0.125,
           'figure.subplot.bottom': 0.125,
           'figure.subplot.top': 0.95,
           'figure.subplot.right': 0.95,
           'lines.linewidth': 0.875,
           }
# pylab.rcParams.update(pparams)


def thick(fontsize=18, labsize=None, legsize=None, left=0.125, bottom=0.125, top=0.95, right=0.95, lw=2):
    if labsize == None:
        labsize = fontsize
    if legsize == None:
        legsize = fontsize - 4
    pparams = {
        'axes.labelsize': labsize,
        'font.size': fontsize,
        'legend.fontsize': legsize,
        'figure.subplot.left': left,
        'figure.subplot.bottom': bottom,
        'figure.subplot.top': top,
        'figure.subplot.right': right,
        'lines.linewidth': lw,
    }
    pylab.rcParams.update(pparams)


pparams1 = {'legend.numpoints': 1}
pylab.rcParams.update(pparams1)


def hline(v=0, c='k', ls='-', **other):
    """HORIZONTAL LINE THAT ALWAYS SPANS THE AXES"""
    return pylab.axhline(v, c=c, ls=ls, **other)


yline = hline


def vline(v=0, c='k', ls='-', **other):
    """VERTICAL LINE THAT ALWAYS SPANS THE AXES"""
    return pylab.axvline(v, c=c, ls=ls, **other)


xline = vline


def axlines(x=0, y=0, c='k', ls='-', **other):
    """VERTICAL LINE THAT ALWAYS SPANS THE AXES"""
    pylab.axvline(x, c=c, ls=ls, **other)
    pylab.axhline(y, c=c, ls=ls, **other)

# from MLab_coe:


def ticks(tx, ax='xy', fmt='%g'):
    def mapfmt(x):
        return fmt % x

    ts = list(map(mapfmt, tx))
    if 'x' in ax:
        pylab.xticks(tx, ts)
    if 'y' in ax:
        pylab.yticks(tx, ts)


def savepdf(figname, saveeps=1):
    if figname[:-4] == '.pdf':
        figname = figname[:-4]
    pylab.savefig(figname + '.eps')
    #os.system('epstopdf %s.eps' % figname)
    os.system('pstopdf %s.eps' % figname)
    if not saveeps:
        os.remove(figname + '.eps')


def savepngpdf(figname, saveeps=1):
    if len(figname) > 4:
        if figname[-4] == '.':
            figname = figname[:-4]
    pylab.savefig(figname + '.png')
    savepdf(figname, saveeps=saveeps)


def savepng(figname):
    if len(figname) > 4:
        if figname[-4] == '.':
            figname = figname[:-4]
    pylab.savefig(figname + '.png')


def ploterrorbars(x, y, dy, ymax=None, color='k', xfac=1, ax=None, xlog=False, **other):
    if ax == None:
        ax = pylab.gca()

    if ymax == None:
        ymin = y - dy
        ymax = y + dy
    else:
        ymin = dy
        ymax = ymax

    if xlog:
        dxlog = 0.005 * xfac * (np.log10(xlim()[1]) - np.log10(xlim()[0]))
        print('dxlog', dxlog)
        dx = dxlog * x / np.log10(np.e)
    else:
        dx = 0.005 * xfac * (xlim()[1] - xlim()[0])
    itemp = pylab.isinteractive()
    xtemp = xlim()
    ytemp = ylim()
    pylab.ioff()
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [ymin[i], ymax[i]], color=color, **other)
        ax.plot([x[i] - dx, x[i] + dx],
                [ymax[i], ymax[i]], color=color, **other)
        ax.plot([x[i] - dx, x[i] + dx],
                [ymin[i], ymin[i]], color=color, **other)

    if itemp:
        pylab.ion()
        pylab.show()

    ax.set_xlim(xtemp[0], xtemp[1])
    ax.set_ylim(ytemp[0], ytemp[1])


#################################


def xlim(lo=None, hi=None):
    if lo == None and hi == None:
        return pylab.xlim()
    else:
        if MLab_coe.singlevalue(lo):
            lo1, hi1 = pylab.xlim()
            if lo == None:
                lo = lo1
            if hi == None:
                hi = hi1
        else:
            lo, hi = lo
        pylab.xlim(lo, hi)


def ylim(lo=None, hi=None):
    if lo == None and hi == None:
        return pylab.ylim()
    else:
        if MLab_coe.singlevalue(lo):
            lo1, hi1 = pylab.ylim()
            if lo == None:
                lo = lo1
            if hi == None:
                hi = hi1
        else:
            lo, hi = lo
        pylab.ylim(lo, hi)

#################################


def rectangle(lolimits, hilimits, fillit=0, **other):
    [xmin, ymin] = lolimits
    [xmax, ymax] = hilimits
    if not fillit:
        return pylab.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], **other)
    else:
        if 'color' in list(other.keys()):
            color = other['color']
            del other['color']
            color = coetools.color2hex(color)
            return pylab.fill([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color, **other)
        else:
            return pylab.fill([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], **other)

# FIX THE AXES ON A PLOT OF AN ARRAY


def retick(lo, hi, N, ndec=1, ytx=None, sh=1):
    N = N - 1.

    if ytx == None:
        ylocs = np.arange(0, N + .001, old_div(N, 4.))
        ytx = ylocs / float(N) * (hi - lo) + lo
    else:
        ylocs = (ytx - lo) * N / float(hi - lo)

    ytxs = []
    for ytick in ytx:
        format = '%%.%df' % ndec
        ytxs.append(format % ytick)
        #ytxs.append('%.1f' % ytick)

    pylab.yticks(ylocs, ytxs)
    pylab.xticks(ylocs, ytxs)

    xlim(0, N)
    ylim(0, N)
    if sh:
        pylab.show()

def mapfmt(x):
    return '%g' % x


def reticky(fmt=mapfmt):
    ytx, yts = pylab.yticks()
    yts = list(map(fmt, ytx))
    pylab.yticks(ytx, yts)


def retickx(fmt=mapfmt):
    xtx, xts = pylab.xticks()
    xts = list(map(fmt, xtx))
    pylab.xticks(xtx, xts)


def fillbetween(x1, y1, x2, y2, **other):
    # MAKE SURE IT'S NOT A LIST, THEN IN CASE IT'S A numpy ARRAY, CONVERT TO LIST, THEN CONVERT TO numarray ARRAY
    if type(x1) != list:
        x1 = x1.tolist()
    if type(y1) != list:
        y1 = y1.tolist()
    if type(x2) != list:
        x2 = x2.tolist()
    if type(y2) != list:
        y2 = y2.tolist()
    x = x1[:]
    x[len(x):] = x2[::-1]
    y = y1[:]
    y[len(y):] = y2[::-1]
    return pylab.fill(x, y, **other)


def prangelog(x, xinclude=None, margin=0.05):
    """RETURNS GOOD RANGE FOR DATA x TO BE PLOTTED IN.
    xinclude = VALUE YOU WANT TO BE INCLUDED IN RANGE.
    margin = FRACTIONAL MARGIN ON EITHER SIDE OF DATA."""
    xmin = min(x)
    xmax = max(x)
    if xinclude != None:
        xmin = min([xmin, xinclude])
        xmax = max([xmax, xinclude])

    fac = old_div(xmax, xmin)
    xmin = old_div(xmin, (fac ** margin))
    xmax = xmax * (fac ** margin)

    return [xmin, xmax]
