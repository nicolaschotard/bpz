from __future__ import division
from __future__ import print_function

from past.utils import old_div
import matplotlib
matplotlib.use('TkAgg')
import pylab

import os
import numpy as np
from bpz import coeio
from colormapdata import colormap_map
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

# Also can adjust real time:
# subplots_adjust(left=0.125)


def linelabelxy(x, y, x1, lab, **other):
    """"Label a line (x,y) with text lab at x1.
    Rotates the text to match angle of the line.
    I should probably take into account the figure margins"""
    xlog = pylab.gca().get_xaxis().get_scale() == 'log'
    ylog = pylab.gca().get_yaxis().get_scale() == 'log'
    if xlog:
        x = np.log10(x)
        x1 = np.log10(x1)
    if ylog:
        y = np.log10(y)
    y1 = np.interp(x1, x, y)
    xr = MLab_coe.p2p(np.log10(xlim()))
    yr = MLab_coe.p2p(ylim())
    x2 = x1 + old_div(xr, 100.)
    y2 = np.interp(x2, x, y)
    dx = old_div((x2 - x1), xr)
    dy = old_div((y2 - y1), yr)
    ang = MLab_coe.atanxy(dx, dy, 1)
    if xlog:
        x1 = 10 ** x1
    if ylog:
        y1 = 10 ** y1
    pylab.text(x1, y1, lab, rotation=ang, va='bottom', ha='center', **other)


def linelabel(p, x1, lab, **other):
    """"Label a line p with text lab at x1.
    Rotates the text to match angle of the line.
    I should probably take into account the figure margins"""
    x, y = p[0].get_data()
    linelabelxy(x, y, x1, lab, **other)


def plotclip(x, y, ymin, **other):
    """Plot leaving gaps where y < ymin"""
    n = len(x)
    i = 0

    if 'dy' in list(other.keys()):
        dy = other['dy']
        del(other['dy'])

    while i < n:
        # skip gap
        while y[i] < ymin:
            i += 1
            if i == n:
                break

        if i == n:
            break

        # gather data and plot
        ilo = i
        while y[i] >= ymin:
            i += 1
            if i == n:
                break

        pylab.plot(x[ilo:i], y[ilo:i] + dy, **other)


def color2gray(color):
    r, g, b = color
    return 0.3 * r + 0.59 * g + 0.11 * b


def gray2str(gray):
    return '%.2f' % gray


def colors2grays(colors):
    grays = list(map(color2gray, colors))
    grays = list(map(gray2str, grays))
    return grays


Ellipse = matplotlib.patches.Ellipse


def ellpatch1(x, y, w, h, ang, fc, ec, alpha=1, zorder=1, fill=1, lw=1):
    patch = Ellipse((x, y), w, h, ang * 180 / np.pi, fill=fill)
    patch.set_fc(fc)
    patch.set_ec(ec)
    patch.set_alpha(alpha)
    patch.set_zorder(zorder)
    patch.set_lw(lw)
    return patch


def polypatch1(x, y, fc, ec, alpha=1, zorder=1, lw=1):
    patch = pylab.Polygon(list(zip(x, y)))
    patch.set_fc(fc)
    patch.set_ec(ec)
    patch.set_alpha(alpha)
    patch.set_zorder(zorder)
    patch.set_lw(lw)
    return patch


def rectpatch1(x, y, dx, dy, fc, ec, alpha=1, zorder=1, lw=1):
    patch = pylab.Rectangle((x - dx, y - dy), 2 * dx, 2 * dy)
    patch.set_fc(fc)
    patch.set_ec(ec)
    patch.set_alpha(alpha)
    patch.set_zorder(zorder)
    patch.set_lw(lw)
    return patch

# /Library/Frameworks/Python.framework/Versions/Current/lib/python2.5/site-packages/matplotlib/contour.py


def contour_set(CS, alpha=None, zorder=None, color=None, sh=1):
    """contourf alpha doesn't work!
    CS = contourf(x, y, z)
    contour_set_alpha(CS, 0.5)"""
    for collection in CS.collections:
        if alpha != None:
            collection.set_alpha(alpha)
        if zorder != None:
            collection.set_zorder(zorder)
        if color != None:
            collection.set_color(color)
    if sh:
        pylab.show()


def contour_set_alpha(CS, alpha, sh=1):
    """contourf alpha doesn't work!
    CS = contourf(x, y, z)
    contour_set_alpha(CS, 0.5)"""
    for collection in CS.collections:
        collection.set_alpha(alpha)
    if sh:
        pylab.show()


def zlabel(s, vert=False, fontsize=None, x=0.88, **other):
    """Places label on colorbar"""
    if fontsize == None:
        fontsize = 14
    if vert:
        pylab.figtext(x, 0.5, s, rotation='vertical',
                va='center', fontsize=fontsize, **other)
    else:
        pylab.figtext(x, 0.5, s, fontsize=fontsize, **other)


def plotsort1(x, **other):
    i = old_div(np.arange(len(x)), (len(x) - 1.))
    pylab.plot(i, np.sort(x), **other)


def plotsort(x, norm=True, **other):
    n = len(x)
    i = np.arange(n)
    if norm:
        i = old_div(i, (n - 1.))
    pylab.plot(i, np.sort(x), **other)
    if not norm:
        xlim(0, n - 1)


def makelines(x1, y1, x2, y2):
    """(x1, y1), (x2, y2) LIST ALL CONNECTIONS
    CONNECT THE DOTS AND RETURN LISTS OF LISTS: x, y"""
    n = len(x1)

    i = 0
    j = i
    while (x1[i + 1] == x2[i]) and (y1[i + 1] == y2[i]):
        i += 1
        if i > n - 2:
            break

    x = x1[j:i + 1].tolist()
    y = y1[j:i + 1].tolist()
    x.append(x2[i])
    y.append(y2[i])

    xx = [x]
    yy = [y]

    while i < n - 2:
        i += 1
        j = i
        while (x1[i + 1] == x2[i]) and (y1[i + 1] == y2[i]):
            i += 1
            if i > n - 2:
                break

        x = x1[j:i + 1].tolist()
        y = y1[j:i + 1].tolist()
        x.append(x2[i])
        y.append(y2[i])
        #
        xx.append(x)
        yy.append(y)

    return xx, yy

#################################
# ZOOM IN ON DATASET
# SEE ~/glens/h0limis/results4ae.py


def xyrcut(xr, yr, i, fac=0.8):
    dx = MLab_coe.p2p(xr)
    dy = MLab_coe.p2p(yr)
    if i == 0:    # RIGHT
        xr = xr[0], xr[0] + fac * dx
    elif i == 1:  # LEFT
        xr = xr[1] - fac * dx, xr[1]
    elif i == 2:  # TOP
        yr = yr[0], yr[0] + fac * dy
    elif i == 3:  # BOTTOM
        yr = yr[1] - fac * dy, yr[1]
    return xr, yr


def catxyrcut(fac, ct, xr, yr, i, justn=0):
    if fac <= 0:
        return np.inf
    elif fac >= 1:
        return 0

    xr2, yr2 = xyrcut(xr, yr, i, 1 - fac)
    ct2 = ct.between(xr2[0], 'x', xr2[1])
    ct2 = ct2.between(yr2[0], 'y', yr2[1])
    # print ct.len()
    # print ct2.len()

    n = old_div((ct.len() - ct2.len()), float(ct.len()))
    n = MLab_coe.divsafe(fac, n)
    if justn:
        return n
    else:
        return n, xr2, yr2, ct2


def funcy(x, func, args=(), y=0):
    # print x, func(x, *args), 'result'
    return abs(func(x, *args) - y)


def zoom(x, y, nfac=30, fac=0.2, margin=0.02):
    cat = coeio.VarsClass()
    cat.add('x', x)
    cat.add('y', y)
    cat.updatedata()

    xr = MLab_coe.minmax(cat.x)
    yr = MLab_coe.minmax(cat.y)

    for i in range(4):
        cat2 = cat
        n = 1e30
        while n > nfac:
            cat = cat2
            xr = MLab_coe.minmax(cat.x)
            yr = MLab_coe.minmax(cat.y)
            n, xr2, yr2, cat2 = catxyrcut(fac, cat, xr, yr, i)
            print(i, n, cat2.len())

    xlim(MLab_coe.prange(xr, margin=margin))
    ylim(MLab_coe.prange(yr, margin=margin))

#################################


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

# log x and/or y axes with nice tick labels
# formatter = FuncFormatter(log_10_product)
# FOR FURTHER USE INSTRUCTIONS, SEE e.g.,
#  ~/LensPerfect/A1689/analysis/NFWfitWSplot.py
# http://www.thescripts.com/forum/thread462268.html


def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    # return '%1i' % (x)
    ndec1 = MLab_coe.ndec(x)
    if ndec1 == 0:
        format = '%d'
    else:
        format = '%%.%df' % ndec1
    # print format, x
    return format % (x)


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


def ploterrorbars1(x, y, dy, ymax=None, color='k', xfac=1, **other):
    if ymax == None:
        ymin = y - dy
        ymax = y + dy
    else:
        ymin = dy
        ymax = ymax

    dx = 0.005 * xfac * (xlim()[1] - xlim()[0])
    itemp = pylab.isinteractive()
    xtemp = xlim()
    ytemp = ylim()
    pylab.ioff()
    for i in range(len(x)):
        pylab.plot([x[i], x[i]], [ymin[i], ymax[i]], color=color, **other)
        pylab.plot([x[i] - dx, x[i] + dx], [ymax[i], ymax[i]], color=color, **other)
        pylab.plot([x[i] - dx, x[i] + dx], [ymin[i], ymin[i]], color=color, **other)

    if itemp:
        pylab.ion()
        pylab.show()

    xlim(xtemp[0], xtemp[1])
    ylim(ytemp[0], ytemp[1])


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


def reaspect(xr, yr, aspect=old_div(5, 7.)):
    dx = xr[1] - xr[0]
    dy = yr[1] - yr[0]

    if old_div(dy, dx) < aspect:
        yr = np.mean(yr) + dx * aspect * (np.arange(2) - 0.5)
    else:
        xr = np.mean(xr) + dy / aspect * (np.arange(2) - 0.5)

    return xr, yr


def setaspect(xr=None, yr=None, aspect=old_div(5, 7.), ret=0, sh=0):
    if xr == None:
        xr = xlim()
    if yr == None:
        yr = ylim()

    xr, yr = reaspect(xr, yr)

    xlim(xr[0], xr[1])
    ylim(yr[0], yr[1])
    if sh:
        pylab.show()

    if ret:
        return xr, yr


def setaspectsquare(shown=True):
    py2 = pylab.rcParams.get('figure.subplot.top')
    py1 = pylab.rcParams.get('figure.subplot.bottom')
    px2 = pylab.rcParams.get('figure.subplot.right')
    px1 = pylab.rcParams.get('figure.subplot.left')

    # http://matplotlib.sourceforge.net/api/figure_api.html
    # gcf().get_height()
    # gcf().get_figwidth()
    fx, fy = pylab.gcf().get_size_inches()  # - array([0.175, 0.175])  # white border
    # The 0.175 white border only gets added after the figure is shown
    if shown:
        fx = fx - 0.175
        fy = fy - 0.175

    pdx = fx * (px2 - px1)
    pdy = fy * (py2 - py1)
    aspect = old_div(pdy, pdx)
    setaspect(aspect=aspect)


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


def len0(x):
    try:
        n = len(x)
    except:
        n = 0
    return n

def killplot():
    pylab.plot([1])
    pylab.title('CLOSE THIS WINDOW!')
    pylab.show()
    pylab.ioff()

def closefig(num=None):
    if num == None:
        pylab.close()
    else:
        pylab.figure(num)
        pylab.close()


def smallfig(num=1, fac=2, reopen=0):
    if reopen:
        closefig(num)

    pylab.figure(num, figsize=(old_div(8, fac), old_div(6, fac)))


def showarr(a, showcolorbar=1, nan=None, valrange=[None, None], sq=0,
            cmap='jet', x=None, y=None, cl=1, sh=1):
    if valrange[0] or valrange[1]:
        if valrange[0] == None:
            valrange[0] = min(a)
        if valrange[1] == None:
            valrange[1] = max(a)
        a = MLab_coe.clip2(a, valrange[0], valrange[1])
    if nan != None:
        a = np.where(np.isnan(a), nan, a)
    if cl:
        pylab.clf()
    pylab.ioff()
    if x != None and y != None:
        xlo, xhi = MLab_coe.minmax(x)
        ylo, yhi = MLab_coe.minmax(y)
        dx = old_div((xhi - xlo), (len(x) - 1.))
        dy = old_div((yhi - ylo), (len(y) - 1.))
        extent = (xlo - old_div(dx, 2.), xhi + old_div(dx, 2.),
                  ylo - old_div(dy, 2.), yhi + old_div(dy, 2.))
    else:
        extent = None
    cmap = pylab.cm.get_cmap(cmap)
    aspect = ['auto', 1][sq]  # 'preserve'
    im = pylab.imshow(a, origin='lower', interpolation='nearest', cmap=cmap,
                aspect=aspect, extent=extent)
    if showcolorbar:
        pylab.colorbar()
    if sh:
        pylab.show()
        pylab.ion()
    return im


def showxyz(x, y, z, **other):
    showarr(z, x=x, y=y, **other)


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


def retick1(x, axs, ntx=4, ndec=1, ytx=None, relim=True):
    N = len(x) - 1
    lo, hi = MLab_coe.minmax(x)

    if ytx == None:
        ylocs = np.arange(0, N + .001, old_div(N, float(ntx)))
        ytx = ylocs / float(N) * (hi - lo) + lo
    else:
        ylocs = (ytx - lo) * N / float(hi - lo)

    ytxs = []
    for ytick in ytx:
        format = '%%.%df' % ndec
        ytxs.append(format % ytick)
        #ytxs.append('%.1f' % ytick)

    if axs == 'x':
        p = pylab.xticks(ylocs, ytxs)
        if relim:
            p = xlim(-0.5, N + 0.5)
    elif axs == 'y':
        p = pylab.yticks(ylocs, ytxs)
        if relim:
            p = ylim(-0.5, N + 0.5)
    else:
        print('Sorry, which axis? --retick1')

    pylab.show()


def retick2(xlo, xhi, Nx, ylo, yhi, Ny, Nxtx=4, Nytx=4, ndec=1):
    Nx = Nx - 1.
    Ny = Ny - 1.

    xlocs = np.arange(0, Nx + .001, old_div(Nx, float(Nxtx)))
    xtx = xlocs / float(Nx) * (xhi - xlo) + xlo
    xtxs = []
    for xtick in xtx:
        format = '%%.%df' % ndec
        xtxs.append(format % xtick)

    ylocs = np.arange(0, Ny + .001, old_div(Ny, float(Nytx)))
    ytx = ylocs / float(Ny) * (yhi - ylo) + ylo
    ytxs = []
    for ytick in ytx:
        format = '%%.%df' % ndec
        ytxs.append(format % ytick)

    p = pylab.xticks(xlocs, xtxs)
    p = pylab.yticks(ylocs, ytxs)

    # xlim(0,Nx)
    # ylim(0,Ny)
    xlim(-0.5, Nx + 0.5)
    ylim(-0.5, Ny + 0.5)
    pylab.show()


def retick3(x, y, Nxtx=4, Nytx=4, ndec=1):
    xlo, xhi = MLab_coe.minmax(x)
    ylo, yhi = MLab_coe.minmax(y)
    Nx = len(x)
    Ny = len(y)
    retick2(xlo, xhi, Nx, ylo, yhi, Ny, Nxtx=Nxtx, Nytx=Nytx, ndec=ndec)


def retickxmult(d):
    lo, hi = MLab_coe.minmax(xlim())
    tx = MLab_coe.multiples(lo, hi, d)
    pylab.xticks(tx)


def retickymult(d):
    lo, hi = MLab_coe.minmax(ylim())
    tx = MLab_coe.multiples(lo, hi, d)
    pylab.yticks(tx)


def retickxlogmult(d=[1, 2, 5]):
    formatter = pylab.FuncFormatter(log_10_product)
    pylab.gca().xaxis.set_major_formatter(formatter)
    pylab.gca().xaxis.set_major_locator(pylab.LogLocator(10, d))


def retickylogmult(d=[1, 2, 5]):
    formatter = pylab.FuncFormatter(log_10_product)
    pylab.gca().yaxis.set_major_formatter(formatter)
    pylab.gca().yaxis.set_major_locator(pylab.LogLocator(10, d))


def reticklog1(x, axs, relim=True):
    """Automatcially places tickmarks at log intervals (0.001, 0.01, etc.)
    x contains data for either the x or y axis
    axs = 'x' or 'y' (which axis?)"""
    eps = 1e-7
    lo = min(np.log10(x))
    hi = max(np.log10(x))
    #lo = floor(lo+eps)
    #tx = np.arange(lo,hi+eps,1)
    tx = MLab_coe.multiples(lo, hi)
    txs = []
    for logx in tx:
        n = 10 ** logx
        tx1 = '%g' % n
        tx1 = tx1.replace('-0', '-')
        tx1 = tx1.replace('+0', '+')
        txs.append(tx1)

    ntx = len(tx)
    nx = len(x) - 1
    #locs = mgrid[0:nx:ntx*1j]
    locs = np.interp(tx, np.array([lo, hi]), np.array([0, nx]))
    if axs == 'x':
        p = pylab.xticks(locs, txs)
        if relim:
            p = xlim(-0.5, nx + 0.5)
    elif axs == 'y':
        p = pylab.yticks(locs, txs)
        if relim:
            p = ylim(-0.5, nx + 0.5)
    else:
        print('Sorry, which axis? --reticklog1')


def reticklog(x, y, relim=True):
    reticklog1(x, 'x', relim=relim)
    reticklog1(y, 'y', relim=relim)


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


def retickx2(fmt=mapfmt, d=1):
    lo, hi = MLab_coe.minmax(xlim())
    tx = MLab_coe.multiples(lo, hi, d)
    ts = list(map(fmt, tx))
    pylab.xticks(tx, ts)


def reticky2(fmt=mapfmt, d=1):
    lo, hi = MLab_coe.minmax(ylim())
    tx = MLab_coe.multiples(lo, hi, d)
    ts = list(map(fmt, tx))
    pylab.yticks(tx, ts)


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


def sqrange(x, y, margin=None, set=1):
    xmax = max(abs(np.array(x)))
    ymax = max(abs(np.array(y)))
    xymax = max([xmax, ymax])
    xyr = -xymax, xymax
    if margin != None:
        xyr = MLab_coe.prange(xyr)
    if set:
        xlim(xyr)
        ylim(xyr)
    return xyr


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


def vectorplot(vx, vy, x=[], y=[], xyfactor=[], heads=1, clear=1, color='k', **other):
    # print 'vx, vy', MLab_coe.minmax(np.ravel(vx)), MLab_coe.minmax(np.ravel(vy))
    nmax = 10
    if clear:
        pylab.clf()
    pylab.ioff()  # DON'T UPDATE PLOT WITH EACH ADDITIION (WAIT UNTIL END...)
    # IF TOO MANY VECTORS, SAMPLE:
    if len(vx.shape) == 2:
        ny, nx = vx.shape
        if (ny > nmax) or (nx > nmax):
            yall, xall = np.indices((ny, nx))
            nyred = min([ny, nmax])
            nxred = min([nx, nmax])
            dy = old_div(ny, float(nyred))
            dx = old_div(nx, float(nxred))
            vxred = np.zeros((nyred, nxred), float)
            vyred = np.zeros((nyred, nxred), float)
            for iy in range(nyred):
                y = int(iy * dy)
                for ix in range(nxred):
                    x = int(ix * dx)
                    vxred[iy, ix] = vx[y, x]
                    vyred[iy, ix] = vy[y, x]
                    #print (iy,ix), (y,x)
                    #print (vy[y,x], vx[y,x]), (vyred[iy,ix], vxred[iy,ix])
            vy = vyred
            vx = vxred
        y, x = np.indices(vx.shape)
        vx = np.ravel(vx)
        vy = np.ravel(vy)
        x = np.ravel(x)
        y = np.ravel(y)
    # print 'RED vx, vy', MLab_coe.minmax(np.ravel(vx)), MLab_coe.minmax(np.ravel(vy))
    try:
        xfactor, yfactor = xyfactor
    except:
        if xyfactor != []:
            xfactor = yfactor = xyfactor
        else:
            xfactor = 1.
            yfactor = 1.
    #ny, nx = vx.shape
    vr = np.hypot(vx, vy)
    # print max(np.ravel(vx))
    # print max(np.ravel(vy))
    #maxvr = max(np.ravel(vr))
    maxvr = max(vr)
    vx = vx / maxvr * 0.1 * ptp(x) * xfactor
    vy = vy / maxvr * 0.1 * ptp(y) * yfactor
    # print maxvr, xfactor, yfactor
    for i in range(len(vx)):
        xo = x[i]
        yo = y[i]
        dx = vx[i]
        dy = vy[i]
        # print dx, dy
        #pause('TO PLOT...')
        dr = np.hypot(dx, dy)
        # IF IT'S TOO SMALL, DRAW A DOT
        if dr < old_div(xfactor, 100.):
            pylab.plot([xo], [yo], 'o', markerfacecolor=color)
            continue
        # OTHERWISE, DRAW A STICK
        pylab.plot([xo - dx, xo + dx], [yo - dy, yo + dy], color=color)
        if heads:
            # NOW FOR THE HEAD OF THE ARROW
            hx = -dx + -dy
            hy = dx + -dy
            #hr = hypot(hx, hy)
            #hx = 0.1 * hx / hr * xfactor
            #hy = 0.1 * hy / hr * yfactor
            hx = 0.2 * hx
            hy = 0.2 * hy
            pylab.plot([xo + dx, xo + dx + hx], [yo + dy, yo + dy + hy], color=color)
            pylab.plot([xo + dx, xo + dx - hy], [yo + dy, yo + dy + hx], color=color)
    #
    pylab.show()  # NOW SHOW THE PLOT
    pylab.ion()  # AND TURN INTERACTIVE MODE BACK ON


def atobplotold(xa, ya, xb, yb, color='k', linetype='arrow', showplot=1, hxfac=1, hyfac=1, **other):
    """DRAWS LINES FROM a TO b"""
    n = len0(xa)
    nb = len0(xb)
    if not n and not nb:
        n = 1
        xa = [xa]
        ya = [ya]
        xb = [xb]
        yb = [yb]
    elif not n:
        n = nb
        xa = [xa] * n
        ya = [ya] * n
    elif not nb:
        xb = [xb] * n
        yb = [yb] * n
    isint = pylab.isinteractive()
    pylab.ioff()
    for i in range(n):
        pylab.plot([xa[i], xb[i]], [ya[i], yb[i]], color=color, **other)
        if linetype == 'arrow':
            # NOW FOR THE HEAD OF THE ARROW
            dx = xb[i] - xa[i]
            dy = yb[i] - ya[i]
            hx = -dx + -dy
            hy = dx + -dy
            #hr = hypot(hx, hy)
            #hx = 0.1 * hx / hr * xfactor
            #hy = 0.1 * hy / hr * yfactor
            hx = 0.1 * hx
            hy = 0.1 * hy
            pylab.plot([xb[i], xb[i] + hx * hyfac],
                 [yb[i], yb[i] + hy * hyfac], color=color, **other)
            pylab.plot([xb[i], xb[i] - hy * hyfac],
                 [yb[i], yb[i] + hx * hyfac], color=color, **other)
        if linetype == 'xo':
            pylab.plot(xa, ya, 'x', markerfacecolor=color,
                 markeredgecolor=color, **other)
            pylab.plot(xb, yb, 'o', markerfacecolor=color, **other)
    if isint and showplot:
        pylab.ion()
        pylab.show()


def atobplot(xa, ya, xb, yb, color='k', linetype='arrow', showplot=1, hxfac=1, hyfac=1, **other):
    """DRAWS LINES FROM a TO b"""
    n = len0(xa)
    nb = len0(xb)
    if not n and not nb:
        n = 1
        xa = [xa]
        ya = [ya]
        xb = [xb]
        yb = [yb]
    elif not n:
        n = nb
        xa = [xa] * n
        ya = [ya] * n
    elif not nb:
        xb = [xb] * n
        yb = [yb] * n
    isint = pylab.isinteractive()
    pylab.ioff()
    for i in range(n):
        pylab.plot([xa[i], xb[i]], [ya[i], yb[i]], color=color, **other)
        if linetype == 'arrow':
            # NOW FOR THE HEAD OF THE ARROW
            dx = xb[i] - xa[i]
            dy = yb[i] - ya[i]
            hx = -dx + -dy
            hy = dx + -dy
            hx = 0.1 * hx
            hy = 0.1 * hy
            #print [xb[i], xb[i]+hx*hxfac], [yb[i], yb[i]+hy*hyfac]
            #print [xb[i], xb[i]-hy*hxfac], [yb[i], yb[i]+hx*hyfac]
            pylab.plot([xb[i], xb[i] + hx * hxfac],
                 [yb[i], yb[i] + hy * hyfac], color=color, **other)
            pylab.plot([xb[i], xb[i] - hy * hxfac],
                 [yb[i], yb[i] + hx * hyfac], color=color, **other)
        if linetype == 'xo':
            pylab.plot(xa, ya, 'x', mfc=color, mec=color, **other)
            pylab.plot(xb, yb, 'o', mfc=color, **other)
    if isint and showplot:
        pylab.ion()
        pylab.show()


def colormaprgb1(val, valrange=[0., 1.], cmap='jet', silent=0):
    if valrange != [0., 1.]:
        lo = float(valrange[0])
        hi = float(valrange[1])
        val = old_div((val - lo), (hi - lo))

    try:
        n = len(val)
    except:  # SINGLE VALUE
        val = np.array([val])

    cmapa = colormap_map[cmap]

    xa = old_div(np.arange(len(cmapa)), float(len(cmapa) - 1))
    ra, ga, ba = np.transpose(cmapa)

    r = MLab_coe.interpn(val, xa, ra, silent)
    g = MLab_coe.interpn(val, xa, ga, silent)
    b = MLab_coe.interpn(val, xa, ba, silent)
    rgb = np.ravel(np.array([r, g, b]))
    return rgb


def colormaprgb(val, valrange=[0., 1.], cmap='jet', silent=0):
    if valrange != [0., 1.]:
        lo = float(valrange[0])
        hi = float(valrange[1])
        val = old_div((val - lo), (hi - lo))

    try:
        n = len(val)
    except:  # SINGLE VALUE
        val = np.array([val])

    if cmap in list(colormap_map.keys()):
        cmapa = colormap_map[cmap]

        xa = old_div(np.arange(len(cmapa)), float(len(cmapa) - 1))
        ra, ga, ba = np.transpose(cmapa)
    else:
        cmapd = pylab.cm.datad[cmap]
        xa = np.array(cmapd['blue'])[:, 0]
        ra = np.array(cmapd['red'])[:, 1]
        ga = np.array(cmapd['green'])[:, 1]
        ba = np.array(cmapd['blue'])[:, 1]

    r = MLab_coe.interpn(val, xa, ra, silent)
    g = MLab_coe.interpn(val, xa, ga, silent)
    b = MLab_coe.interpn(val, xa, ba, silent)
    rgb = np.ravel(np.array([r, g, b]))
    return rgb

# colormap IS ALREADY BUILT IN, BUT NAMED scatter
# AND THIS WAY, YOU CAN ADD A colorbar()!


def colormap(x, y, z, showcolorbar=1, ticks=None, s=30, edgecolors="None", **other):
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)
    pylab.scatter(x, y, c=z, s=s, edgecolors=edgecolors, **other)
    if showcolorbar:
        pylab.colorbar(ticks=ticks)

# SEE colormap.py


def colormap_obsolete(x, y, z, valrange=[None, None], cmap='jet', markersize=5):
    pylab.ioff()

    if valrange[0] == None:
        valrange[0] = min(z)
    if valrange[1] == None:
        valrange[1] = max(z)
    n = len(x)

    for i in range(n):
        if not (MLab_coe.isNaN(z[i]) or (valrange[0] == valrange[1])):
            pylab.plot([x[i]], [y[i]], 'o', markerfacecolor=colormaprgb(
                z[i], valrange=valrange), markersize=markersize)

    pylab.colorbar()
    pylab.ion()
    pylab.show()


def test():
    x = np.arange(10)
    y = np.arange(10)
    z = np.arange(10)
    colormap(x, y, z)

# test()

# THEY MUST'VE WRITTEN THIS, BUT I CAN'T FIND IT!


def circle(xc, yc, r, n=100, **other):
    ang = np.arange(n + 1) / float(n) * 2 * np.pi
    x = xc + r * np.cos(ang)
    y = yc + r * np.sin(ang)
    pylab.plot(x, y, **other)


def circles(xc, yc, r, n=100, **other):
    if MLab_coe.singlevalue(xc):
        xc = xc * np.ones(len(r))
    if MLab_coe.singlevalue(yc):
        yc = yc * np.ones(len(r))
    for i in range(len(xc)):
        circle(xc[i], yc[i], r[i], n=n, **other)

# COULD ALSO USE matplotlib.patches.Ellipse


def ellipse(xc, yc, a, b, ang=0, n=100, **other):
    t = np.arange(n + 1) / float(n) * 2 * np.pi
    x = a * np.cos(t)
    y = b * np.sin(t)
    if ang:
        x, y = MLab_coe.rotdeg(x, y, ang)
    x = x + xc
    y = y + yc
    pylab.plot(x, y, **other)


def ellipses(xc, yc, a, b, ang=None, n=100, **other):
    if ang == None:
        ang = np.zeros(len(a))
    for i in range(len(xc)):
        ellipse(xc[i], yc[i], a[i], b[i], ang[i], n=n, **other)


def texts(x, y, z, format='%d', showit=True, **other):
    isint = pylab.isinteractive()
    pylab.ioff()
    if MLab_coe.singlevalue(z):
        z = z * np.ones(len(x))
    for i in range(len(x)):
        t = pylab.text(x[i], y[i], format % z[i], **other)
    if isint and showit:
        pylab.ion()
        pylab.show()