from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import numpy as np
sys.float_output_precision = 5  # PRINTING ARRAYS: # OF DECIMALS
numerix = os.environ.get('NUMERIX', '')

pwd = os.getcwd
die = sys.exit
home = os.environ.get('HOME', '')


def color1to255(color):
    # CONVERT TO 0-255 SCALE
    return tuple((np.array(color) * 255. + 0.49).astype(int).tolist())


def color2hex(color):
    if 0:  # 0 < max(color) <= 1:  # 0-1 SCALE
        # BUT EVERY ONCE IN A WHILE, YOU'LL GET A (0,0,1) OUT OF 255...
        color = color1to255(color)
    colorhex = '#'
    for val in color:
        h = hex(val)[2:]
        if len(h) == 1:
            h = '0' + h
        colorhex += h
    return colorhex


def singlevalue(x):
    """IS x A SINGLE VALUE?  (AS OPPOSED TO AN ARRAY OR LIST)"""
    # return type(x) in [float, int]  THERE ARE MORE TYPECODES IN Numpy
    # THERE ARE MORE TYPECODES IN Numpy
    return not isinstance(x, (list, np.ndarray))


def str2num(strg, rf=0):
    """CONVERTS A STRING TO A NUMBER (INT OR FLOAT) IF POSSIBLE
    ALSO RETURNS FORMAT IF rf=1"""
    try:
        num = int(strg)
        format = 'd'
    except:
        try:
            num = float(strg)
            format = 'f'
        except:
            if not strg.strip():
                num = None
                format = ''
            else:
                num = strg
                format = 's'
    if rf:
        return (num, format)
    else:
        return num


def minmax(x, range=None):
    if range:
        lo, hi = range
        good = between(lo, x, hi)
        x = np.compress(good, x)
    return min(x), max(x)

#############################################################################
# ARRAYS
#
# PYTHON USES BACKWARDS NOTATION: a[row,column] OR a[iy,ix] OR a[iy][ix]
# NEED TO MAKE size GLOBAL (I THINK) OTHERWISE, YOU CAN'T CHANGE IT!
# COULD HAVE ALSO USED get_data IN ~txitxo/Python/useful.py


def FltArr(n0, n1):
    """MAKES A 2-D FLOAT ARRAY"""
    #a = np.ones([n0,n1], dtype=float32)
    # a = np.ones([n0,n1], float32)  # DATA READ IN LESS ACCURATELY IN loaddata !!
    # float32 can't handle more than 8 significant digits
    a = np.ones([n0, n1], float)
    return(a[:])


def striskey(str):
    """IS str AN OPTION LIKE -C or -ker
    (IT'S NOT IF IT'S -2 or -.9)"""
    iskey = 0
    if str:
        if str[0] == '-':
            iskey = 1
            if len(str) > 1:
                iskey = str[1] not in ['0', '1', '2', '3',
                                       '4', '5', '6', '7', '8', '9', '.']
    return iskey


def pause(text=''):
    inp = input(text)


def stringsplitatoi(strg, separator=''):
    if separator:
        words = strg.split(separator)
    else:
        words = strg.split()
    vals = []
    for word in words:
        vals.append(int(word))
    return vals


def stringsplitatof(str, separator=''):
    if separator:
        words = str.split(separator)
    else:
        words = str.split()
    vals = []
    for word in words:
        vals.append(float(word))
    return vals


def strbegin(str, phr):
    return str[:len(phr)] == phr


def strend(str, phr):
    return str[-len(phr):] == phr


def strbtw(s, left, right=None, r=False):
    """RETURNS THE PART OF STRING s BETWEEN left & right
    EXAMPLE strbtw('det_lab.reg', '_', '.') RETURNS 'lab'
    EXAMPLE strbtw('det_{a}.reg', '{}') RETURNS 'a'
    EXAMPLE strbtw('det_{{a}, b}.reg', '{}', r=1) RETURNS '{a}, b'"""
    out = None
    if right == None:
        if len(left) == 1:
            right = left
        elif len(left) == 2:
            left, right = left
    i1 = s.find(left)
    if (i1 > -1):
        i1 += len(left) - 1
        if r:  # search from the right
            i2 = s.rfind(right, i1 + 1)
        else:
            i2 = s.find(right, i1 + 1)
        if (i2 > i1):
            out = s[i1 + 1:i2]
    return out


def getanswer(question=''):
    ans = -1
    while ans == -1:
        inp = input(question)
        if inp:
            if inp[0].upper() == 'Y':
                ans = 1
            if inp[0].upper() == 'N':
                ans = 0
    return ans


ask = getanswer
    

def common(id1, id2):
    # ASSUME NO IDS ARE NEGATIVE
    id1 = np.array(id1).astype(int)
    id2 = np.array(id2).astype(int)
    n = max((max(id1), max(id2)))
    in1 = np.zeros(n + 1, int)
    in2 = np.zeros(n + 1, int)
    put(in1, id1, 1)
    put(in2, id2, 1)
    inboth = in1 * in2
    ids = np.arange(n + 1)
    ids = np.compress(inboth, ids)
    return ids


def invertselection(ids, all):
    if type(all) == int:  # size input
        all = np.arange(all) + 1
        put(all, np.array(ids) - 1, 0)
        all = np.compress(all, all)
        return all
    else:
        out = []
        for val in all:
            # if val not in ids:
            if not floatin(val, ids):
                out.append(val)
        return out


def findmatch1(x, xsearch, tol=1e-4):
    """RETURNS THE INDEX OF x WHERE xsearch IS FOUND"""
    i = np.argmin(abs(x - xsearch))
    if tol:
        if abs(x[i] - xsearch) > tol:
            print(xsearch, 'NOT FOUND IN findmatch1')
            i = -1
    return i


def findmatch(x, y, xsearch, ysearch, dtol=4, silent=0, returndist=0, xsorted=0):
    """FINDS AN OBJECT GIVEN A LIST OF POSITIONS AND SEARCH COORDINATE
    RETURNS INDEX OF THE OBJECT OR n IF NOT FOUND"""

    n = len(x)
    if silent < 0:
        print('n=', n)
    if not xsorted:
        SI = np.argsort(x)
        x = np.take(x, SI)
        y = np.take(y, SI)
    else:
        SI = np.arange(n)

    dist = 99  # IN CASE NO MATCH IS FOUND

    # SKIP AHEAD IN CATALOG TO x[i] = xsearch - dtol
    # print "SEARCHING..."
    if xsearch > dtol + max(x):
        done = 'too far'
    else:
        done = ''
        i = 0
        while xsearch - x[i] > dtol:
            if silent < 0:
                print(i, xsearch, x[i])
            i = i + 1

    while not done:
        if silent < 0:
            print(i, x[i], xsearch)
        if x[i] - xsearch > dtol:
            done = 'too far'
        else:
            dist = np.sqrt((x[i] - xsearch) ** 2 + (y[i] - ysearch) ** 2)
            if dist < dtol:
                done = 'found'
            elif i == n - 1:
                done = 'last gal'
            else:
                i = i + 1
        if silent < 0:
            print(done)

    if done == 'found':
        if not silent:
            print('MATCH FOUND %1.f PIXELS AWAY AT (%.1f, %.1f)' %
                  (dist, x[i], y[i]))
        ii = SI[i]
    else:
        if not silent:
            print('MATCH NOT FOUND')
        ii = n
    if returndist:
        return ii, dist
    else:
        return ii


def findmatches2(x1, y1, x2, y2):
    """MEASURES ALL DISTANCES, FINDS MINIMA
    SEARCHES FOR 2 IN 1
    RETURNS INDICES AND DISTANCES"""
    dx = subtract.outer(x1, x2)
    dy = subtract.outer(y1, y2)
    d = np.sqrt(dx**2 + dy**2)
    i = np.argmin(d, 0)

    n1 = len(x1)
    n2 = len(x2)
    j = np.arange(n2)
    di = n2 * i + j
    dmin = np.take(d, di)
    return i, dmin

def takeids(data, ids, idrow=0, keepzeros=0):
    """TAKES data COLUMNS CORRESPONDING TO ids.
    data's ID's ARE IN idrow, ITS FIRST ROW BY DEFAULT"""
    dataids = data[idrow].astype(int)
    ids = ids.astype(int)
    outdata = []
    n = data.shape[1]
    for id in ids:
        gotit = 0
        for i in range(n):
            if id == dataids[i]:
                gotit = 1
                break
        if gotit:
            outdata.append(data[:, i])
        elif keepzeros:
            outdata.append(0. * data[:, 0])
    return np.transpose(np.array(outdata))