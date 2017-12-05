from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# Automatically adapted for numpy Jun 08, 2006 by convertcode.py

# Useful functions and definitions

from past.utils import old_div
import os
import sys
import numpy as np
import time
import string


def ask(what="?"):
    """
    Usage:
    ans=ask(pregunta)
    This function prints the string what, 
    (usually a question) and asks for input 
    from the user. It returns the value 0 if the 
    answer starts by 'n' and 1 otherwise, even 
    if the input is just hitting 'enter'
    """
    if what[-1] != '\n':
        what = what + '\n'
    ans = input(what)
    try:
        if ans[0] == 'n':
            return 0
    except:
        pass
    return 1

# Input/Output subroutines

# Read/write headers


def get_header(file):
    """ Returns a string containing all the lines 
    at the top of a file which start by '#'"""
    buffer = ''
    for line in open(file).readlines():
        if line[0] == '#':
            buffer = buffer + line
        else:
            break
    return buffer


def put_header(file, text, comment=1):
    """Adds text (starting by '#' and ending by '\n')
    to the top of a file."""
    if len(text) == 0:
        return
    if text[0] != '#' and comment:
        text = '#' + text
    if text[-1] != '\n':
        text = text + '\n'
    buffer = text + open(file).read()
    open(file, 'w').write(buffer)

# Files containing strings


def get_str(file, cols=0, nrows='all'):
    """ 
        Reads strings from a file
        Usage: 
             x,y,z=get_str('myfile.cat',(0,1,2))
        x,y,z are returned as string lists
    """
    if type(cols) == type(0):
        cols = (cols,)
        nvar = 1
    else:
        nvar = len(cols)
    lista = []
    for i in range(nvar):
        lista.append([])
    buffer = open(file).readlines()
    if nrows == 'all':
        nrows = len(buffer)
    counter = 0
    for lines in buffer:
        if counter >= nrows:
            break
        if lines[0] == '#':
            continue
        pieces = lines.split()
        if len(pieces) == 0:
            continue
        for j in range(nvar):
            lista[j].append(pieces[cols[j]])
        counter = counter + 1
    if nvar == 1:
        return lista[0]
    else:
        return tuple(lista)


def put_str(file, tupla):
    """ Writes tuple of string lists to a file
        Usage:
          put_str(file,(x,y,z))
    """
    if type(tupla) != type((2,)):
        raise 'Need a tuple of variables'
    f = open(file, 'w')
    for i in range(1, len(tupla)):
        if len(tupla[i]) != len(tupla[0]):
            raise 'Variable lists have different lenght'
    for i in range(len(tupla[0])):
        cosas = []
        for j in range(len(tupla)):
            cosas.append(str(tupla[j][i]))
        f.write(''.join(cosas) + '\n')
    f.close()

# Files containing data


def get_data(file, cols=0, nrows='all'):
    """ Returns data in the columns defined by the tuple
    (or single integer) cols as a tuple of float arrays 
    (or a single float array)"""
    if type(cols) == type(0):
        cols = (cols,)
        nvar = 1
    else:
        nvar = len(cols)
    data = get_str(file, cols, nrows)
    if nvar == 1:
        return np.array(list(map(float, data)))
    else:
        data = list(data)
        for j in range(nvar):
            data[j] = np.array(list(map(float, data[j])))
        return tuple(data)


def write(file, variables, header='', format='', append='no'):
    """ Writes tuple of list/arrays to a file 
        Usage:
          put_data(file,(x,y,z),header,format)
        where header is any string  
        and format is a string of the type:
           '%f %f %i ' 
        The default format is all strings
    """
    if type(variables) != type((2,)):
        raise 'Need a tuple of variables'
    if format == '':
        format = '%s  ' * len(variables)
    if append == 'yes':
        f = open(file, 'a')
    else:
        f = open(file, 'w')
    if header != "":
        if header[0] != '#':
            header = '#' + header
        if header[-1] != '\n':
            header = header + '\n'
        f.write(header)
    for i in range(len(variables[0])):
        cosas = []
        for j in range(len(variables)):
            cosas.append(variables[j][i])
        line = format % tuple(cosas)
        f.write(line + '\n')
    f.close()


put_data = write

# Read/write 2D arrays


def get_2Darray(file, cols='all', nrows='all', verbose='no'):
    """Read the data on the defined columns of a file 
    to an 2 array
    Usage:
    x=get_2Darray(file)
    x=get_2Darray(file,range(len(p))
    x=get_2Darray(file,range(0,10,2),nrows=5000)
    Returns x(nrows,ncols)
    """
    if cols == 'all':
        # Get the number of columns in the file
        for line in open(file).readlines():
            pieces = line.split()
            if len(pieces) == 0:
                continue
            if line[0] == '#':
                continue
            nc = len(pieces)
            cols = list(range(nc))
            if verbose == 'yes':
                print('cols=', cols)
            break
    else:
        nc = len(cols)

    lista = get_data(file, cols, nrows)
    nl = len(lista[0])
    x = np.zeros((nl, nc), float)
    for i in range(nc):
        x[:, i] = lista[i]
    return x


def put_2Darray(file, array, header='', format='', append='no'):
    """ Writes a 2D array to a file, where the first 
    index changes along the lines and the second along
    the columns
    Usage: put_2Darray(file,a,header,format)
        where header is any string  
        and format is a string of the type:
           '%f %f %i ' 
    """
    lista = []
    for i in range(array.shape[1]):
        lista.append(array[:, i])
    lista = tuple(lista)
    put_data(file, lista, header, format, append)


class watch(object):
    def set(self):
        self.time0 = time.time()
        print('')
        print('Current time ', time.ctime(self.time0))
        print()

    def check(self):
        if self.time0:
            print()
            print("Elapsed time", time.strftime(
                '%H:%M:%S', time.gmtime(time.time() - self.time0)))
            print()
        else:
            print()
            print('You have not set the initial time')
            print()


def params_file(file):
    """ 
        Read a input file containing the name of several parameters 
        and their values with the following format:

        KEY1   value1,value2,value3   # comment
        KEY2   value

        Returns the dictionary
        dict['KEY1']=(value1,value2,value3)
        dict['KEY2']=value
    """
    dict = {}
    for line in open(file, 'r').readlines():
        if line[0] == ' ' or line[0] == '#':
            continue
        halves = line.split('#')
        # replace commas in case they're present
        halves[0] = halves[0].replace(',', ' ')
        pieces = halves[0].split()
        if len(pieces) == 0:
            continue
        key = pieces[0]
        if len(pieces) < 2:
            mensaje = 'No value(s) for parameter  ' + key
            raise mensaje
        dict[key] = tuple(pieces[1:])
        if len(dict[key]) == 1:
            dict[key] = dict[key][0]
    return dict


def params_commandline(lista):
    """ Read an input list (e.g. command line) 
        containing the name of several parameters 
        and their values with the following format:

        ['-KEY1','value1,value2,value3','-KEY2','value',etc.] 

         Returns a dictionary containing 
        dict['KEY1']=(value1,value2,value3)
        dict['KEY2']=value 
        etc.
    """
    if len(lista) % 2 != 0:
        print('Error: The number of parameter names and values does not match')
        sys.exit()
    dict = {}
    for i in range(0, len(lista), 2):
        key = lista[i]
        if type(key) != type(''):
            raise 'Keyword not string!'
        # replace commas in case they're present
        if key[0] == '-':
            key = key[1:]
        lista[i + 1] = lista[i + 1].replace(',', ' ')
        values = tuple(lista[i + 1].split())
        if len(values) < 1:
            mensaje = 'No value(s) for parameter  ' + key
            raise mensaje
        dict[key] = values
        if len(dict[key]) == 1:
            dict[key] = dict[key][0]
    return dict


def view_keys(dict):
    """Prints sorted dictionary keys"""
    claves = list(dict.keys())
    claves.sort()
    for line in claves:
        print(line.upper(), '  =  ', dict[line])


class params(object):
    """This class defines and manages a parameter dictionary"""

    def __init__(self, d=None):
        if d == None:
            self.d = {}
        else:
            self.d = d

    # Define a few useful methods:

    def fromfile(self, file):
        """Update the parameter dictionary with a file"""
        self.d.update(params_file(file))

    def fromcommandline(self, command_line):
        """Update the parameter dictionary with command line options (sys.argv[i:])"""
        self.d.update(params_commandline(command_line))

    def update(self, dict):
        """Update the parameter information with a dictionary"""
        for key in list(dict.keys()):
            self.d[key] = dict[key]

    def check(self):
        """Interactively check the values of the parameters"""
        view_keys(self.d)
        paso1 = input('Do you want to change any parameter?(y/n)\n')
        while paso1[0] == 'y':
            key = input('Which one?\n')
            if key not in self.d:
                paso2 = input("This parameter is not in the dictionary.\
Do you want to include it?(y/n)\n")
                if paso2[0] == 'y':
                    value = input('value(s) of ' + key + '?= ')
                    self.d[key] = tuple(value.replace(',', ' ').split())
                else:
                    continue
            else:
                value = input('New value(s) of ' + key + '?= ')
                self.d[key] = tuple(value.replace(',', ' ').split())
            view_keys(self.d)
            paso1 = input('Anything else?(y/n)\n')

    def write(self, file):
        claves = list(self.d.keys())
        claves.sort()
        buffer = ''
        for key in claves:
            if type(self.d[key]) == type((2,)):
                values = list(map(str, self.d[key]))
                line = key + ' ' + string.join(values, ',')
            else:
                line = key + ' ' + str(self.d[key])
            buffer = buffer + line + '\n'
        print(line)
        open(file, 'w').write(buffer)


# Some miscellaneous numerical functions

def ascend(x):
    """True if vector x is monotonically ascendent, false otherwise 
       Recommended usage: 
       if not ascend(x): sort(x) 
    """
    return np.alltrue(np.greater_equal(x[1:], x[0:-1]))


def match_resol(xg, yg, xf, method="linear"):
    """ 
    Interpolates and/or extrapolate yg, defined on xg, onto the xf coordinate set.
    Usage:
    ygn=match_resol(xg,yg,xf)
    """
    if type(xf) == type(1.):
        xf = np.array([xf])
    ng = len(xg)
    # print argmin(xg[1:]-xg[0:-1]),min(xg[1:]-xg[0:-1]),xg[argmin(xg[1:]-xg[0:-1])]
    d = old_div((yg[1:] - yg[0:-1]), (xg[1:] - xg[0:-1]))
    # Get positions of the new x coordinates
    ind = np.clip(np.searchsorted(xg, xf) - 1, 0, ng - 2)
    ygn = np.take(yg, ind) + np.take(d, ind) * (xf - np.take(xg, ind))
    if len(ygn) == 1:
        ygn = ygn[0]
    return ygn


def dist(x, y, xc=0., yc=0.):
    """Distance between point (x,y) and a center (xc,yc)"""
    return np.sqrt((x - xc)**2 + (y - yc)**2)


def loc2d(a, extremum='max'):
    """ Locates the maximum of an 2D array
        Usage:
        max_vec=max_loc2d(a)
    """
    forma = a.shape
    if len(forma) > 2:
        raise "Array dimension > 2"
    if extremum != 'min' and extremum != 'max':
        raise 'Which extremum are you looking for?'
    x = np.ravel(a)
    if extremum == 'min':
        i = np.argmin(x)
    else:
        i = np.argmax(x)
    i1 = old_div(i, forma[1])
    i2 = i % forma[1]
    return i1, i2


def hist(a, bins):
    """
    Histogram of 'a' defined on the bin grid 'bins'
       Usage: h=hist(p,xp)
    """
    n = np.searchsorted(np.sort(a), bins)
    n = np.concatenate([n, [len(a)]])
    n = np.array(list(map(float, n)))
#    n=np.array(n)
    return n[1:] - n[:-1]


def bin_stats(x, y, xbins, stat='average'):
    """Given the variable y=f(x), and 
    the bins limits xbins, return the 
    corresponding statistics, e.g. <y(xbins)>
    Options are rms, median y average
    """
    nbins = len(xbins)
    if stat == 'average' or stat == 'mean':
        func = np.mean
    elif stat == 'median':
        func = np.median
    elif stat == 'rms' or stat == 'std':
        func = np.std
    elif stat == 'std_robust' or stat == 'rms_robust':
        func = std_robust
    elif stat == 'mean_robust':
        func = mean_robust
    elif stat == 'median_robust':
        func = median_robust
    elif stat == 'sum':
        func = sum
    results = []
    for i in range(nbins):
        if i < nbins - 1:
            good = (np.greater_equal(x, xbins[i])
                    * np.less(x, xbins[i + 1]))
        else:
            good = (np.greater_equal(x, xbins[-1]))
        if sum(good) > 1.:
            results.append(func(np.compress(good, y)))
        else:
            results.append(0.)
            print('Bin starting at xbins[%i] has %i points' % (i, sum(good)))
    return np.array(results)


#def bin_aver(x, y, xbins):
#    return bin_stats(x, y, xbins, stat='average')


def p2p(x):
    return max(x) - min(x)


def purge_outliers(x, n_sigma=3., n=5):
    # Experimental yet. Only 1 dimension
    for i in range(n):
        med = np.median(x)
        # rms=std_log(x)
        rms = np.std(x)
        x = np.compress(np.less_equal(abs(x - med), n_sigma * rms), x)
    return x


class stat_robust(object):
    # Generates robust statistics using a sigma clipping
    # algorithm. It is controlled by the parameters n_sigma
    # and n, the number of iterations
    def __init__(self, x, n_sigma=3, n=5, reject_fraction=None):
        self.x = x
        self.n_sigma = n_sigma
        self.n = n
        self.reject_fraction = reject_fraction

    def run(self):
        good = np.ones(len(self.x))
        nx = sum(good)
        if self.reject_fraction == None:
            for i in range(self.n):
                if i > 0:
                    xs = np.compress(good, self.x)
                else:
                    xs = self.x
                #            aver=mean(xs)
                aver = np.median(xs)
                std1 = np.std(xs)
                good = good * np.less_equal(abs(self.x - aver), self.n_sigma * std1)
                nnx = sum(good)
                if nnx == nx:
                    break
                else:
                    nx = nnx
        else:
            npx = float(len(self.x))
            nmin = int((0.5 * self.reject_fraction) * npx)
            nmax = int((1. - 0.5 * self.reject_fraction) * npx)
            orden = np.argsort(self.x)
            #connect(np.arange(len(self.x)), np.sort(self.x))
            good = np.greater(orden, nmin) * np.less(orden, nmax)

        self.remaining = np.compress(good, self.x)
        self.max = max(self.remaining)
        self.min = min(self.remaining)
        self.mean = np.mean(self.remaining)
        self.rms = np.std(self.remaining)
        self.rms0 = np.rms(self.remaining)  # --DC
        self.median = np.median(self.remaining)
        self.outliers = np.compress(np.logical_not(good), self.x)
        self.n_remaining = len(self.remaining)
        self.n_outliers = len(self.outliers)
        self.fraction = 1. - \
            (old_div(float(self.n_remaining), float(len(self.x))))


def std_robust(x, n_sigma=3., n=5):
    x = purge_outliers(x, n_sigma, n)
    return np.std(x - np.mean(x))


def mean_robust(x, n_sigma=3., n=5):
    x = purge_outliers(x, n_sigma, n)
    return np.mean(x)


def median_robust(x, n_sigma=3., n=5):
    x = purge_outliers(x, n_sigma, n)
    return np.median(x)


def multicompress(condition, variables):
    lista = list(variables)
    n = len(lista)
    for i in range(n):
        lista[i] = np.compress(condition, lista[i])
    return tuple(lista)


def multisort(first, followers):
    # sorts the vector first and matches the ordering
    # of followers to it
    # Usage:
    # new_followers=multi_sort(first,followers)
    order = np.argsort(first)
    if type(followers) != type((1,)):
        return np.take(followers, order)
    else:
        nvectors = len(followers)
        lista = []
        for i in range(nvectors):
            lista.append(np.take(followers[i], order))
        return tuple(lista)


def erfc(x):
    """
    Returns the complementary error function erfc(x)
    erfc(x)=1-erf(x)=2/np.sqrt(pi)*\int_x^\inf e^-t^2 dt   
    """
    try:
        x.shape
    except:
        x = np.array([x])
    z = abs(x)
    t = old_div(1., (1. + 0.5 * z))
    erfcc = t * np.exp(-z * z -
                    1.26551223 + t * (
                        1.00002368 + t * (
                            0.37409196 + t * (
                                0.09678418 + t * (
                                    -0.18628806 + t * (
                                        0.27886807 + t * (
                                            -1.13520398 + t * (
                                                1.48851587 + t * (
                                                    -0.82215223 + t * 0.17087277)
                                            ))))))))
    erfcc = np.where(np.less(x, 0.), 2. - erfcc, erfcc)
    return erfcc


def erf(x):
    """
    Returns the error function erf(x)
    erf(x)=2/np.sqrt(pi)\int_0^x \int e^-t^2 dt
    """
    return 1. - erfc(x)


def gauss_int_erf(x=(0., 1.), average=0., sigma=1.):
    """
    Returns integral (x) of p=int_{-x1}^{+x} 1/np.sqrt(2 pi)/sigma exp(-(t-a)/2sigma^2) dt
    """
    x = (x - average) / np.sqrt(2.) / sigma
    return (erf(x) - erf(x[0])) * .5


gauss_int = gauss_int_erf


def inv_gauss_int(p):
    # Brute force approach. Limited accuracy for >3sigma
    # find something better
    # DO NOT USE IN LOOPS (very slow)
    """
    Calculates the x sigma value corresponding to p
    p=int_{-x}^{+x} g(x) dx
    """
    if p < 0. or p > 1.:
        print('Wrong value for p(', p, ')!')
        sys.exit()
    step = .00001
    xn = np.arange(0., 4. + step, step)
    gn = 1. / np.sqrt(2. * np.pi) * np.exp(old_div(-xn**2, 2.))
    cgn = np.add.accumulate(gn) * step
    p = old_div(p, 2.)
    ind = np.searchsorted(cgn, p)
    return xn[ind]


class lsq(object):
    # Defines a least squares minimum estimator given two
    # vectors x and y
    def __init__(self, x, y, dy=0.):
        try:
            dy.shape
        except:
            dy = x * 0. + 1.
        dy2 = dy**2
        s = np.add.reduce(old_div(1., dy2))
        sx = np.add.reduce(old_div(x, dy2))
        sy = np.add.reduce(old_div(y, dy2))
        sxx = np.add.reduce(x * x / dy2)
        sxy = np.add.reduce(x * y / dy2)
        delta = s * sxx - sx * sx
        self.a = old_div((sxx * sy - sx * sxy), delta)
        self.b = old_div((s * sxy - sx * sy), delta)
        self.da = np.sqrt(old_div(sxx, delta))
        self.db = np.sqrt(old_div(s, delta))

    def fit(self, x):
        return self.b * x + self.a


# Tests

def Testing(test):
    print('Testing ', test, '...')
