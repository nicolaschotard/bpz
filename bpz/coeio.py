from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# Automatically adapted for numpy Jun 0, 2006
# By hand: 'float' -> float, Float -> float, Int -> int

# coeio.py
# INPUT / OUTUT OF FILES

from builtins import range
from builtins import object
from past.utils import old_div
from .coetools import *
from . import MLab_coe	
import string

#import fitsio
try:
    import pyfits  # , numarray
    pyfitsloaded = True
except:
    pyfitsloaded = False
    # pass # print "pyfits not installed, so not importing it"

try:
    import Image
    from coeim import *
except:
    pass  # print "Image not installed, so not importing it"

from os.path import exists, join
import numpy as np


def strspl(s):
    if type(s) == str:
        if string.find(s, ' ') > -1:
            s = s.split()
    return s

    
def pint(A, n=0):
    """Makes it easier to view float arrays:
    prints A.astype(int)"""
    if type(A) in [list, tuple]:
        A = np.array(A)
    if n != 0:
        A = A * 10**n
    print(A.astype(int))


def pintup(A, n=0):
    """Makes it easier to view float arrays:
    prints A.astype(int)"""
    pint(flipud(A), n)


if pyfitsloaded:
    # UNLESS $NUMERIX IS SET TO numpy, pyfits(v1.1b) USES NumArray
    pyfitsusesnumpy = (float(pyfits.__version__[:3]) >= 1.1) and (
        numerix == 'numpy')
    if not pyfitsusesnumpy:
        print('You probably should have done this first: setenv NUMERIX numpy')
        import numarray


def recapfile(name, ext):
    """CHANGE FILENAME EXTENSION"""
    if ext[0] != '.':
        ext = '.' + ext
    i = string.rfind(name, ".")
    if i == -1:
        outname = name + ext
    else:
        outname = name[:i] + ext
    return outname


def capfile(name, ext):
    """ADD EXTENSION TO FILENAME IF NECESSARY"""
    if ext[0] != '.':
        ext = '.' + ext
    n = len(ext)
    if name[-n:] != ext:
        name += ext
    return name


def decapfile(name, ext=''):
    """REMOVE EXTENSION FROM FILENAME IF PRESENT
    IF ext LEFT BLANK, THEN ANY EXTENSION WILL BE REMOVED"""
    if ext:
        if ext[0] != '.':
            ext = '.' + ext
        n = len(ext)
        if name[-n:] == ext:
            name = name[:-n]
    else:
        i = string.rfind(name, '.')
        if i > -1:
            name = name[:i]
    return name


uncapfile = decapfile


def params_cl():
    """RETURNS PARAMETERS FROM COMMAND LINE ('cl') AS DICTIONARY:
    KEYS ARE OPTIONS BEGINNING WITH '-'
    VALUES ARE WHATEVER FOLLOWS KEYS: EITHER NOTHING (''), A VALUE, OR A LIST OF VALUES
    ALL VALUES ARE CONVERTED TO INT / FLOAT WHEN APPROPRIATE"""
    list = sys.argv[:]
    i = 0
    dict = {}
    oldkey = ""
    key = ""
    list.append('')  # EXTRA ELEMENT SO WE COME BACK AND ASSIGN THE LAST VALUE
    while i < len(list):
        if striskey(list[i]) or not list[i]:  # (or LAST VALUE)
            if key:  # ASSIGN VALUES TO OLD KEY
                if value:
                    if len(value) == 1:  # LIST OF 1 ELEMENT
                        value = value[0]  # JUST ELEMENT
                dict[key] = value
            if list[i]:
                key = list[i][1:]  # REMOVE LEADING '-'
                value = None
                dict[key] = value  # IN CASE THERE IS NO VALUE!
        else:  # VALUE (OR HAVEN'T GOTTEN TO KEYS)
            if key:  # (HAVE GOTTEN TO KEYS)
                if value:
                    value.append(str2num(list[i]))
                else:
                    value = [str2num(list[i])]
        i += 1

    return dict


def delfile(file, silent=0):
    if os.path.exists(file) or os.path.islink(file):  # COULD BE BROKEN LINK!
        if not silent:
            print('REMOVING ', file, '...')
        os.remove(file)
    else:
        if not silent:
            print("CAN'T REMOVE", file, "DOES NOT EXIST.")


rmfile = delfile


def dirfile(filename, dir=""):
    """RETURN CLEAN FILENAME COMPLETE WITH PATH
    JOINS filename & dir, CHANGES ~/ TO home"""
    if filename[0:2] == '~/':
        filename = os.path.join(home, filename[2:])
    else:
        if dir[0:2] == '~/':
            dir = os.path.join(home, dir[2:])
        filename = os.path.join(dir, filename)
    return filename


def loadfile(filename, dir="", silent=0, keepnewlines=0):
    infile = dirfile(filename, dir)
    if not silent:
        print("Loading ", infile, "...\n")
    fin = open(infile, 'r')
    sin = fin.readlines()
    fin.close()
    if not keepnewlines:
        for i in range(len(sin)):
            sin[i] = sin[i][:-1]
    return sin


def loadheader(filename, dir="", silent=0, keepnewlines=0):
    infile = dirfile(filename, dir)
    if not silent:
        print("Loading ", infile, "...\n")
    fin = open(infile, 'r')
    line = '#'
    sin = []
    while line:
        line = fin.readline()
        if line[0] != '#':
            break
        else:
            sin.append(line)
    fin.close()
    if not keepnewlines:
        for i in range(len(sin)):
            sin[i] = sin[i][:-1]
    return sin


def fileempty(filename, dir="", silent=0, delifempty=0):
    """CHECK IF A FILE ACTUALLY HAS ANYTHING IN IT
    OR IF IT'S JUST CONTAINS BLANK / COMMENTED LINES"""
    filename = dirfile(filename, dir)
    gotdata = 0
    if os.path.exists(filename):
        fin = open(filename, 'r')
        line = 'x'
        while line and not gotdata:
            line = fin.readline()
            if line:
                if line[0] != '#':
                    gotdata = 1
        if delifempty:
            if not gotdata:
                os.remove(filename)
        fin.close()
    return (gotdata == 0)


def delfileifempty(filename, dir="", silent=0):
    fileempty(filename, dir, silent, 1)


def assigndict(keys, values):
    n = len(keys)
    if n != len(values):
        print("keys & values DON'T HAVE SAME LENGTH IN coeio.assigndict!")
    else:
        d = {}
        for i in range(n):
            d[keys[i]] = values[i]
        return d


def loaddict1(filename, dir="", silent=0):
    lines = loadfile(filename, dir, silent)
    dict = {}
    for line in lines:
        if line[0] != '#':
            words = line.split()
            key = str2num(words[0])
            val = ''  # if nothing there
            if len(words) == 2:
                val = str2num(words[1])
            elif len(words) > 2:
                val = []
                for word in words[1:]:
                    val.append(str2num(word))

            dict[key] = val
    return dict


def loaddict(filename, dir="", silent=0):
    lines = loadfile(filename, dir, silent)
    dict = {}
    for line in lines:
        if line[0] != '#':
            words = line.split()
            key = str2num(words[0])
            val = ''  # if nothing there
            valstr = string.join(words[1:], ' ')
            valtuple = False
            if valstr[0] in '[(' and valstr[-1] in '])':  # LIST / TUPLE!
                valtuple = valstr[0] == '('
                valstr = valstr[1:-1].replace(',', '')
                words[1:] = valstr.split()
            if len(words) == 2:
                val = str2num(words[1])
            elif len(words) > 2:
                val = []
                for word in words[1:]:
                    val.append(str2num(word))
                if valtuple:
                    val = tuple(val)

            dict[key] = val
    return dict


# THE LONG AWAITED MIXED FORMAT LOADER!
def loadcols(infile, format='', pl=0):
    """LOADS A DATA FILE CONTAINING COLUMNS OF DIFFERENT TYPES (STRING, FLOAT, & INT
    RETURNS A LIST OF LISTS
    format (OPTIONAL) INPUT AS A STRING, ONE LETTER (s, d, or f) FOR EACH COLUMN
    ARRAY OUTPUT FOR NUMBERS: ADD AN 'A' TO THE BEGINNING OF format 
    USAGE: labels, x, y = loadcols('~/A1689/arcsnew_lab.txt', format='sdd')"""
    txt = loadfile(infile)
    while txt[0][0] == '#':
        txt = txt[1:]
    line = txt[0]
    words = line.split()
    ncols = len(words)
    data = [[]]
    for icol in range(ncols - 1):
        data.append([])

    arrayout = 0

    if format:
        if format[0] == 'A':
            format = format[1:]
            arrayout = 1

    if not format:  # FIGURE IT OUT BASED ON FIRST LINE ONLY
        for word in words:
            try:
                datum = int(word)
                format += 'd'
            except:
                try:
                    datum = float(word)
                    format += 'f'
                except:
                    format += 's'

    # print format
    roundcols = []
    for line in txt:
        if line:
            if line[0] != '#':
                words = line.split()
                if pl:
                    print(line)
                for iword in range(len(words)):
                    if iword > len(format) - 1:
                        print('EXTRA CONTENT IN LINE: ', end=' ')
                        print(string.join(words[iword:]))
                        break
                    # print iword
                    word = words[iword]
                    formatum = format[iword]
                    if formatum == 'f':
                        datum = float(word)
                    elif formatum == 'd':
                        try:
                            datum = int(word)
                        except:
                            #datum = int(round(float(word)))
                            datum = float(word)
                            try:
                                datum = roundint(datum)
                                if not (iword + 1) in roundcols:
                                    roundcols.append(iword + 1)
                            except:
                                pass
                    else:
                        datum = word
                    data[iword].append(datum)

    if roundcols:
        if len(roundcols) > 1:
            print(
                'WARNING, THE FOLLOWING COLUMNS WERE ROUNDED FROM FLOAT TO INT: ', roundcols)
        else:
            print(
                'WARNING, THE FOLLOWING COLUMN WAS ROUNDED FROM FLOAT TO INT: ', roundcols)

    if arrayout:
        for icol in range(ncols):
            if format[icol] in 'df':
                data[icol] = np.array(data[icol])

    return data

# CRUDE


def savecols(data, filename, format=''):
    ncols = len(data)
    nrows = len(data[0])
    if not format:
        for icol in range(ncols):
            datum = data[icol][0]
            if type(datum) == int:
                format += 'd'
            elif type(datum) == float:
                format += 'f'
            else:
                format += 's'

    # CHANGE format from 'sdd' TO ' %s %d %d\n'
    ff = ' '
    for f in format:
        if f == 'f':
            ff += '%.3f '
        else:
            ff += '%' + f + '  '
    format = ff[:-1]
    format += '\n'
    fout = open(filename, 'w')
    for irow in range(nrows):
        dataline = []
        for icol in range(ncols):
            dataline.append(data[icol][irow])
        fout.write(format % tuple(dataline))

    fout.close()


def savedata(data, filename, dir="", header="", separator="  ", format='', labels='', descriptions='', units='', notes=[], pf=0, maxy=300, machine=0, silent=0):
    """Saves an array as an ascii data file into an array."""
    # AUTO FORMATTING (IF YOU WANT, ALSO OUTPUTS FORMAT SO YOU CAN USE IT NEXT TIME w/o HAVING TO CALCULATE IT)
    # maxy: ONLY CHECK THIS MANY ROWS FOR FORMATTING
    # LABELS MAY BE PLACED ABOVE EACH COLUMN
    # IMPROVED SPACING

    dow = filename[-1] == '-'  # DON'T OVERWRITE
    if dow:
        filename = filename[:-1]

    tr = filename[-1] == '+'
    if tr:
        data = np.transpose(data)
        filename = filename[:-1]

    if machine:
        filename = 'datafile%d.txt' % machine  # doubles as table number
    outfile = dirfile(filename, dir)

    if dow and os.path.exists(outfile):
        print(outfile, " ALREADY EXISTS")
    else:
        skycat = strend(filename, '.scat')
        if skycat:
            separator = '\t'
        if len(data.shape) == 1:
            data = np.reshape(data, (len(data), 1))
            #data = data[:,NewAxis]
        [ny, nx] = data.shape
        # WHETHER THE COLUMN HAS ANY NEGATIVE NUMBERS: 1=YES, 0=NO
        colneg = [0] * nx
        collens = []
        if format:
            if type(format) == dict:  # CONVERT DICTIONARY FORM TO LIST
                dd = ' '
                for label in labels:
                    if label in list(format.keys()):
                        dd += format[label]
                    else:
                        print("WARNING: YOU DIDN'T SUPPLY A FORMAT FOR",
                              label + ".  USING %.3f AS DEFAULT")
                        dd += '%.3f'
                    dd += '  '
                dd = dd[:-2] + '\n'  # REMOVE LAST WHITESPACE, ADD NEWLINE
                format = dd
                # print format
        else:
            if not silent:
                print("Formatting... ")
            coldec = [0] * nx  # OF DECIMAL PLACES
            # LENGTH BEFORE DECIMAL PLACE (INCLUDING AN EXTRA ONE IF IT'S NEGATIVE)
            colint = [0] * nx
            # colneg = [0] * nx  # WHETHER THE COLUMN HAS ANY NEGATIVE NUMBERS: 1=YES, 0=NO
            # WHETHER THE COLUMN HAS ANY REALLY BIG NUMBERS THAT NEED exp FORMAT : 1=YES, 0=NO
            colexp = [0] * nx

            if machine:
                maxy = 0
            if maxy is None or ny <= maxy:
                yyy = list(range(ny))
            else:
                yyy = np.arange(maxy) * (old_div((ny - 1.), (maxy - 1.)))
                yyy = yyy.astype(int)
            for iy in yyy:
                for ix in range(nx):
                    datum = data[iy, ix]
                    print(datum)
                    if not isinstance(datum, (list, np.ndarray)) or \
                       MLab_coe.isNaN(datum):
                        ni, nd = 1, 1
                    else:
                        # IF TOO BIG OR TOO SMALL, NEED exp FORMAT
                        if (abs(datum) > 1.e9) or (0 < abs(datum) < 1.e-5):
                            ni, nd = 1, 3
                            colexp[ix] = 1
                        else:
                            ni = len("% d" % datum) - 1
                            if ni <= 3:
                                nd = ndec(datum, max=4)
                            else:
                                nd = ndec(datum, max=7 - ni)
                            # Float32: ABOUT 7 DIGITS ARE ACCURATE (?)

                    if ni > colint[ix]:  # IF BIGGEST, YOU GET TO DECIDE NEG SPACE OR NO
                        colneg[ix] = (datum < 0)
                        # print '>', ix, colneg[ix], nd, coldec[ix]
                    # IF MATCH BIGGEST, YOU CAN SET NEG SPACE ON (NOT OFF)
                    elif ni == colint[ix]:
                        colneg[ix] = (datum < 0) or colneg[ix]
                        # print '=', ix, colneg[ix], nd, coldec[ix]
                    coldec[ix] = max([nd, coldec[ix]])
                    colint[ix] = max([ni, colint[ix]])

            # print colneg
            # print colint
            # print coldec

            collens = []
            for ix in range(nx):
                if colexp[ix]:
                    collen = 9 + colneg[ix]
                else:
                    # EXTRA ONES FOR DECIMAL POINT / - SIGN
                    collen = colint[ix] + coldec[ix] + \
                        (coldec[ix] > 0) + (colneg[ix] > 0)
                if labels and not machine:
                    # MAKE COLUMN BIG ENOUGH TO ACCOMODATE LABEL
                    collen = max((collen, len(labels[ix])))
                collens.append(collen)

            format = ' '
            for ix in range(nx):
                collen = collens[ix]
                format += '%'
                if colneg[ix]:  # NEGATIVE
                    format += ' '
                if colexp[ix]:  # REALLY BIG (EXP FORMAT)
                    format += '.3e'
                else:
                    if coldec[ix]:  # FLOAT
                        format += "%d.%df" % (collen, coldec[ix])
                    else:  # DECIMAL
                        format += "%dd" % collen
                if ix < nx - 1:
                    format += separator
                else:
                    format += "\n"
            if pf:
                print("format='%s\\n'" % format[:-1])

        # NEED TO BE ABLE TO ALTER INPUT FORMAT
        if machine:  # machine readable
            collens = []  # REDO collens (IN CASE format WAS INPUT)
            mformat = ''
            separator = ' '
            colformats = format.split('%')[1:]
            format = ''  # redoing format, too
            for ix in range(nx):
                # print ix, colformats
                cf = colformats[ix]
                format += '%'
                if cf[0] == ' ':
                    format += ' '
                cf = string.strip(cf)
                format += cf
                mformat += {'d': 'I', 'f': 'F', 'e': 'E'}[cf[-1]]
                mformat += cf[:-1]
                if ix < nx - 1:
                    format += separator
                    mformat += separator
                else:
                    format += "\n"
                    mformat += "\n"
                # REDO collens (IN CASE format WAS INPUT)
                colneg[ix] = string.find(cf, ' ') == -1
                if string.find(cf, 'e') > -1:
                    collen = 9 + colneg[ix]
                else:
                    cf = cf.split('.')[0]  # FLOAT: Number before '.'
                    cf = cf.split('d')[0]  # INT:   Number before 'd'
                    collen = int(cf)
                collens.append(collen)
        else:
            if not collens:
                collens = []  # REDO collens (IN CASE format WAS INPUT)
                colformats = format.split('%')[1:]
                for ix in range(nx):
                    cf = colformats[ix]
                    colneg[ix] = string.find(cf, ' ') == -1
                    if string.find(cf, 'e') > -1:
                        collen = 9 + colneg[ix]
                    else:
                        # FLOAT: Number before '.'
                        cf = cf.split('.')[0]
                        # INT:   Number before 'd'
                        cf = cf.split('d')[0]
                        collen = int(cf)
                    if labels:
                        # MAKE COLUMN BIG ENOUGH TO ACCOMODATE LABEL
                        collen = max((collen, len(labels[ix])))
                    collens.append(collen)


# if machine:  # machine readable
##                 mformat = ''
# for ix in range(nx):
##                     collen = collens[ix]
# if colexp[ix]: # EXP
##                         mformat += 'E5.3'
# elif coldec[ix]: # FLOAT
##                         mformat += 'F%d.%d' % (collen, coldec[ix])
# else: # DECIMAL
##                         mformat += 'I%d' % collen
# if ix < nx - 1:
##                         mformat += separator
# else:
##                         mformat += "\n"

        if descriptions:
            if type(descriptions) == dict:  # CONVERT DICTIONARY FORM TO LIST
                dd = []
                for label in labels:
                    dd.append(descriptions.get(label, ''))
                descriptions = dd

        if units:
            if type(units) == dict:  # CONVERT DICTIONARY FORM TO LIST
                dd = []
                for label in labels:
                    dd.append(units.get(label, ''))
                units = dd

        if not machine:
            # if not descriptions:
            ##                 descriptions = labels
            if labels:
                headline = ''
                maxcollen = 1
                for label in labels:
                    maxcollen = max([maxcollen, len(label)])
                for ix in range(nx):
                    label = string.ljust(labels[ix], maxcollen)
                    headline += '# %2d %s' % (ix + 1, label)
                    if descriptions:
                        if descriptions[ix]:
                            headline += '  %s' % descriptions[ix]
                    headline += '\n'
                    # headline += '# %2d %s  %s\n' % (ix+1, label, descriptions[ix])
                    # ff = '# %%2d %%%ds  %%s\n' % maxcollen  # '# %2d %10s %s\n'
                    #headline += ff % (ix+1, labels[ix], descriptions[ix])
                    # headline += '# %2d %s\n' % (ix+1, descriptions[ix])
                headline += '#\n'
                headline += '#'
                colformats = format.split('%')[1:]
                if not silent:
                    print()
                for ix in range(nx):
                    cf = colformats[ix]
                    collen = collens[ix]
                    # label = labels[ix][:collen]  # TRUNCATE LABEL TO FIT COLUMN DATA
                    label = labels[ix]
                    label = string.center(label, collen)
                    headline += label + separator
                headline += '\n'
                if not header:
                    header = [headline]
                else:
                    if header[-1] != '.':  # SPECIAL CODE TO REFRAIN FROM ADDING TO HEADER
                        header.append(headline)

                if skycat:
                    headline1 = ''
                    headline2 = ''
                    for label in labels:
                        headline1 += label + '\t'
                        headline2 += '-' * len(label) + '\t'
                    headline1 = headline1[:-1] + '\n'
                    headline2 = headline2[:-1] + '\n'
                    header.append(headline1)
                    header.append(headline2)

        elif machine:  # Machine readable Table!
            maxlabellen = 0
            for ix in range(nx):
                flaglabel = 0
                if len(labels[ix]) >= 2:
                    flaglabel = (labels[ix][1] == '_')
                if not flaglabel:
                    labels[ix] = '  ' + labels[ix]
                if len(labels[ix]) > maxlabellen:
                    maxlabellen = len(labels[ix])
            # labelformat = '%%%ds' % maxlabellen
            if not header:
                header = []
                header.append('Title:\n')
                header.append('Authors:\n')
                header.append('Table:\n')
            header.append('=' * 80 + '\n')
            header.append('Byte-by-byte Description of file: %s\n' % filename)
            header.append('-' * 80 + '\n')
            #header.append('   Bytes Format Units   Label    Explanations\n')
            headline = '   Bytes Format Units   '
            headline += string.ljust('Label', maxlabellen - 2)
            headline += '  Explanations\n'
            header.append(headline)
            header.append('-' * 80 + '\n')
            colformats = mformat.split()
            byte = 1
            for ix in range(nx):
                collen = collens[ix]
                headline = ' %3d-%3d' % (byte, byte + collen - 1)  # bytes
                headline += ' '
                # format:
                cf = colformats[ix]
                headline += cf
                headline += ' ' * (7 - len(cf))
                # units:
                cu = ''
                if units:
                    cu = units[ix]
                if not cu:
                    cu = '---'
                headline += cu
                headline += '   '
                # label:
                label = labels[ix]
                headline += string.ljust(labels[ix], maxlabellen)
                # descriptions:
                if descriptions:
                    headline += '  '
                    headline += descriptions[ix]
                headline += '\n'
                header.append(headline)
                byte += collen + 1
            header.append('-' * 80 + '\n')
            if notes:
                for inote in range(len(notes)):
                    headline = 'Note (%d): ' % (inote + 1)
                    note = notes[inote].split('\n')
                    headline += note[0]
                    if headline[-1] != '\n':
                        headline += '\n'
                    header.append(headline)
                    if len(note) > 1:
                        for iline in range(1, len(note)):
                            # make sure it's not blank (e.g., after \n)
                            if note[iline]:
                                headline = ' ' * 10
                                headline += note[iline]
                                if headline[-1] != '\n':
                                    headline += '\n'
                                header.append(headline)
            header.append('-' * 80 + '\n')

        if not silent:
            print("Saving ", outfile, "...\n")

        fout = open(outfile, 'w')

        # SPECIAL CODE TO REFRAIN FROM ADDING TO HEADER:
        # LAST ELEMENT IS A PERIOD
        if header:
            if header[-1] == '.':
                header = header[:-1]

        for headline in header:
            fout.write(headline)
            if not (headline[-1] == '\n'):
                fout.write('\n')

        for iy in range(ny):
            fout.write(format % tuple(data[iy].tolist()))

        fout.close()


def loaddata(filename, dir="", silent=0, headlines=0):
    """Loads an ascii data file (OR TEXT BLOCK) into an array.
    Skips header (lines that begin with #), but saves it in the variable 'header', which can be accessed by:
    from coeio import header"""

    global header

    tr = 0
    if filename[-1] == '+':  # NP.TRANSPOSE OUTPUT
        tr = 1
        filename = filename[:-1]

    if len(filename[0]) > 1:
        sin = filename
    else:
        sin = loadfile(filename, dir, silent)

    header = sin[0:headlines]
    sin = sin[headlines:]

    headlines = 0
    while (headlines < len(sin)) and (sin[headlines][0] == '#'):
        headlines = headlines + 1
    header[len(header):] = sin[0:headlines]

    ny = len(sin) - headlines
    if ny == 0:
        if headlines:
            ss = sin[headlines - 1].split()[1:]
        else:
            ss = []
    else:
        ss = sin[headlines].split()

    nx = len(ss)
    #size = [nx,ny]
    data = FltArr(ny, nx)

    sin = sin[headlines:ny + headlines]

    for iy in range(ny):
        ss = sin[iy].split()
        for ix in range(nx):
            try:
                data[iy, ix] = float(ss[ix])
            except:
                continue
                #print(ss)
                #print(ss[ix])
                #data[iy, ix] = ss[ix]

    if tr:
        data = np.transpose(data)

    if data.shape[0] == 1:  # ONE ROW
        return np.ravel(data)
    else:
        return data


def loadlist(filename, dir="./"):
    """Loads an ascii data file into a list.
    The file has one number on each line.
    Skips header (lines that begin with #), but saves it in the variable 'header'."""

    global header

    #    os.chdir("/home/coe/imcat/ksb/A1689txitxo/R/02/")

    infile = dirfile(filename, dir)
    print("Loading ", infile, "...\n")

    fin = open(infile, 'r')
    sin = fin.readlines()
    fin.close

    headlines = 0
    while sin[headlines][0] == '#':
        headlines = headlines + 1
    header = sin[0:headlines - 1]

    n = len(sin) - headlines

    sin = sin[headlines:n + headlines]

    list = []
    for i in range(n):
        list.append(float(sin[i]))

    return list


def machinereadable(filename, dir=''):
    if filename[-1] == '+':
        filename = filename[:-1]
    filename = dirfile(filename, dir)
    fin = open(filename, 'r')
    line = fin.readline()
    return line[0] == 'T'  # BEGINS WITH Title:


def loadmachine(filename, dir="", silent=0):
    """Loads machine-readable ascii data file into a VarsClass()
    FORMAT...
    Title:
    Authors:
    Table:
    ================================================================================
    Byte-by-byte Description of file: datafile1.txt
    --------------------------------------------------------------------------------
       Bytes Format Units   Label    Explanations 
    --------------------------------------------------------------------------------
    (columns & descriptions)
    --------------------------------------------------------------------------------
    (notes)
    --------------------------------------------------------------------------------
    (data)
    """

    cat = VarsClass('')
    filename = dirfile(filename, dir)
    fin = open(filename, 'r')

    # SKIP HEADER
    line = ' '
    while string.find(line, 'Bytes') == -1:
        line = fin.readline()
    fin.readline()

    # COLUMNS & DESCRIPTIONS
    cols = []
    cat.labels = []
    line = fin.readline()
    while line[0] != '-':
        xx = []
        xx.append(int(line[1:4]))
        xx.append(int(line[5:8]))
        cols.append(xx)
        cat.labels.append(line[9:].split()[2])
        line = fin.readline()

    nx = len(cat.labels)

    # NOW SKIP NOTES:
    line = fin.readline()
    while line[0] != '-':
        line = fin.readline()

    # INITIALIZE DATA
    for ix in range(nx):
        exec('cat.%s = []' % cat.labels[ix])

    # LOAD DATA
    while line:
        line = fin.readline()
        if line:
            for ix in range(nx):
                s = line[cols[ix][0] - 1:cols[ix][1]]
                # print cols[ix][0], cols[ix][1], s
                val = float(s)
                exec('cat.%s.append(val)' % cat.labels[ix])

    # FINALIZE DATA
    for ix in range(nx):
        exec('cat.%s = np.array(cat.%s)' % (cat.labels[ix], cat.labels[ix]))

    return cat


def loadpymc(filename, dir="", silent=0):
    filename = dirfile(filename, dir)
    ind = loaddict(filename + '.ind')
    i, data = loaddata(filename + '.out+')

    cat = VarsClass()
    for label in list(ind.keys()):
        ilo, ihi = ind[label]
        chunk = data[ilo - 1:ihi]
        cat.add(label, chunk)

    return cat


class Cat2D_xyflip(object):
    def __init__(self, filename='', dir="", silent=0, labels='x y z'.split()):
        if len(labels) == 2:
            labels.append('z')
        self.labels = labels
        if filename:
            if filename[-1] != '+':
                filename += '+'
            self.data = loaddata(filename, dir)
            self.assigndata()

    def assigndata(self):
        exec('self.%s = self.x = self.data[1:,0]' % self.labels[0])
        exec('self.%s = self.y = self.data[0,1:]' % self.labels[1])
        exec('self.%s = self.z = self.data[1:,1:]' % self.labels[2])

    def get(self, x, y, dointerp=0):
        ix = interp(x, self.x, np.arange(len(self.x)))
        iy = interp(y, self.y, np.arange(len(self.y)))
        if not dointerp:  # JUST GET NEAREST
            #ix = searchsorted(self.x, x)
            #iy = searchsorted(self.y, y)
            ix = roundint(ix)
            iy = roundint(iy)
            z = self.z[ix, iy]
        else:
            z = bilin2(iy, ix, self.z)
        return z


class Cat2D(object):
    def __init__(self, filename='', dir="", silent=0, labels='x y z'.split()):
        if len(labels) == 2:
            labels.append('z')
        self.labels = labels
        if filename:
            if filename[-1] == '+':
                filename = filename[:-1]
            self.data = loaddata(filename, dir)
            self.assigndata()

    def assigndata(self):
        exec('self.%s = self.x = self.data[0,1:]' % self.labels[0])
        exec('self.%s = self.y = self.data[1:,0]' % self.labels[1])
        exec('self.%s = self.z = self.data[1:,1:]' % self.labels[2])

    def get(self, x, y, dointerp=0):
        ix = interp(x, self.x, np.arange(len(self.x)))
        iy = interp(y, self.y, np.arange(len(self.y)))
        if not dointerp:  # JUST GET NEAREST
            #ix = searchsorted(self.x, x)
            #iy = searchsorted(self.y, y)
            ix = roundint(ix)
            iy = roundint(iy)
            z = self.z[iy, ix]
        else:
            z = bilin2(ix, iy, self.z)
        return z


def loadcat2d(filename, dir="", silent=0, labels='x y z'):
    """INPUT: ARRAY w/ SORTED NUMERIC HEADERS (1ST COLUMN & 1ST ROW)
    OUTPUT: A CLASS WITH RECORDS"""
    if type(labels) == str:
        labels = labels.split()
    outclass = Cat2D(filename, dir, silent, labels)
    # outclass.z = np.transpose(outclass.z)  # NOW FLIPPING since 12/5/09
    return outclass


def savecat2d(data, x, y, filename, dir="", silent=0):
    """OUTPUT: FILE WITH data IN BODY AND x & y ALONG LEFT AND TOP"""
    x = x.np.reshape(1, len(x))
    data = np.concatenate([x, data])
    y = np.concatenate([[0], y])
    y = y.np.reshape(len(y), 1)
    data = np.concatenate([y, data], 1)
    if filename[-1] == '+':
        filename = filename[:-1]
    savedata(data, filename, dir, header='.')


def savecat2d_xyflip(data, x, y, filename, dir="", silent=0):
    """OUTPUT: FILE WITH data IN BODY AND x & y ALONG LEFT AND TOP"""
    #y = y[NewAxis, :]
    y = np.reshape(y, (1, len(y)))
    data = np.concatenate([y, data])
    x = np.concatenate([[0], x])
    #x = x[:, NewAxis]
    x = np.reshape(x, (len(x), 1))
    data = np.concatenate([x, data], 1)
    if filename[-1] != '+':
        filename += '+'
    #savedata(data, filename)
    savedata1(data, filename, dir)


def savedata1d(data, filename, dir="./", format='%6.5e ', header=""):
    fout = open(filename, 'w')
    for datum in data:
        fout.write('%d\n' % datum)
    fout.close()


def loadvars(filename, dir="", silent=0):
    """INPUT: CATALOG w/ LABELED COLUMNS
    OUTPUT: A STRING WHICH WHEN EXECUTED WILL LOAD DATA INTO VARIABLES WITH NAMES THE SAME AS COLUMNS
    >>> exec(loadvars('file.cat'))
    NOTE: DATA IS ALSO SAVED IN ARRAY data"""
    global data, labels, labelstr
    if filename[-1] != '+':
        filename += '+'
    data = loaddata(filename, dir, silent)
    labels = header[-1][1:].split()
    labelstr = string.join(labels, ',')
    print(labelstr + ' = data')
    # STRING TO BE EXECUTED AFTER EXIT
    return 'from coeio import data,labels,labelstr\n' + labelstr + ' = data'
    # return 'from coeio import data\n' + labelstr + ' = data'  # STRING TO BE EXECUTED AFTER EXIT


class VarsClass(object):
    def __init__(self, filename='', dir="", silent=0, labels='', labelheader='', headlines=0, loadheader=0):
        self.header = ''
        if filename:
            if strend(filename, '.fits'):  # FITS TABLE
                self.name = filename
                filename = dirfile(filename, dir)
                hdulist = pyfits.open(filename)
                self.labels = hdulist[1].columns.names
                tbdata = hdulist[1].data
                self.labels = labels or self.labels
                for label in self.labels:
                    print(label)
                    print(tbdata.field(label)[:5])
                    exec("self.%s = np.array(tbdata.field('%s'), 'f')" %
                         (label, label))
                    print(self.get(label)[:5])
                self.updatedata()
            elif machinereadable(filename, dir):
                #self = loadmachine(filename, dir, silent)
                self2 = loadmachine(filename, dir, silent)
                self.labels = self2.labels[:]
                for label in self.labels:
                    exec('self.%s = self2.%s[:]' % (label, label))
                self.name = filename
                # self.header = '' # for now...
            else:
                if filename[-1] != '+':
                    filename += '+'
                self.name = filename[:-1]
                self.data = loaddata(filename, dir, silent, headlines)
                # NOTE header IS global, GETS READ IN BY loaddata
                if loadheader:
                    self.header = header or labelheader
                if header:
                    labelheader = labelheader or header[-1][1:]
                self.labels = labels or labelheader.split()
                self.assigndata()
        else:
            self.name = ''
            self.labels = []
        self.descriptions = {}
        self.units = {}
        self.notes = []

    def assigndata(self):
        for iii in range(len(self.labels)):
            label = self.labels[iii]
            try:
                exec('self.%s = self.data[iii]' % label)
            except:
                print('BAD LABEL NAME:', label)

    def copy(self):
        # return copy.deepcopy(self)
        selfcopy = VarsClass()
        selfcopy.labels = self.labels[:]
        selfcopy.data = self.updateddata()
        selfcopy.assigndata()
        selfcopy.descriptions = self.descriptions
        selfcopy.units = self.units
        selfcopy.notes = self.notes
        selfcopy.header = self.header
        return selfcopy

    def updateddata(self):
        selflabelstr = ''
        for label in self.labels:
            # if label <> 'flags':
            selflabelstr += 'self.' + label + ', '
            # print label
            #exec('print self.%s') % label
            #exec('print type(self.%s)') % label
            #exec('print self.%s.shape') % label
            # print
        selflabelstr = selflabelstr[:-2]
        # print 'x', selflabelstr, 'x'
        #data1 = np.array([self.id, self.area])
        # print np.array([self.id, self.area])
        #s = 'data3 = np.array([%s])' % selflabelstr
        # print s
        # exec(s)
        # print data1
        # print 'data3 = np.array([%s])' % selflabelstr
        data3 = np.array([selflabelstr])
        return data3

    def updatedata(self):
        self.data = self.updateddata()

    def len(self):
        if self.labels:
            x = self.get(self.labels[0])
            l = len(x)
            try:
                l = l[:-1]
            except:
                pass
        else:
            l = 0
        return l

    def subset(self, good):
        #selfcopy = self.copy()
        # if len(self.id) <> len(good):
        # print "VarsClass: SUBSET CANNOT BE CREATED: good LENGTH = %d, data LENGTH = %d" % (len(self.id), len(good))
        if self.len() != len(good):
            print("VarsClass: SUBSET CANNOT BE CREATED: good LENGTH = %d, data LENGTH = %d" % (
                self.len(), len(good)))
        else:
            selfcopy = self.copy()
            data = self.updateddata()
            # print data.shape
            # print total(good), '/', len(good)
            selfcopy.data = np.compress(good, data)
            # print selfcopy.data.shape
            selfcopy.assigndata()
            # PRESERVE UN-UPDATED DATA ARRAY
            selfcopy.data = np.compress(good, self.data)
            selfcopy.taken = np.compress(good, np.arange(self.len()))
            selfcopy.good = good
            return selfcopy

    def between(self, lo, labels, hi):
        """labels = list of labels or just one label"""
        if type(labels) == list:
            exec('good = between(lo, self.%s, hi)' % labels[0])
            for label in labels[1:]:
                exec('good = good * between(lo, self.%s, hi)' % label)
        else:
            exec('good = between(lo, self.%s, hi)' % labels)
        self.good = good
        return self.subset(good)

    def take(self, indices):
        indices = indices.astype(int)
        sub = VarsClass()
        sub.labels = self.labels[:]
        sub.taken = sub.takeind = indices
        sub.data = take(self.updateddata(), indices, 1)
        sh = sub.data.shape
        if len(sh) == 3:
            sub.data = np.reshape(sub.data, sh[:2])
        sub.assigndata()
        return sub

    def put(self, label, indices, values):
        #exec('put(self.%s, indices, values)' % label)
        exec('x = self.%s.copy()' % label)
        put(x, indices, values)
        exec('self.%s = x' % label)

    def takeid(self, id, idlabel='id'):
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        i = argmin(abs(selfid - id))
        if selfid[i] != id:
            print("PROBLEM! ID %d NOT FOUND IN takeid" % id)
            return None
        else:
            return self.take(np.array([i]))

    def putid(self, label, id, value, idlabel='id'):
        # print "putid UNTESTED!!"  -- STILL TRUE
        # print "(Hit Enter to continue)"
        # pause()
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        i = argmin(abs(selfid - id))
        if selfid[i] != id:
            print("PROBLEM! ID %d NOT FOUND IN putid" % id)
            return None
        else:
            exec('x = self.%s.copy()' % label)
            put(x, i, value)
            exec('self.%s = x' % label)
        # print self.takeid(id).get(label)

    def takeids(self, ids, idlabel='id'):
        # selfid = self.id.astype(int) # [6 4 5]
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        indexlist = np.zeros(max(selfid) + 1, int) - 1
        put(indexlist, selfid, np.arange(len(selfid)))  # [- - - - 1 2 0]
        # self.good = np.greater(selfid, -1)  # TOTALLY WRONG!  USED IN bpzphist
        # ids = [4 6]  ->  indices = [1 0]
        indices = take(indexlist, np.array(ids).astype(int))
        # print type(indices[0])
        goodindices = np.compress(np.greater(indices, -1), indices)
        good = np.zeros(self.len(), int)
        # print 'takeids'
        good = good.astype(int)
        goodindices = goodindices.astype(int)
        # print type(good[0]) #good.type()
        # print type(goodindices[0])  #goodindices.type()
        # pause()
        put(good, goodindices, 1)
        self.good = good
        if -1 in indices:
            print("PROBLEM! NOT ALL IDS FOUND IN takeids!")
            print(np.compress(np.less(indices, 0), ids))
        return self.take(indices)

    def putids(self, label, ids, values, idlabel='id', rep=True):
        # print "putids UNTESTED!!"
        # print "putids not fully tested"
        # print "(Hit Enter to continue)"
        # pause()
        # selfid = self.id.astype(int) # [6 4 5]
        # Given selfid, at ids, place values
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        maxselfid = max(selfid)
        #idstochange = set(ids)
        exec('x = self.%s.copy()' % label)
        idchecklist = selfid.copy()
        done = False
        while not done:  # len(idstochange):
            indexlist = np.zeros(maxselfid + 1, int) - 1
            put(indexlist, idchecklist, np.arange(self.len()))  # [- - - - 1 2 0]
            # ids = [4 6]  ->  indices = [1 0]
            indices = take(indexlist, np.array(ids).astype(int))
            if (-1 in indices) and (rep < 2):
                print("PROBLEM! NOT ALL IDS FOUND IN putids!")
                print(np.compress(np.less(indices, 0), ids))
            if singlevalue(values):
                values = np.zeros(self.len(), float) + values
            put(x, indices, values)
            put(idchecklist, indices, 0)
            if rep:  # Repeat if necessary
                #idstochange = set(x) & set(ids)
                #done = total(idsdone) == self.len()
                done = total(idchecklist) == 0
                rep += 1
            else:
                #idstochange = []
                done = 1
            if 0:
                print(x)  # [:10]
                print(ids)  # [:10]
                print(indexlist)  # [:10]
                # print idsdone[:10]
                print(idchecklist)  # [:10]
                print(len(x))
                print(len(x) - len(np.compress(idchecklist, idchecklist)))
                # print len(idstochange)
                pause()
        exec('self.%s = x' % label)

    def takecids(self, ids, idlabel='id'):  # only take common ids
        # selfid = self.id.astype(int) # [6 4 5]
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        n = max((max(selfid), max(ids)))
        indexlist = np.zeros(n + 1, int)
        #indexlist = np.zeros(max(selfid)+1)
        put(indexlist, selfid, np.arange(len(selfid)) + 1)  # [- - - - 1 2 0]
        # ids = [4 6]  ->  indices = [1 0]
        indices = take(indexlist, np.array(ids).astype(int))
        indices = np.compress(indices, indices - 1)
        goodindices = np.compress(np.greater(indices, -1), indices)
        good = np.zeros(self.len(), int)
        put(good, goodindices, 1)
        self.good = good
        return self.take(indices)

    def removeids(self, ids, idlabel='id'):
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        if singlevalue(ids):
            ids = [ids]
        # newids = set(selfid) - set(ids)  # SCREWS UP ORDER!
        #newids = list(newids)
        newids = invertselection(ids, selfid)
        return self.takeids(newids)

    def get(self, label, orelse=None):
        if label in self.labels:
            out = getattr(self, label)
        else:
            out = orelse
        return out

    def set(self, label, data):
        if singlevalue(data):
            data = np.zeros(self.len(), float) + data
        exec('self.%s = data' % label)

    def add(self, label, data):
        if 1:  # self.labels:
            if singlevalue(data):
                if self.len():
                    data = np.zeros(self.len(), float) + data
                else:
                    data = np.array([float(data)])
            elif self.len() and (len(data) != self.len()):
                print('WARNING!! in loadvarswithclass.add:')
                print('len(%s) = %d BUT len(id) = %d' %
                      (label, len(data), self.len()))
                print()
        self.labels.append(label)
        exec('self.%s = data.astype(float)' % label)

    def assign(self, label, data):
        if label in self.labels:
            self.set(label, data)
        else:
            self.add(label, data)

    def append(self, self2, over=0):
        # APPENDS THE CATALOG self2 TO self
        labels = self.labels[:]
        labels.sort()
        labels2 = self2.labels[:]
        labels2.sort()
        if labels != labels2:
            print("ERROR in loadvarswithclass.append: labels don't match")
        else:
            if over:  # OVERWRITE OLD OBJECTS WITH NEW WHERE IDS ARE THE SAME
                commonids = common(self.id, self2.id)
                if commonids:
                    selfuniqueids = invertselection(commonids, self.id)
                    self = self.takeids(selfuniqueids)
            for label in self.labels:
                exec('self.%s = np.concatenate((self.get(label), self2.get(label)))' % label)
            self.updatedata()
        return self

    def merge(self, self2, labels=None, replace=0):
        # self2 HAS NEW INFO (LABELS) TO ADD TO self
        if 'id' in self.labels:
            if 'id' in self2.labels:
                self2 = self2.takeids(self.id)
        labels = labels or self2.labels
        for label in labels:
            if label not in self.labels:
                self.add(label, self2.get(label))
            elif replace:
                exec('self.%s = self2.%s' % (label, label))

    def sort(self, label):  # label could also be an array
        if type(label) == str:
            if (label == 'random') and ('random' not in self.labels):
                SI = argsort(random(self.len()))
            else:
                if label[0] == '-':  # Ex.: -odds
                    label = label[1:]
                    reverse = 1
                else:
                    reverse = 0
                exec('SI = argsort(self.%s)' % label)
                if reverse:
                    SI = SI[::-1]
        else:
            SI = argsort(label)  # label contains an array
        self.updatedata()
        self.data = take(self.data, SI, 1)
        self.assigndata()

    def findmatches(self, searchcat1, dtol=4):
        """Finds matches for self in searchcat1
        match distances within dtol, but may not always be closest
        see also findmatches2"""
        matchids = []
        dists = []
        searchcat = searchcat1.copy()
        # searchcat.sort('x')
        if 'dtol' in self.labels:
            dtol = self.dtol
        else:
            dtol = dtol * np.ones(self.len())
        for i in range(self.len()):
            if not (i % 100):
                print("%d / %d" % (i, self.len()))
            # matchid, dist = findmatch(searchcat.x, searchcat.y, self.x[i], self.y[i], dtol=dtol[i], silent=1, returndist=1, xsorted=1)  # silent=2*(i<>38)-1
            matchid, dist = findmatch(
                searchcat.x, searchcat.y, self.x[i], self.y[i], dtol=dtol[i], silent=1, returndist=1, xsorted=0)  # silent=2*(i<>38)-1
# print self.x[i], self.y[i], matchid,
# if matchid < self.len():
# print searchcat.id[matchid], searchcat.x[matchid], searchcat.y[matchid]
# else:
# print
# pause()
            matchids.append(matchid)
            dists.append(dist)
        matchids = np.array(matchids)
        dists = np.array(dists)
        matchids = where(equal(matchids, searchcat.len()), -1, matchids)
        self.assign('matchid', matchids)
        self.assign('dist', dists)

    def findmatches2(self, searchcat, dtol=0):
        """Finds closest matches for self within searchcat"""
        i, d = findmatches2(searchcat.x, searchcat.y, self.x, self.y)
        if dtol:
            i = where(np.less(d, dtol), i, -1)
        self.assign('matchi', i)
        self.assign('dist', d)

    def rename(self, oldlabel, newlabel):
        self.set(newlabel, self.get(oldlabel))
        i = self.labels.index(oldlabel)
        self.labels[i] = newlabel
        if self.descriptions:
            if oldlabel in list(self.descriptions.keys()):
                self.descriptions[newlabel] = self.descriptions[oldlabel]
        if self.units:
            if oldlabel in list(self.units.keys()):
                self.units[newlabel] = self.units[oldlabel]

    def save(self, name='', dir="", header='', format='', labels=1, pf=0, maxy=300, machine=0, silent=0):
        if type(labels) == list:
            self.labels = labels
        labels = labels and self.labels  # if labels then self.labels, else 0
        name = name or self.name  # if name then name, else self.name
        header = header or self.header  # if header then header, else self.header
        savedata(self.updateddata(), name + '+', dir=dir, labels=labels, header=header, format=format, pf=pf, maxy=maxy,
                 machine=machine, descriptions=self.descriptions, units=self.units, notes=self.notes, silent=silent)

    def savefitstable(self, name='', header='', format={}, labels=1, overwrite=1):  # FITS TABLE
        # if name then name, else self.name
        name = name or recapfile(self.name, 'fits')
        name = capfile(name, 'fits')  # IF WAS name (PASSED IN) NEED TO capfile
        if (not overwrite) and os.path.exists(name):
            print(name, 'ALREADY EXISTS, AND YOU TOLD ME NOT TO OVERWRITE IT')
        else:
            units = self.units
            header = header or self.header  # if header then header, else self.header
            if type(labels) == list:
                self.labels = labels
            labels = labels and self.labels  # if labels then self.labels, else 0
            collist = []
            for label in self.labels:
                a = self.get(label)
                if not pyfitsusesnumpy:
                    a = numarray.array(a)
                if label in list(units.keys()):
                    col = pyfits.Column(name=label, format=format.get(
                        label, 'E'), unit=units[label], array=a)
                else:
                    col = pyfits.Column(
                        name=label, format=format.get(label, 'E'), array=a)
                collist.append(col)
            cols = pyfits.ColDefs(collist)
            tbhdu = pyfits.new_table(cols)
            if not self.descriptions:
                delfile(name)
                tbhdu.writeto(name)
            else:
                hdu = pyfits.PrimaryHDU()
                hdulist = pyfits.HDUList(hdu)
                hdulist.append(tbhdu)
                prihdr = hdulist[1].header
                descriptions = self.descriptions
                for ilabel in range(len(labels)):
                    label = labels[ilabel]
                    if label in list(descriptions.keys()):
                        description = descriptions[label]
                        if len(description) < 48:
                            description1 = description
                            description2 = ''
                        else:
                            i = string.rfind(description[:45], ' ')
                            description1 = description[:i] + '...'
                            description2 = '...' + description[i + 1:]
                        prihdr.update('TTYPE%d' %
                                      (ilabel + 1), label, description1)
                        if description2:
                            prihdr.update('TFORM%d' % (ilabel + 1),
                                          format.get(label, 'E'), description2)
                for inote in range(len(self.notes)):
                    words = self.notes[inote].split('\n')
                    for iword in range(len(words)):
                        word = words[iword]
                        if word:
                            if iword == 0:
                                prihdr.add_comment(
                                    '(%d) %s' % (inote + 1, word))
                            else:
                                prihdr.add_comment('    %s' % word)
                                # prihdr.add_blank(word)
                headlines = header.split('\n')
                for headline in headlines:
                    if headline:
                        key, value = headline.split('\t')
                        prihdr.update(key, value)
                hdulist.writeto(name)

    def pr(self, header='', more=True):  # print
        self.save('tmp.cat', silent=True, header=header)
        if more:
            os.system('cat tmp.cat | more')
        else:
            os.system('cat tmp.cat')
        os.remove('tmp.cat')
# def takecids(self, ids):
##         selfid = self.id.astype(int)
##         ids = ids.astype(int)
##         n = max((max(selfid), max(ids)))
##         takeme1 = np.zeros(n+1)
##         takeme2 = np.zeros(n+1)
##         put(takeme1, selfid, 1)
##         put(takeme2, ids, 1)
##         takeme = takeme1 * takeme2
##         takeme = take(takeme, selfid)
# return self.subset(takeme)
# def labelstr(self):
# return string.join(labels, ', ')[:-2]
        # FORGET THIS!  JUST USE copy.deepcopy
##         selfcopy = VarsClass()
##         selfcopy.data = self.data[:]
##         selfcopy.labels = self.labels.copy()


def loadvarswithclass(filename, dir="", silent=0, labels='', header='', headlines=0):
    """INPUT: CATALOG w/ LABELED COLUMNS
    OUTPUT: A CLASS WITH RECORDS NAMED AFTER EACH COLUMN
    >>> mybpz = loadvars('my.bpz')
    >>> mybpz.id
    >>> mybpz.data -- ARRAY WITH ALL DATA"""
    outclass = VarsClass(filename, dir, silent, labels, header, headlines)
    # outclass.assigndata()
    return outclass


loadcat = loadvarswithclass

# def loadcat(filename, dir="", silent=0):


def loadimcat(filename, dir="", silent=0):
    """LOADS A CATALOG CREATED BY IMCAT
    STORES VARIABLES IN A DICTIONARY OF ARRAYS!"""

    infile = dirfile(filename, dir)
    if not silent:
        print("Loading ", infile, "...\n")

    fin = open(infile, 'r')
    sin = fin.readlines()
    fin.close

    headlines = 0
    while sin[headlines][0] == '#':
        headlines = headlines + 1
    names = sin[headlines - 1][1:].split()

    sin = sin[headlines:]  # REMOVE HEADLINES
    nx = len(names)
    ny = len(sin)
    data = FltArr(ny, nx)

    for iy in range(ny):
        ss = sin[iy].split()
        for ix in range(nx):
            data[iy, ix] = float(ss[ix])

    cat = {}
    for i in range(nx):
        cat[names[i]] = data[:, i]

    return cat


def savedict(dict, filename, dir="", silent=0):
    """SAVES A DICTIONARY OF STRINGS"""
    outfile = dirfile(filename, dir)
    fout = open(outfile, 'w')
    for key in list(dict.keys()):
        fout.write('%s %s\n' % (key, dict[key]))
    fout.close()


def savefile(lines, filename, dir="", silent=0):
    """SAVES lines TO filename"""
    outfile = dirfile(filename, dir)
    fout = open(outfile, 'w')
    for line in lines:
        if line[-1] != '\n':
            line += '\n'
        fout.write(line)
    fout.close()


# def savecat(cat, filename, dir="./", silent=0):
def saveimcat(cat, filename, dir="./", silent=0):
    # DOESN'T WORK RIGHT YET!!!  HEADER INCOMPLETE.
    """SAVES A DICTIONARY OF 1-D ARRAYS AS AN IMCAT CATALOGUE"""
    outfile = dirfile(filename, dir)
    fout = open(outfile, 'w')
    fout.write(
        "# IMCAT format catalogue file -- edit with 'lc' or my Python routines\n")

    # COLUMN HEADERS
    fout.write("#")
    for key in list(cat.keys()):
        fout.write(string.rjust(key, 15))
    fout.write("\n")

    n = len(cat[key])
    for i in range(n):
        fout.write(" ")
        keys = list(cat.keys())
        keys.sort()
        for key in keys:
            x = cat[key][i]
            if (x - int(x)):
                fout.write("%15.5f" % x)
            else:
                fout.write("%15d" % x)
        fout.write("\n")

    fout.close()


def prunecols(infile, cols, outfile, separator=" "):
    """TAKES CERTAIN COLUMNS FROM infile AND OUTPUTS THEM TO OUTFILE
    COLUMN NUMBERING STARTS AT 1!
    ALSO AVAILABLE AS STANDALONE PROGRAM prunecols.py"""
    fin = open(infile, 'r')
    sin = fin.readlines()
    fin.close()

    fout = open(outfile, 'w')
    for line in sin:
        print(line)
        line = string.strip(line)
        words = line.split(separator)
        print(words)
        for col in cols:
            fout.write(words[col - 1] + separator)
        fout.write("\n")
    fout.close()


#################################
# SExtractor/SExSeg CATALOGS / CONFIGURATION FILES

class SExSegParamsClass(object):
    def __init__(self, filename='', dir="", silent=0, headlines=0):
        # CONFIGURATION
        #   configkeys -- PARAMETERS IN ORDER
        #   config[key] -- VALUE
        #   comments[key] -- COMMENTS (IF ANY)
        # PARAMETERS
        #   params -- PARAMETERS IN ORDER
        #   comments[key] -- COMMENTS (IF ANY)
        self.name = filename
        self.configkeys = []
        self.config = {}
        self.comments = {}
        self.params = []
        txt = loadfile(filename, dir, silent)
        for line in txt:
            if string.strip(line) and (line[:1] != '#'):
                # READ FIRST WORD AND DISCARD IT FROM line
                key = line.split()[0]
                line = line[len(key):]
                # READ COMMENT AND DISCARD IT FROM line
                i = string.find(line, '#')
                if i > -1:
                    self.comments[key] = line[i:]
                    line = line[:i]
                else:
                    self.comments[key] = ''
                # IF ANYTHING IS LEFT, IT'S THE VALUE, AND YOU'VE BEEN READING FROM THE CONFIGURATION SECTION
                # OTHERWISE IT WAS A PARAMETER (TO BE INCLUDED IN THE SEXTRACTOR CATALOG)
                line = string.strip(line)
                if string.strip(line):  # CONFIGURATION
                    self.configkeys.append(key)
                    self.config[key] = line
                else:  # PARAMETERS
                    self.params.append(key)

    def save(self, name='', header=''):
        name = name or self.name  # if name then name, else self.name
        # QUICK CHECK: IF ANY CONFIG PARAMS WERE ADDED TO THE DICT, BUT NOT TO THE LIST:
        for key in list(self.config.keys()):
            if key not in self.configkeys:
                self.configkeys.append(key)
        # OKAY...
        fout = open(name, 'w')
        # fout.write('#######################################\n')
        # fout.write('# CONFIGURATION\n')
        # fout.write('\n')
        fout.write('# ----- CONFIGURATION -----\n')
        for key in self.configkeys:
            line = ''
            line += string.ljust(key, 20) + ' '
            value = self.config[key]
            comment = self.comments[key]
            if not comment:
                line += value
            else:
                line += string.ljust(value, 20) + ' '
                line += comment
            line += '\n'
            fout.write(line)
        fout.write('\n')
        # fout.write('#######################################\n')
        # fout.write('# PARAMETERS\n')
        # fout.write('\n')
        fout.write('# ----- PARAMETERS -----\n')
        for param in self.params:
            line = ''
            comment = self.comments[param]
            if not comment:
                line += param
            else:
                line += string.ljust(param, 20) + ' '
                line += comment
            line += '\n'
            fout.write(line)
        fout.close()

    def merge(self, filename='', dir="", silent=0, headlines=0):
        self2 = loadsexsegconfig(filename, dir, silent, headlines)
        for key in self2.configkeys:
            self.config[key] = self2.config[key]
        if self2.params:
            self.params = self2.params
        for key in list(self2.comments.keys()):
            if self2.comments[key]:
                self.comments[key] = self2.comments[key]


def loadsexsegconfig(filename='', dir="", silent=0, headlines=0):
    return SExSegParamsClass(filename, dir, silent, headlines)


def loadsexcat(infile, purge=1, maxflags=8, minfwhm=1, minrf=0, maxmag=99, magname="MAG_AUTO", ma1name='APER', silent=0, dir=''):
    """>>> exec(loadsexcat('sexfile.cat'<, ...>))
    LOADS A SEXTRACTOR CATALOG DIRECTLY INTO VARIABLES x, y, fwhm, etc.
    PURGES (DEFAULT) ACCORDING TO flags, fwhm, mag (AS SET)
    NOW MODELED AFTER loadvars -- OUTPUT STRING NEEDS TO BE EXECUTED, THEN ALL VARIABLES ARE LOADED
    NOW TAKES ON *ALL* VARIABLES, AND ADJUSTS NAMES ACCORDINGLY"""
    # outdata is a list of arrays (most are 1-D, but some (mag_aper) are 2-D)
    #global data, labels, labelstr, params, paramstr, outdata
    global params, paramstr, data, fullparamnames
    #global id, x, y, fwhm, mag, magerr, magauto, magerrauto, magautoerr, flags, a, b, theta, stellarity, rf, ell, rk, assoc, magaper, magapererr, magerraper, ma1, ema1, mb, emb, cl, flag, xpeak, ypeak, area

    # Txitxo variable translation:
    # cl = stellarity
    # flag = flags
    # ma1 = mag_aper   ema1 = error for mag_aper
    # mb  = mag_auto   emb  = error for mag_auto

    infile = join(dir, infile)
    if not silent:
        print("LOADING SExtractor catalog " + infile, end=' ')

    # req = {'fwhm': 1, 'mag': 99, 'flags': 4}  # REQUIREMENTS FOR DATA TO BE INCLUDED (NOT PURGED)
    req = {}
    req['FLAGS'] = maxflags
    req['FWHM'] = minfwhm
    req['MAG'] = maxmag
    req['RF'] = minrf

    if magname:
        magname = string.upper(magname)
        if magname[:4] != 'MAG_':
            magname = 'MAG_' + magname
        #magerrname = 'MAGERR_' + magname[-4:]
        magerrname = 'MAGERR_' + magname[4:]
    else:
        magerrname = ''
        ma1name = ''

    sin = loadfile(infile, silent=1)

    # REMOVE HEADLINES FROM sin, CREATE header
    header = []
    while sin[0][0] == "#":
        if sin[0][1] != '#':
            header.append(sin[0])  # Only add lines beginning with single #
        sin = sin[1:]

    nx = len(sin[0].split())
    ny = len(sin)
    data = FltArr(ny, nx)

    for iy in range(ny):
        ss = sin[iy].split()
        for ix in range(nx):
            try:
                data[iy, ix] = float(ss[ix])
            except:
                print(iy, ix, nx)
                print(ss)
                die()

    data = np.transpose(data)
    paramcol = {}
    params = []

    flags = None
    fwhm = None
    mag = None
    rf = None

    # print 'TRYING NEW MAG ASSIGNMENT...'
    lastcol = 0  # COLUMN OF PREVIOUS PARAM
    lastparam = ''
    params = []
    fullparamnames = []
    for headline in header:
        # ['#', '15', 'X_IMAGE', 'Object position along x', '[pixel]']
        ss = headline.split()
        if len(ss) == 1:
            break
        # 15  -- DON'T SUBTRACT 1 FROM col!  DON'T WANT A 0 COLUMN!  FACILITATES DATA DISTRIBUTION
        col = int(ss[1])
        ncols = col - lastcol
        param = ss[2]    # "X_IMAGE"
        fullparamnames.append(param)
        if param[-1] == ']':
            param = param.split('[')[0]
        if param[:4] == "MAG_":  # MAG_AUTO or MAG_APER but not MAGERR_AUTO
            # if (param == magname) or not magname or 'MAG' not in paramcol.keys():  # magname IF YOU ONLY WANT MAG_AUTO (DEFAULT)
            if (param == magname) or not magname:  # magname IF YOU ONLY WANT MAG_AUTO (DEFAULT)
                magname = param
                param = "MAG"
        if param[:7] == "MAGERR_":  # MAGERR_AUTO or MAGERR_APER
            # if (param == magerrname) or not magerrname or 'MAG' not in paramcol.keys():  # magname IF YOU ONLY WANT MAG_AUTO (DEFAULT)
            # magname IF YOU ONLY WANT MAG_AUTO (DEFAULT)
            if (param == magerrname) or not magerrname:
                magerrname = param
                param = "MAGERR"
        if param[-6:] == "_IMAGE":  # TRUNCATE "_IMAGE"
            param = param[:-6]
        if param in ["FLAGS", "IMAFLAGS_ISO"]:
            if not flags:
                flags = np.ravel(data[col - 1]).astype(int)
                param = "FLAGS"
            else:
                # "FLAGS" OR "IMAFLAGS_ISO"
                flags = bitwise_or(flags, np.ravel(data[col - 1]).astype(int))
                param = ''
                lastcol += 1
# if (param == "FLAGS") and paramcol.has_key("FLAGS"):
# param = "SHWAG"  # "IMAFLAGS_ISO" (THE REAL FLAGS) HAVE ALREADY BEEN FOUND
# if param == "IMAFLAGS_ISO":  # FLAGS OR-ED WITH FLAG MAP IMAGE
##          param = "FLAGS"
        #paramcol[param] = col
        # if vector > 1
        # ASSIGN COLUMN(S), NOW THAT WE KNOW HOW MANY THERE ARE
        if param != lastparam and param:
            if lastcol:
                paramcol[lastparam] = np.arange(ncols) + lastcol
            lastcol = col
            lastparam = param
            params.append(param)
        # print params

# IN CASE WE ENDED ON AN ARRAY (MAG_APER[4]) -- TAKEN CARE OF BELOW?
# if param == lastparam:
# if lastcol:
##             paramcol[lastparam] = np.arange(ncols) + lastcol
##         lastcol = col
##         lastparam = param
# params.append(param)

    # print len(params)

    bigparamnames = params[:]
    paramstr = string.join(params, ',')
    # ASSIGN LAST COLUMN(S)
    ncols = nx - lastcol + 1
    paramcol[lastparam] = np.arange(ncols) + lastcol

    col = paramcol.get("FWHM")
    #fwhm = col and np.ravel(data[col-1])
    if col.any():
        fwhm = np.ravel(data[col - 1])
    col = paramcol.get("FLUX_RADIUS")
    #rf = col and np.ravel(data[col-1])
    if col.any():
        rf = np.ravel(data[col - 1])
    col = paramcol.get("MAG")
    #mag = col and np.ravel(data[col-1])
    if col.any():
        mag = np.ravel(data[col - 1])

    good = np.ones(ny)
    if not silent:
        print(sum(good), end=' ')
    if purge:
        if "FLAGS" in req and (flags != None):
            good = np.less(flags, maxflags)
        if "FWHM" in req and (fwhm != None):
            good = good * np.greater(fwhm, minfwhm)
        if "RF" in req and (rf != None):
            good = good * np.greater(rf, minrf)
        if "MAG" in req and (mag != None):
            good = good * np.less(mag, maxmag)

    if not silent:
        print(sum(good))

    if purge and not alltrue(good):
        data = np.compress(good, data)
        if (flags != None):
            flags = np.compress(good, flags)
        if (mag != None):
            mag = np.compress(good, mag)
        if (fwhm != None):
            fwhm = np.compress(good, fwhm)
        if (rf != None):
            rf = np.compress(good, rf)

    outdata = []
    #params = paramcol.keys()
    # RENAME params
    paramtranslate = {'NUMBER': 'id', 'CLASS_STAR': 'stellarity',
                      'KRON_RADIUS': 'rk', 'FLUX_RADIUS': 'rf', 'ISOAREA': 'area'}
    for ii in range(len(params)):
        param = params[ii]
        # CHANGE IT IF IN DICTIONARY, OTHERWISE LEAVE IT ALONE
        param = paramtranslate.get(param, param)
        param = string.replace(param, '_IMAGE', '')
        param = string.replace(param, 'PROFILE', 'PROF')
        param = string.lower(param)
        param = string.replace(param, '_', '')
        #param = string.replace(param, 'magerr', 'dmag')
        # if param in ['a', 'b']:
        #    param = string.upper(param)
        params[ii] = param

    # print params
    # for kk in bigparamnames: #paramcol.keys():
    for ii in range(len(bigparamnames)):  # paramcol.keys():
        pp = params[ii]
        # print
        # print pp
        if pp in ['flags', 'fwhm', 'rf', 'mag']:
            #exec('print type(%s)' % pp)
            #exec('print '+pp)
            exec('outdata.append(%s)' % pp)
            # outdata.append(flags)
        else:
            kk = bigparamnames[ii]
            col = paramcol[kk]
            if len(col) == 1:
                #exec(kk + '= data[col-1]')
                # print data[col-1]
                # print shape(data[col-1])
                # print type(data[col-1])
                outdata.append(np.ravel(data[col - 1]))
            else:
                #exec(kk + '= take(data, col-1)')
                outdata.append(np.ravel(take(data, col - 1)))

    paramstr = string.join(params, ',')
    #exec(paramstr + ' = outdata')

    # CALCULATE ell (IF NOT CALCULATED ALREADY)
    # print params
    # print params.index('a')
    # print len(outdata)
    if 'ell' not in params:
        if 'a' in params and 'b' in params:
            a = outdata[params.index('a')]
            b = outdata[params.index('b')]
            try:
                ell = 1 - old_div(b, a)
            except:
                ell = a * 0.
                for ii in range(len(a)):
                    if a[ii]:
                        ell[ii] = 1 - old_div(b[ii], a[ii])
                    else:
                        ell[ii] = 99
            params.append('ell')
            paramstr += ', ell'
            outdata.append(ell)
            fullparamnames.append('ELLIPTICITY')

    # COMBINE flags & imaflagsiso
    if 'imaflagsiso' in params:
        flags = outdata[params.index('flags')]
        imaflagsiso = outdata[params.index('imaflagsiso')]
        # "FLAGS" OR "IMAFLAGS_ISO"
        flags = np.bitwise_or(flags.astype(int), imaflagsiso.astype(int))
        outdata[params.index('flags')] = flags.astype(float)

    # FOR Txitxo's photometry.py
    # print 'COMMENTED OUT photometry.py LINES...'
    photcom = '\n'
    if 'stellarity' in params:
        photcom += 'cl = stellarity\n'
    if 'flags' in params:
        photcom += 'flag = flags\n'
    if 'mag' in params:
        photcom += 'mb = mag\n'
    if 'magerr' in params:
        photcom += 'emb = np.ravel(magerr)\n'

    if ma1name:
        magtype = string.lower(ma1name.split('[')[0])
        magtype = {'profile': 'prof', 'isophotal': 'iso'}.get(magtype, magtype)
        pos = string.find(ma1name, '[')
        if pos > -1:
            ma1i = int(ma1name[pos + 1:-1])
        else:
            ma1i = 0
        if magtype == 'aper':
            photcom += 'ma1 = np.ravel(magaper[%d])\n' % ma1i
            photcom += 'ema1 = np.ravel(magerraper[%d])\n' % ma1i
        else:
            photcom += 'ma1 = np.ravel(mag%s)\n' % magtype
            photcom += 'ema1 = np.ravel(magerr%s)\n' % magtype

    data = outdata
    # print 'params:', params
    # print photcom
    #outstr = 'from coeio import params,paramstr,data,fullparamnames\n' + paramstr + ' = data' + photcom
    # print outstr
    # return outstr
    # return 'from coeio import data,labels,labelstr,params,outdata\n' + paramstr + ' = outdata' + photcom  # STRING TO BE EXECUTED AFTER EXIT
    # STRING TO BE EXECUTED AFTER EXIT
    return 'from coeio import params,paramstr,data,fullparamnames\n' + paramstr + ' = data' + photcom


def loadsexcat2(infile, purge=1, maxflags=8, minfwhm=1, minrf=0, maxmag=99, magname="MAG_AUTO", ma1name='APER', silent=0, dir=''):
    """RETURNS A VarsClass() VERSION OF THE CATALOG"""
    loadsexcat(infile, purge=purge, maxflags=maxflags, minfwhm=minfwhm, minrf=minrf,
               maxmag=maxmag, magname=magname, ma1name=ma1name, silent=silent, dir=dir)
    # LOADS infile INTO data, params...
    cat = VarsClass()
    cat.name = infile
    cat.data = data
    cat.labels = params
    cat.assigndata()
    for label in cat.labels:
        exec('cat.%s = np.ravel(cat.%s).astype(float)' % (label, label))
    #cat.flags = cat.flags[NewAxis,:]
    #cat.flags = cat.flags.astype(float)
    return cat


def loadsexdict(sexfile):
    """LOADS A SEXTRACTOR CONFIGURATION (.sex) FILE INTO A DICTIONARY
       COMMENTS NOT LOADED"""
    sextext = loadfile(sexfile)
    sexdict = {}
    for line in sextext:
        if line:
            if line[0] != '#':
                words = line.split()
                if len(words) > 1:
                    key = words[0]
                    if key[0] != '#':
                        # sexdict[words[0]] = str2num(words[1])
                        restofline = string.join(words[1:])
                        value = restofline.split('#')[0]
                        if value[0] == '$':
                            i = string.find(value, '/')
                            value = os.getenv(value[1:i]) + value[i:]
                        sexdict[key] = str2num(value)
    return sexdict


def savesexdict(sexdict, sexfile):
    """SAVES A SEXTRACTOR CONFIGURATION (.sex) FILE
    BASED ON THE sexdict DICTIONARY"""
    fout = open(sexfile, 'w')
    keys = list(sexdict.keys())
    keys.sort()
    for key in keys:
        fout.write('%s\t%s\n' % (key, sexdict[key]))
    fout.close()

#################################
# DS9 REGIONS FILES


def saveregions1(x, y, filename, coords='image', color="green", symbol="circle", size=0, width=0):
    """SAVES POSITIONS AS A ds9 REGIONS FILE"""
    fout = open(filename, 'w')
    fout.write('global color=' + color +
               ' font="helvetica 10 normal" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n')
    # fout.write("image\n")
    fout.write(coords + "\n")
    n = len(x)
    for i in range(n):
        if not size and not width:
            fout.write("%s point %s %s\n" % (symbol, x[i], y[i]))
        else:
            sout = '%s %s %s' % (symbol, x[i], y[i])
            if size:
                sout += ' %d' % size
            if width:
                sout += ' # width = %d' % width
            sout += '\n'
            fout.write(sout)
# if size:
##             fout.write("%s %6.1f %6.1f %d\n" % (symbol, x[i], y[i], size))
# else:
##             fout.write("%s point %6.1f %6.1f\n" % (symbol, x[i], y[i]))

    fout.close()


def saveregions(x, y, filename, labels=[], precision=1, coords='image', color="green", symbol="circle", size=0, width=0):
    """SAVES POSITIONS AND LABELS AS A ds9 REGIONS FILE"""
    fout = open(filename, 'w')
    fout.write('global color=' + color +
               ' font="helvetica 10 normal" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n')
    # fout.write("image\n")
    fout.write(coords + "\n")
    n = len(x)
    for i in range(n):
        if not size and not width:
            sout = '%s point %s %s' % (symbol, x[i], y[i])
        else:
            sout = '%s %s %s' % (symbol, x[i], y[i])
            if size:
                sout += ' %d' % size
            if width:
                sout += ' # width = %d' % width
        if i < len(labels):
            label = "%%.%df" % precision % labels[i]
            sout += ' # text={%s}' % label
        print(sout)
        sout += '\n'
        fout.write(sout)

    fout.close()


def savelabels(x, y, labels, filename, coords='image', color="green", symbol="circle", precision=1, fontsize=12, bold=1):
    """SAVES POSITIONS AS A ds9 REGIONS FILE"""
    if type(labels) in [int, float]:
        labels = np.arange(len(x)) + 1
    fout = open(filename, 'w')
    fout.write('global color=%s font="helvetica %d %s" select=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n' % (
        color, fontsize, ['normal', 'bold'][bold]))
    fout.write(coords + "\n")
    n = len(x)
    for i in range(n):
        label = labels[i]
        # if type(label) == int:  # IntType
        #    label = "%d" % label
        # elif type(label) == float: # FloatType
        #    label = "%%.%df" % precision % label
        label = "%%.%df" % precision % label
        fout.write("text %d %d {%s}\n" % (x[i], y[i], label))
    fout.close()


#################################
# FITS FILES

def savefits(data, filename, dir="", silent=0, xx=None, yy=None):
    """SAVES data (A 2-D ARRAY) AS filename (A .fits FILE)"""
    # THIS PROGRAM HAS FEWER OPTIONS THAN writefits IN fitsio, SO IT GETS THE JOB DONE EASILY!
    filename = capfile(filename, '.fits')
    filename = dirfile(filename, dir)
    if not silent:
        print('savefits:', filename)
    # print type(data)
    if os.path.exists(filename):
        os.remove(filename)
    # UNLESS $NUMERIX IS SET TO numpy, pyfits(v1.1b) USES NumArray
    if not pyfitsusesnumpy:
        data = numarray.array(data.tolist())
    pyfits.writeto(filename, data)
    if xx != None:
        f = pyfits.open(filename)
        hdu = f[0]
        hdu.header.update('XMIN', min(xx))
        hdu.header.update('XMAX', max(xx))
        hdu.header.update('YMIN', min(yy))
        hdu.header.update('YMAX', max(yy))
        delfile(filename, silent=True)
        hdu.writeto(filename)


def loadfits(filename, dir="", index=0):
    """READS in the data of a .fits file (filename)"""
    filename = capfile(filename, '.fits')
    filename = dirfile(filename, dir)
    if os.path.exists(filename):
        # CAN'T RETURN data WHEN USING memmap
        # THE POINTER GETS MESSED UP OR SOMETHING
        # return pyfits.open(filename, memmap=1)[0].data
        data = pyfits.open(filename)[index].data
        # UNLESS $NUMERIX IS SET TO numpy, pyfits(v1.1b) USES NumArray
        if not pyfitsusesnumpy:
            data = np.array(data)  # .tolist() ??
        return data
    else:
        print()
        print(filename, "DOESN'T EXIST")
        FILE_DOESNT_EXIST[9] = 3


def fitsrange(filename):
    """RETURNS (xmin, xmax, ymin, ymax)"""
    filename = capfile(filename, '.fits')
    f = pyfits.open(filename, memmap=1)
    header = f[0].header
    xmin = header['XMIN']
    ymin = header['YMIN']
    xmax = header['XMAX']
    ymax = header['YMAX']
    return xmin, xmax, ymin, ymax


def fitssize(filename):
    """RETURNS (ny, nx)"""
    filename = capfile(filename, '.fits')
    return pyfits.open(filename, memmap=1)[0]._dimShape()


def txt2fits(textfile, fitsfile):
    """CONVERTS A TEXT FILE DATA ARRAY TO A FITS FILE"""
    savefits(loaddata(textfile), fitsfile)

def loadpixelscale(image):
    if os.path.exists('temp.txt'):
        os.remove('temp.txt')
    print('imsize ' + capfile(image, '.fits') + ' > temp.txt')
    os.system('imsize ' + capfile(image, '.fits') + ' > temp.txt')
    s = loadfile('temp.txt')[0]
    if string.find(s, '/pix') == -1:
        print('PIXEL SCALE NOT GIVEN IN IMAGE HEADER OF', capfile(image, '.fits'))
        pixelscale = 0
    else:
        s = s.split('/pix')[0]
        s = s.split('/')[1]
        pixelscale = float(s[:-1])
    os.remove('temp.txt')
    return pixelscale
