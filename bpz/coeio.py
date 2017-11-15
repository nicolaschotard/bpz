from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# Automatically adapted for numpy Jun 0, 2006


from past.utils import old_div
from . import coetools
from . import MLab_coe
from . import coeio
import string

import pyfits
import os
import sys
import numpy as np


def recapfile(name, ext):
    """CHANGE FILENAME EXTENSION"""
    if ext[0] != '.':
        ext = '.' + ext
    i = name.rfind(".")
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
        i = name.rfind('.')
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
    value = key = None
    oldkey = ""
    key = ""
    list.append('')  # EXTRA ELEMENT SO WE COME BACK AND ASSIGN THE LAST VALUE
    while i < len(list):
        if coetools.striskey(list[i]) or not list[i]:  # (or LAST VALUE)
            if key:  # ASSIGN VALUES TO OLD KEY
                if value:
                    print(value)
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
                    value.append(coetools.str2num(list[i]))
                else:
                    value = [coetools.str2num(list[i])]
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
        filename = os.path.join(coetools.home, filename[2:])
    else:
        if dir[0:2] == '~/':
            dir = os.path.join(coetools.home, dir[2:])
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


def loaddict(filename, dir="", silent=0):
    lines = loadfile(filename, dir, silent)
    dict = {}
    for line in lines:
        if line[0] != '#':
            words = line.split()
            key = coetools.str2num(words[0])
            val = ''  # if nothing there
            valstr = ' '.join(words[1:])
            valtuple = False
            if valstr[0] in '[(' and valstr[-1] in '])':  # LIST / TUPLE!
                valtuple = valstr[0] == '('
                valstr = valstr[1:-1].replace(',', '')
                words[1:] = valstr.split()
            if len(words) == 2:
                val = coetools.str2num(words[1])
            elif len(words) > 2:
                val = []
                for word in words[1:]:
                    val.append(coetools.str2num(word))
                if valtuple:
                    val = tuple(val)

            dict[key] = val
    return dict


def savedata(data, filename, dir="", header="", separator="  ", format='', labels='',
             descriptions='', units='', notes=[], pf=0, maxy=300, machine=0, silent=0):
    """Saves an array as an ascii data file into an array."""
    # AUTO FORMATTING (IF YOU WANT, ALSO OUTPUTS FORMAT SO YOU CAN USE IT NEXT TIME w/o HAVING
    # TO CALCULATE IT)
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
        skycat = coetools.strend(filename, '.scat')
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
                    if MLab_coe.isNaN(datum):
                        ni, nd = 1, 1
                    else:
                        # IF TOO BIG OR TOO SMALL, NEED exp FORMAT
                        if (abs(datum) > 1.e9) or (0 < abs(datum) < 1.e-5):
                            ni, nd = 1, 3
                            colexp[ix] = 1
                        else:
                            ni = len("% d" % datum) - 1
                            if ni <= 3:
                                nd = MLab_coe.ndec(datum, max=4)
                            else:
                                nd = MLab_coe.ndec(datum, max=7 - ni)
                            # Float32: ABOUT 7 DIGITS ARE ACCURATE (?)

                    if ni > colint[ix]:  # IF BIGGEST, YOU GET TO DECIDE NEG SPACE OR NO
                        colneg[ix] = (datum < 0)
                        # print '>', ix, colneg[ix], nd, coldec[ix]
                    # IF MATCH BIGGEST, YOU CAN SET NEG SPACE ON (NOT OFF)
                    elif ni == colint[ix]:
                        colneg[ix] = (datum < 0) or colneg[ix]
                        # print '=', ix, colneg[ix], nd, coldec[ix]
                    coldec[ix] = np.max([nd, coldec[ix]])
                    colint[ix] = np.max([ni, colint[ix]])

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
            if labels:
                headline = ''
                maxcollen = 1
                for label in labels:
                    maxcollen = max([maxcollen, len(label)])
                for ix in range(nx):
                    label = labels[ix].ljust(maxcollen)
                    headline += '# %2d %s' % (ix + 1, label)
                    if descriptions:
                        if descriptions[ix]:
                            headline += '  %s' % descriptions[ix]
                    headline += '\n'
                headline += '#\n'
                headline += '#'
                colformats = format.split('%')[1:]
                if not silent:
                    print()
                for ix in range(nx):
                    cf = colformats[ix]
                    collen = collens[ix]
                    label = labels[ix]
                    label = label.center(collen)
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
            if not header:
                header = []
                header.append('Title:\n')
                header.append('Authors:\n')
                header.append('Table:\n')
            header.append('=' * 80 + '\n')
            header.append('Byte-by-byte Description of file: %s\n' % filename)
            header.append('-' * 80 + '\n')
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
    data = coetools.FltArr(ny, nx)

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
        ix = MLab_coe.interp(x, self.x, np.arange(len(self.x)))
        iy = MLab_coe.interp(y, self.y, np.arange(len(self.y)))
        if not dointerp:  # JUST GET NEAREST
            ix = MLab_coe.roundint(ix)
            iy = MLab_coe.roundint(iy)
            z = self.z[ix, iy]
        else:
            z = MLab_coe.bilin2(iy, ix, self.z)
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
        ix = np.interp(x, self.x, np.arange(len(self.x)))
        iy = np.interp(y, self.y, np.arange(len(self.y)))
        if not dointerp:  # JUST GET NEAREST
            ix = MLab_coe.roundint(ix)
            iy = MLab_coe.roundint(iy)
            z = self.z[iy, ix]
        else:
            z = MLab_coe.bilin2(ix, iy, self.z)
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
    y = np.reshape(y, (1, len(y)))
    data = np.concatenate([y, data])
    x = np.concatenate([[0], x])
    x = np.reshape(x, (len(x), 1))
    data = np.concatenate([x, data], 1)
    if filename[-1] != '+':
        filename += '+'
    coeio.savedata1(data, filename, dir)


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
    labelstr = ','.join(labels)
    print(labelstr + ' = data')
    # STRING TO BE EXECUTED AFTER EXIT
    return 'from coeio import data,labels,labelstr\n' + labelstr + ' = data'


class VarsClass(object):
    def __init__(self, filename='', dir="", silent=0, labels='', labelheader='', headlines=0, loadheader=0):
        self.header = ''
        if filename:
            if coetools.strend(filename, '.fits'):  # FITS TABLE
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
                self2 = loadmachine(filename, dir, silent)
                self.labels = self2.labels[:]
                for label in self.labels:
                    exec('self.%s = self2.%s[:]' % (label, label))
                self.name = filename
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
        return np.array([getattr(self, label) for label in self.labels])

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
        if self.len() != len(good):
            print("VarsClass: SUBSET CANNOT BE CREATED: good LENGTH = %d, data LENGTH = %d" % (
                self.len(), len(good)))
        else:
            selfcopy = self.copy()
            data = self.updateddata()
            selfcopy.data = np.compress(good, data)
            selfcopy.assigndata()
            # PRESERVE UN-UPDATED DATA ARRAY
            selfcopy.data = np.compress(good, self.data)
            selfcopy.taken = np.compress(good, np.arange(self.len()))
            selfcopy.good = good
            return selfcopy

    def between(self, lo, labels, hi):
        """labels = list of labels or just one label"""
        if isinstance(labels, list):
            good = MLab_coe.between(lo, getattr(self, labels[0]), hi)
            for label in labels[1:]:
                good = good * MLab_coe.between(lo, getattr(self, label), hi)
        else:
            good = MLab_coe.between(lo, getattr(self, labels), hi)
        self.good = good
        return self.subset(good)

    def take(self, indices):
        indices = indices.astype(int)
        sub = VarsClass()
        sub.labels = self.labels[:]
        sub.taken = sub.takeind = indices
        sub.data = np.take(self.updateddata(), indices, 1)
        sh = sub.data.shape
        if len(sh) == 3:
            sub.data = np.reshape(sub.data, sh[:2])
        sub.assigndata()
        return sub

    def put(self, label, indices, values):
        x = getattr(self, label).copy()
        np.put(x, indices, values)
        setattr(self, label, x)

    def takeid(self, id, idlabel='id'):
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        i = np.argmin(abs(selfid - id))
        if selfid[i] != id:
            print("PROBLEM! ID %d NOT FOUND IN takeid" % id)
            return None
        else:
            return self.take(np.array([i]))

    def putid(self, label, id, value, idlabel='id'):
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        i = np.argmin(abs(selfid - id))
        if selfid[i] != id:
            print("PROBLEM! ID %d NOT FOUND IN putid" % id)
            return None
        else:
            x = getattr(self, label).copy()
            np.put(x, i, value)
            setattr(self, label, x)

    def takeids(self, ids, idlabel='id'):
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        indexlist = np.zeros(max(selfid) + 1, int) - 1
        np.put(indexlist, selfid, np.arange(len(selfid)))  # [- - - - 1 2 0]
        indices = np.take(indexlist, np.array(ids).astype(int))
        goodindices = np.compress(np.greater(indices, -1), indices)
        good = np.zeros(self.len(), int)
        good = good.astype(int)
        goodindices = goodindices.astype(int)
        np.put(good, goodindices, 1)
        self.good = good
        if -1 in indices:
            print("PROBLEM! NOT ALL IDS FOUND IN takeids!")
            print(np.compress(np.less(indices, 0), ids))
        return self.take(indices)

    def putids(self, label, ids, values, idlabel='id', rep=True):
        # Given selfid, at ids, place values
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        maxselfid = max(selfid)
        x = getattr(self, label).copy()
        idchecklist = selfid.copy()
        done = False
        while not done:  # len(idstochange):
            indexlist = np.zeros(maxselfid + 1, int) - 1
            np.put(indexlist, idchecklist, np.arange(self.len()))  # [- - - - 1 2 0]
            # ids = [4 6]  ->  indices = [1 0]
            indices = np.take(indexlist, np.array(ids).astype(int))
            if (-1 in indices) and (rep < 2):
                print("PROBLEM! NOT ALL IDS FOUND IN putids!")
                print(np.compress(np.less(indices, 0), ids))
            if coetools.singlevalue(values):
                values = np.zeros(self.len(), float) + values
            np.put(x, indices, values)
            np.put(idchecklist, indices, 0)
            if rep:  # Repeat if necessary
                done = MLab_coe.total(idchecklist) == 0
                rep += 1
            else:
                done = 1
        exec('self.%s = x' % label)

    def takecids(self, ids, idlabel='id'):  # only take common ids
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        n = max((max(selfid), max(ids)))
        indexlist = np.zeros(n + 1, int)
        np.put(indexlist, selfid, np.arange(len(selfid)) + 1)  # [- - - - 1 2 0]
        indices = np.take(indexlist, np.array(ids).astype(int))
        indices = np.compress(indices, indices - 1)
        goodindices = np.compress(np.greater(indices, -1), indices)
        good = np.zeros(self.len(), int)
        np.put(good, goodindices, 1)
        self.good = good
        return self.take(indices)

    def removeids(self, ids, idlabel='id'):
        selfid = self.get(idlabel).astype(int)  # [6 4 5]
        if coetools.singlevalue(ids):
            ids = [ids]
        newids = coetools.invertselection(ids, selfid)
        return self.takeids(newids)

    def get(self, label, orelse=None):
        if label in self.labels:
            out = getattr(self, label)
        else:
            out = orelse
        return out

    def set(self, label, data):
        if coetools.singlevalue(data):
            data = np.zeros(self.len(), float) + data
        exec('self.%s = data' % label)

    def add(self, label, data):
        if coetools.singlevalue(data):
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
                commonids = coetools.common(self.id, self2.id)
                if commonids:
                    selfuniqueids = coetools.invertselection(commonids, self.id)
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
                SI = np.argsort(np.random(self.len()))
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
            SI = np.argsort(label)  # label contains an array
        self.updatedata()
        self.data = np.take(self.data, SI, 1)
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
            matchid, dist = coetools.findmatch(searchcat.x, searchcat.y, self.x[i],
                                               self.y[i], dtol=dtol[i], silent=1,
                                               returndist=1, xsorted=0)
            matchids.append(matchid)
            dists.append(dist)
        matchids = np.array(matchids)
        dists = np.array(dists)
        matchids = np.where(np.equal(matchids, searchcat.len()), -1, matchids)
        self.assign('matchid', matchids)
        self.assign('dist', dists)

    def findmatches2(self, searchcat, dtol=0):
        """Finds closest matches for self within searchcat"""
        i, d = coetools.findmatches2(searchcat.x, searchcat.y, self.x, self.y)
        if dtol:
            i = np.where(np.less(d, dtol), i, -1)
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

    def save(self, name='', dir="", header='', format='', labels=1, pf=0, maxy=300,
             machine=0, silent=0):
        if type(labels) == list:
            self.labels = labels
        labels = labels and self.labels  # if labels then self.labels, else 0
        name = name or self.name  # if name then name, else self.name
        header = header or self.header  # if header then header, else self.header
        savedata(self.updateddata(), name + '+', dir=dir, labels=labels, header=header,
                 format=format, pf=pf, maxy=maxy, machine=machine, descriptions=self.descriptions,
                 units=self.units, notes=self.notes, silent=silent)

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
                a = np.array(self.get(label))
                #if not pyfitsusesnumpy:
                #    a = numarray.array(a)
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
                            i = description[:45].rfind(' ')
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
        #if not pyfitsusesnumpy:
        #data = np.array(data)  # .tolist() ??
        return np.array(data)
    else:
        print()
        print(filename, "DOESN'T EXIST")
        # FILE_DOESNT_EXIST[9] = 3
