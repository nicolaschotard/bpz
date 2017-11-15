from __future__ import division
from __future__ import print_function
# ~/bpz-1.99.3/plots/webpage.py

# BPZ results for catalog (SED fits and P(z))

# python $BPZPATH/plots/webpage.py root
#   Produces html/index.html by default

# python $BPZPATH/plots/webpage.py root i1-10
# python $BPZPATH/plots/webpage.py root i-10
#   First 10 objects

# python $BPZPATH/plots/webpage.py root 2225,1971,7725
#   Objects with ID numbers 2225, 1971, 7725

# python $BPZPATH/plots/webpage.py root some.i
#   Objects with ID numbers listed in file some.i (one per line)

# python $BPZPATH/plots/webpage.py root -ZMAX 6
#   Max z for P(z)  (Default 7)

# python $BPZPATH/plots/webpage.py root -DIR myhtmldir
#   Makes myhtmldir/index.html

# python $BPZPATH/plots/webpage.py root -REDO
# python $BPZPATH/plots/webpage.py root -REDO sed
# python $BPZPATH/plots/webpage.py root -REDO prob
#   Redo all plots or just the sed or P(z) plots
#   Default: Don't redo plots if they already exist

# ~/bpz-1.99.2/plots/webpage.py
# ~/ACS/CL0024/colorpro/webpage.py
# label.py
# ~/p/stampbpzhtml.py

# id x y segmid?
# outdir
# segm.fits
# color image
import os
from bpz import coetools
from bpz import coeio
import sedplotAB
import probplot
import string
import numpy as np
import sys


n = 200  # IMAGE SIZE


def sedplots(cat, root, outdir, redo=False):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    ids = cat.get('segmid', cat.id).round().astype(int)
    b = sedplotAB.bpzPlots(root, ids, probs=None)
    redo = redo in [True, 'sed']
    b.flux_comparison_plots(show_plots=0, save_plots='png', colors={
    }, nomargins=0, outdir=outdir, redo=redo)
    #os.system('\mv %s_sed_*.png %s' % (root, outdir))


def probplots(cat, root, outdir, zmax=7, redo=False):
    if not os.path.exists(root + '.probs'):
        return

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    ids = cat.get('segmid', cat.id).round().astype(int)
    redo = redo in [True, 'prob']
    for i in range(cat.len()):
        id = ids[i]
        probplot.probplot(root, id, zmax=zmax, nomargins=0,
                          outdir=outdir, redo=redo)

    #os.system('\mv probplot*png ' + outdir)


def webpage(cat, bpzroot, outfile, ncolor=1, idfac=1.):
    fout = open(outfile, 'w')

    coloraddons = list(string.ascii_lowercase)
    coloraddons[0] = ''
    coloraddons = coloraddons[:ncolor]

    bpzpath = os.environ.get('BPZPATH')
    inroll = os.path.join(bpzpath, 'plots')
    fout.write('\n')
    fout.write('<h1>BPZ results for %s.cat</h1>\n\n' % bpzroot)
    ids = cat.id.round().astype(int)
    segmids = cat.get('segmid', ids).round().astype(int)
    for i in range(cat.len()):
        id = ids[i]
        segmid = segmids[i]
        id2 = id * idfac
        fout.write('Object #%s' % str(int(id2))) #num2str(id2))
        # fout.write('Object #%d' % id)
        if 'zb' in cat.labels:
            fout.write(' &nbsp; BPZ = %.2f' % cat.zb[i])
        if ('zbmin' in cat.labels) and ('zbmax' in cat.labels):
            fout.write(' [%.2f--%.2f]' % (cat.zbmin[i], cat.zbmax[i]))
        if 'tb' in cat.labels:
            fout.write(' &nbsp; type = %.2f' % cat.tb[i])
        if 'chisq2' in cat.labels:
            fout.write(' &nbsp; chisq2 = %.2f' % cat.chisq2[i])
        if 'odds' in cat.labels:
            fout.write(' &nbsp; ODDS = %.2f' % cat.odds[i])
        if 'zspec' in cat.labels:
            fout.write(' &nbsp; spec-z = %.2f' % cat.zspec[i])
        fout.write('<br>\n')

        for addon in coloraddons:
            fout.write(' <a href="#">')
            fout.write(' hsrc="segm/segm%d.gif"' % segmid)
            fout.write(' border=0')
            fout.write('></a>\n')

        fout.write(
            ' <img src="sedplots/%s_sed_%d.png"   border=0 height=300 width=400>\n' % (bpzroot, segmid))
        fout.write(
            ' <img src="probplots/probplot%d.png" border=0 height=300 width=400>' % segmid)
        fout.write('<br>\n')
        fout.write('<br>\n\n')

    fout.close()

# WANT INDICES AS OUTPUT
# None: all
# deez.i: ids in text file
# i0: 1st (i = 0) object
# 285: id = 285
# 285,63: ids = 285,63

# python $BPZPATH/plots/webpage.py root ids -DIR outdir -SEGM segmids


def run():
    bpzroot = sys.argv[1]
    #cat = loadcat(bpzroot+'.cat')
    #cat = loadcat(bpzroot+'_photbpz.cat')
    cat = coeio.loadcat(bpzroot + '_bpz.cat')

    ids = None
    mycat = cat
    if len(sys.argv) > 2:
        if sys.argv[2][0] != '-':
            id_str = sys.argv[2]

            if id_str[-2:] == '.i':  # External file with IDs (one per line)
                ids = np.ravel(coeio.loaddata(id_str).round().astype(int))
                mycat = cat.takeids(ids)
            elif id_str[0] == 'i':  # Indices
                num = id_str[1:]
                if string.find(num, '-') == -1:  # single number
                    i = int(id_str[1:])
                    mycat = cat.take(np.array([i + 1]))
                else:
                    lo, hi = num.split('-')
                    lo = lo or 1
                    lo = int(lo)
                    hi = int(hi)
                    hi = hi or cat.len()
                    ii = np.arange(lo - 1, hi)
                    mycat = cat.take(ii)
            else:  # IDs separated by commas
                ids = coetools.stringsplitatoi(id_str, ',')
                mycat = cat.takeids(ids)

    params = coeio.params_cl()
    outdir = params.get('DIR', 'html')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    idfac = params.get('IDFAC', 1.)

    zmax = params.get('ZMAX', 7.)

    redo = params.get('REDO', False)
    if redo == None:
        redo = True

    ltrs = list(string.ascii_lowercase)
    ltrs[0] = ''

    colorfiles = []

    sedplots(mycat, bpzroot, os.path.join(outdir, 'sedplots'), redo=redo)  # id
    probplots(mycat, bpzroot, os.path.join(outdir, 'probplots'),
              zmax=zmax, redo=redo)  # id
    webpage(mycat, bpzroot, os.path.join(outdir, 'index.html'),
            len(colorfiles), idfac=idfac)  # id segmid


if __name__ == '__main__':
    run()
else:
    pass
