from __future__ import division
from __future__ import print_function
# $p/probplot.py bvizjh 1234
# $p/probplot.py bvizjh_cut_sexseg2_allobjs_newres_offset3_djh_jhbad-99_ssp25Myrz008 6319

# plots P(z) for a given galaxy ID #
# blue line is zb
# green lines are zbmin, zbmax
# red lines bracket 95% odds interval (hopefully that's what you want!)
# -- i could make this an input...

# from ~/UDF/probstest.py

# SINGLE GALAXY P(z)
# BPZ RUN WITH -ODDS 0.95
# 95% = 2-sigma
# OUTPUT ODDS = P(dz) = P(zb-dz:zb+dz), where dz = 2 * 0.06 * (1 + z)  [2 for 2-sigma]
# zbmin & zbmax CONTAIN 95% OF P(z) (EACH WING CONTAINS 2.5%)
# -- UNLESS P(z) IS TOO SHARP: MIN_RMS=0.06 TAKES OVER: dz >= 2 * 0.06 * (1+z)

#from ksbtools import *
#htmldir = loadfile('/home/coe/UDF/htmldir.txt')[0]

from builtins import range
from past.utils import old_div
from bpz.coetools import *
from coeplott import *
from os.path import exists, join

htmldir = ''


def probplot(root, id1, zmax=None, zspec=-1, save=1, pshow=0, nomargins=0, outdir='', redo=False):
    outoforder = 1
    # LOAD PROBABILITIES, ONE LINE AT A TIME (MUCH MORE EFFICIENT!!)
    # FIRST I CHECK IF THE PLOT EXISTS ALREADY
    # THIS MAKES IT TOUGH TO CALL WITH i3
    #  BECAUSE I HAVE TO READ THE PROBS FILE FIRST TO GET THE ID NUMBER...
    outimg = 'probplot%d.png' % id1
    if exists(join(outdir, outimg)) and not pshow and not redo:
        print(join(outdir, outimg), 'ALREADY EXISTS')
    else:
        if save:
            print('CREATING ' + join(outdir, outimg) + '...')
        if nomargins:
            figure(1, figsize=(4, 2))
            clf()
            axes([0.03, 0.2, 0.94, 0.8])
        else:
            figure()
            clf()
        ioff()
        fprob = open(root + '.probs', 'r')
        line = fprob.readline()  # header
        zrange = strbtw(line, 'arange(', ')')
        zrange = stringsplitatof(zrange, ',')
        # print zrange
        z = arange(zrange[0], zrange[1], zrange[2])

        if type(id1) == str:
            i = int(id1[1:]) - 1
            for ii in range(i + 1):
                line = fprob.readline()
            i = line.find(' ')
            id = line[:i]
            id1 = id[:]
            print(id1, 'id1')
            id = int(id)
        else:
            id = 0
            while (id != id1) and line and ((id < id1) or outoforder):
                line = fprob.readline()
                i = line.find(' ')
                id = line[:i]
                id = int(id)
            if (id != id1):
                print('%d NOT FOUND' % id1)
                print('QUITTING.')
                sys.exit()
            # print 'FOUND IT...'

        fprob.close()

        prob = stringsplitatof(line[i:-1])
        n = len(prob)
        #z = (arange(n) + 1) * 0.01
        if zmax == None:
            zmax = max(z)
        else:
            nmax = roundint(old_div(zmax, zrange[2]))
            z = z[:nmax]  # nmax+1
            prob = prob[:nmax]  # nmax+1

        zc, pc = z, prob
        pmax = max(prob)

        # LOAD BPZ RESULTS, ONE LINE AT A TIME (MUCH MORE EFFICIENT!!)
        fin = open(root + '_bpz.cat', 'r')
        line = '#'
        while line[0] == '#':
            lastline = line[:]
            line = fin.readline()  # header
        labels = lastline[1:-1].split()

        line = line.strip()
        i = line.find(' ')
        id = line[:i]
        id = int(id)
        while id != id1 and (id < id1 or outoforder):
            line = fin.readline()
            line = line.strip()
            i = line.find(' ')
            id = line[:i]
            id = int(id)

        fin.close()

        data = stringsplitatof(line[i:])
        labels = labels[1:]  # get rid of "id"
        vars = {labels[i]: data[i] for i in range(len(labels) - 1)}
        dz = 2 * 0.06 * (1 + vars['zb'])  # 2-sigma = 95% (ODDS)
        zlo = vars['zb'] - dz
        zhi = vars['zb'] + dz

        if vars['zspec'] >= 0:
            lw = [5, 3][nomargins]
            vline([vars['zspec']], (1, 0, 0), linewidth=lw, alpha=0.5)
          
        fillbetween(zc, zeros(len(pc)), zc, pc, facecolor='blue')
        plot([zc[0], zc[-1]], [0, 0], color='white')
        if 0:
            p.add(FillBelow(zc, pc, color='grey50'))
            p.add(Curve([zb, zb], [0, pmax * .1], color='cyan', linewidth=5))
            p.add(Curve([zbmin, zbmin], [0, pmax * .07],
                        color='green', linewidth=5))
            p.add(Curve([zbmax, zbmax], [0, pmax * .07],
                        color='green', linewidth=5))
            p.add(Curve([zlo, zlo], [0, pmax * .07], color='red'))
            p.add(Curve([zhi, zhi], [0, pmax * .07], color='red'))
      
        xlabel('z')
        if nomargins:
            ax = gca()
            ax.set_yticklabels([])
        else:
            ylabel('P(z)')
        ylim(0, 1.05 * pmax)
        if save:
            savefig(join(outdir, outimg))
            os.system('chmod 644 ' + join(outdir, outimg))
        if pshow:
            show()
            print('KILL PLOT WINOW TO TERMINATE OR CONTINUE.')


if __name__ == '__main__':
    id1 = sys.argv[2]
    if id1[0] != 'i':
        id1 = int(id1)
    params = params_cl()
    save_plots = 'SAVE' in params
    show_plots = 1 - save_plots
    zspec = params.get('ZSPEC', -1)
    zmax = params.get('ZMAX', None)
    nomargins = 'NOMARGINS' in params
    probplot(sys.argv[1], id1, zmax=zmax, zspec=zspec,
             save=save_plots, pshow=show_plots, nomargins=nomargins)
else:
    pass
