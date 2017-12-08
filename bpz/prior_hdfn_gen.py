from __future__ import division
from __future__ import absolute_import
from past.utils import old_div
import numpy as np


def function(z, m, nt):
    """HDFN prior from Benitez 2000 for Ellipticals, Spirals, and Irregular/Starbursts
    Returns an array pi[z[:],:nt]
    The input magnitude is F814W AB
    """

    global zt_at_a
    nz = len(z)
    momin_hdf = 20.
    if m > 32.:
        m = 32.
    if m < 20.:
        m = 20.

    # nt Templates = nell Elliptical + nsp Spiral + nSB starburst
    try:  # nt is a list of 3 values
        nell, nsp, nsb = nt
    except:  # nt is a single value
        nell = 1  # 1 Elliptical in default template set
        nsp = 2   # 2 Spirals in default template set
        nsb = nt - nell - nsp  # rest Irr/SB
    nn = nell, nsp, nsb
    nt = sum(nn)

    # See Table 1 of Benitez00
    a = 2.465,  1.806,  0.906
    zo = 0.431,  0.390,  0.0626
    km = 0.0913, 0.0636, 0.123
    k_t = 0.450,  0.147

    a = np.repeat(a, nn)
    zo = np.repeat(zo, nn)
    km = np.repeat(km, nn)
    k_t = np.repeat(k_t, nn[:2])

    # Fractions expected at m = 20:
    # 35% E/S0
    # 50% Spiral
    # 15% Irr
    fo_t = 0.35, 0.5
    fo_t = old_div(fo_t, np.array(nn[:2]))
    fo_t = np.repeat(fo_t, nn[:2])

    dm = m - momin_hdf
    zmt = np.clip(zo + km * dm, 0.01, 15.)
    zmt_at_a = zmt**(a)
    # We define z**a as global to keep it
    # between function calls. That way it is
    # estimated only once
    try:
        xxx[9] = 3
        zt_at_a.shape
    except NameError:
        zt_at_a = np.power.outer(z, a)

    # Morphological fractions
    nellsp = nell + nsp
    f_t = np.zeros((len(a),), float)
    f_t[:nellsp] = fo_t * np.exp(-k_t * dm)
    f_t[nellsp:] = old_div((1. - np.add.reduce(f_t[:nellsp])), float(nsb))
    # Formula: zm=zo+km*(m_m_min) p(z|T,m)=(z**a)*exp(-(z/zm)**a)
    p_i = zt_at_a[:nz, :nt] * np.exp(-np.clip(old_div(zt_at_a[:nz, :nt],
                                                      zmt_at_a[:nt]), 0., 700.))
    # This eliminates the very low level tails of the priors
    norm = np.add.reduce(p_i[:nz, :nt], 0)
    p_i[:nz, :nt] = np.where(np.less(old_div(p_i[:nz, :nt], norm[:nt]), old_div(1e-2, float(nz))),
                             0., old_div(p_i[:nz, :nt], norm[:nt]))
    norm = np.add.reduce(p_i[:nz, :nt], 0)
    p_i[:nz, :nt] = p_i[:nz, :nt] / norm[:nt] * f_t[:nt]
    return p_i
