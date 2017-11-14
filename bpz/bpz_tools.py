#!/usr/local/bin/python

""" bpz_tools.py: Contains useful functions for I/O and math.
    TO DO:
      Include higher order interpolations
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from past.utils import old_div
from numpy import *
from .MLab_coe import *
from .useful import *
from string import *
import os
import sys

clight_AHz = 2.99792458e18
Vega = 'Vega_reference'

# Smallest number accepted by python
eps = 1e-300
eeps = log(eps)

#from .MLab_coe import log10 as log10a


#def log10(x):
#    return log10a(x + eps)
#    # return log10clip(x, -33)


# This quantities are used by the AB files
zmax_ab = 12.
dz_ab = 0.01
ab_clip = 1e-6


# Initialize path info
bpz_dir = os.getenv('BPZPATH')
fil_dir = bpz_dir + 'FILTER/'
sed_dir = bpz_dir + 'SED/'
ab_dir = bpz_dir + 'AB/'

# Auxiliary synthetic photometry functions


def flux(xsr, ys, yr, ccd='yes', units='nu'):
    """ Flux of spectrum ys observed through response yr,
        both defined on xsr 
        Both f_nu and f_lambda have to be defined over lambda
        If units=nu, it gives f_nu as the output
    """
    if ccd == 'yes':
        yr = yr * xsr
    norm = trapz(yr, xsr)
    f_l = old_div(trapz(ys * yr, xsr), norm)
    if units == 'nu':
        # Pivotal wavelenght
        lp = sqrt(old_div(norm, trapz(yr / xsr / xsr, xsr)))
        return f_l * lp**2 / clight_AHz
    else:
        return f_l


def pivotal_wl(filter, ccd='yes'):
    xr, yr = get_filter(filter)
    if ccd == 'yes':
        yr = yr * xr
    norm = trapz(yr, xr)
    return sqrt(old_div(norm, trapz(yr / xr / xr, xr)))


def filter_center(filter, ccd='yes'):
    """Estimates the central wavelenght of the filter"""
    if type(filter) == type(""):
        xr, yr = get_filter(filter)
    else:
        xr = filter[0]
        yr = filter[1]
    if ccd == 'yes':
        yr = yr * xr
    return old_div(trapz(yr * xr, xr), trapz(yr, xr))


def AB(flux):
    """AB magnitude from f_nu"""
    return -2.5 * np.log10(flux) - 48.60


def flux2mag(flux):
    """Convert arbitrary flux to magnitude"""
    return -2.5 * np.log10(flux)


def AB2Jy(ABmag):
    """Convert AB magnitudes to Jansky"""
    return old_div(10.**(-0.4 * (ABmag + 48.60)), 1e-23)


def mag2flux(mag):
    """Convert flux to arbitrary flux units"""
    return 10.**(-.4 * mag)


def e_frac2mag(fracerr):
    """Convert fractionary flux error to mag error"""
    return 2.5 * np.log10(1. + fracerr)


def e_mag2frac(errmag):
    """Convert mag error to fractionary flux error"""
    return 10.**(.4 * errmag) - 1.


# Synthetic photometry functions


def etau_madau(wl, z):
    """
    Madau 1995 extinction for a galaxy spectrum at redshift z 
    defined on a wavelenght grid wl
    """
    n = len(wl)
    l = array([1216., 1026., 973., 950.])
    xe = 1. + z

    # If all the spectrum is redder than (1+z)*wl_lyman_alfa
    if wl[0] > l[0] * xe:
        return zeros(n) + 1.

    # Madau coefficients
    c = array([3.6e-3, 1.7e-3, 1.2e-3, 9.3e-4])
    ll = 912.
    tau = wl * 0.
    i1 = searchsorted(wl, ll)
    i2 = n - 1
    # Lyman series absorption
    for i in range(len(l)):
        i2 = searchsorted(wl[i1:i2], l[i] * xe)
        tau[i1:i2] = tau[i1:i2] + c[i] * (old_div(wl[i1:i2], l[i]))**3.46

    if ll * xe < wl[0]:
        return exp(-tau)

    # Photoelectric absorption
    xe = 1. + z
    i2 = searchsorted(wl, ll * xe)
    xc = old_div(wl[i1:i2], ll)
    xc3 = xc**3
    tau[i1:i2] = tau[i1:i2] +\
        (0.25 * xc3 * (xe**.46 - xc**0.46)
         + 9.4 * xc**1.5 * (xe**0.18 - xc**0.18)
         - 0.7 * xc3 * (xc**(-1.32) - xe**(-1.32))
         - 0.023 * (xe**1.68 - xc**1.68))

    tau = clip(tau, 0, 700)
    return exp(-tau)
    # if tau>700. : return 0.
    # else: return exp(-tau)


def etau(wl, z):
    """
    Madau 1995 and Scott 2000 extinction for a galaxy spectrum
    at redshift z observed on a wavelenght grid wl
    """

    n = len(wl)
    l = array([1216., 1026., 973., 950.])
    xe = 1. + z

    # If all the spectrum is redder than (1+z)*wl_lyman_alfa
    if wl[0] > l[0] * xe:
        return zeros(n) + 1.

    # Extinction coefficients

    c = array([1., 0.47, 0.33, 0.26])
    if z > 4.:
        # Numbers from Madau paper
        coeff = 0.0036
        gamma = 2.46
    elif z < 3:
        # Numbers from Scott et al. 2000 paper
        coeff = 0.00759
        gamma = 1.35
    else:
        # Interpolate between two numbers
        coeff = .00759 + (0.0036 - 0.00759) * (z - 3.)
        gamma = 1.35 + (2.46 - 1.35) * (z - 3.)
    c = coeff * c

    ll = 912.
    tau = wl * 0.
    i1 = searchsorted(wl, ll)
    i2 = n - 1
    # Lyman series absorption
    for i in range(len(l)):
        i2 = searchsorted(wl[i1:i2], l[i] * xe)
        tau[i1:i2] = tau[i1:i2] + c[i] * \
            (old_div(wl[i1:i2], l[i]))**(1. + gamma)

    if ll * xe < wl[0]:
        return exp(-tau)

    # Photoelectric absorption
    xe = 1. + z
    i2 = searchsorted(wl, ll * xe)
    xc = old_div(wl[i1:i2], ll)
    xc3 = xc**3
    tau[i1:i2] = tau[i1:i2] +\
        (0.25 * xc3 * (xe**.46 - xc**0.46)
         + 9.4 * xc**1.5 * (xe**0.18 - xc**0.18)
         - 0.7 * xc3 * (xc**(-1.32) - xe**(-1.32))
         - 0.023 * (xe**1.68 - xc**1.68))
    return exp(-tau)


def get_sednfilter(sed, filter):
    # Gets a pair of SED and filter from the database
    # And matches the filter resolution to that of the spectrum
    # where they overlap
    """Usage:
    xs,ys,yr=get_sednfilter(sed,filter)
    """
    # Figure out the correct names
    if filter[-4:] != '.res':
        filter = filter + '.res'
    if sed[-4:] != '.sed':
        sed = sed + '.sed'
    sed = sed_dir + sed
    filter = fil_dir + filter
    # Get the data
    x_sed, y_sed = get_data(sed, list(range(2)))
    nsed = len(x_sed)
    x_res, y_res = get_data(filter, list(range(2)))
    nres = len(x_res)
    if not ascend(x_sed):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % sed)
        print('They should start with the shortest lambda and end with the longest')
    if not ascend(x_res):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % filter)
        print('They should start with the shortest lambda and end with the longest')

    # Define the limits of interest in wavelenght
    i1 = searchsorted(x_sed, x_res[0]) - 1
    i1 = maximum(i1, 0)
    i2 = searchsorted(x_sed, x_res[nres - 1]) + 1
    i2 = minimum(i2, nsed - 1)
    r = match_resol(x_res, y_res, x_sed[i1:i2])
    r = where(less(r, 0.), 0., r)  # Transmission must be >=0
    return x_sed[i1:i2], y_sed[i1:i2], r


def get_sed(sed):
    # Get x_sed,y_sed from a database spectrum
    """Usage:
    xs,ys=get_sed(sed)
    """
    # Figure out the correct names
    if sed[-4:] != '.sed':
        sed = sed + '.sed'
    sed = sed_dir + sed
    # Get the data
    x, y = get_data(sed, list(range(2)))
    if not ascend(x):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % sed)
        print('They should start with the shortest lambda and end with the longest')
    return x, y


def get_filter(filter):
    # Get x_res,y_res from a database spectrum
    """Usage:
    xres,yres=get_filter(filter)
    """
    # Figure out the correct names
    if filter[-4:] != '.res':
        filter = filter + '.res'
    filter = fil_dir + filter
    # Get the data
    x, y = get_data(filter, list(range(2)))
    if not ascend(x):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % filter)
        print('They should start with the shortest lambda and end with the longest')
    return x, y


def redshift(wl, flux, z):
    """ Redshift spectrum y defined on axis x 
      to redshift z
      Usage:
         y_z=redshift(wl,flux,z) 
    """
    if z == 0.:
        return flux
    else:
        f = match_resol(wl, flux, old_div(wl, (1. + z)))
        return where(less(f, 0.), 0., f)


def obs_spectrum(sed, z, madau=1):
    """Generate a redshifted and madau extincted spectrum"""
    # Figure out the correct names
    if sed[-4:] != '.sed':
        sed = sed + '.sed'
    sed = sed_dir + sed
    # Get the data
    x_sed, y_sed = get_data(sed, list(range(2)))
    # ys_z will be the redshifted and corrected spectrum
    ys_z = match_resol(x_sed, y_sed, old_div(x_sed, (1. + z)))
    if madau:
        ys_z = etau_madau(x_sed, z) * ys_z
    return x_sed, ys_z


def f_z_sed(sed, filter, z=array([0.]), ccd='yes', units='lambda', madau='yes'):
    """
    Returns array f with f_lambda(z) or f_nu(z) through a given filter 
    Takes into account intergalactic extinction. 
    Flux normalization at each redshift is arbitrary 
    """

    if type(z) == type(0.):
        z = array([z])

    # Figure out the correct names
    if sed[-4:] != '.sed':
        sed = sed + '.sed'
    sed = sed_dir + sed
    if filter[-4:] != '.res':
        filter = filter + '.res'
    filter = fil_dir + filter

    # Get the data
    x_sed, y_sed = get_data(sed, list(range(2)))
    nsed = len(x_sed)
    x_res, y_res = get_data(filter, list(range(2)))
    nres = len(x_res)

    if not ascend(x_sed):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % sed)
        print('They should start with the shortest lambda and end with the longest')
        print('This will probably crash the program')

    if not ascend(x_res):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % filter)
        print('They should start with the shortest lambda and end with the longest')
        print('This will probably crash the program')

    if x_sed[-1] < x_res[-1]:  # The SED does not cover the whole filter interval
        print('Extrapolating the spectrum')
        # Linear extrapolation of the flux using the last 4 points
        d_extrap = old_div((x_sed[-1] - x_sed[0]), len(x_sed))
        x_extrap = arange(x_sed[-1] + d_extrap, x_res[-1] + d_extrap, d_extrap)
        extrap = lsq(x_sed[-5:], y_sed[-5:])
        y_extrap = extrap.fit(x_extrap)
        y_extrap = clip(y_extrap, 0., max(y_sed[-5:]))
        x_sed = concatenate((x_sed, x_extrap))
        y_sed = concatenate((y_sed, y_extrap))

    # Wavelenght range of interest as a function of z
    wl_1 = old_div(x_res[0], (1. + z))
    wl_2 = old_div(x_res[-1], (1. + z))
    n1 = clip(searchsorted(x_sed, wl_1) - 1, 0, 100000)
    n2 = clip(searchsorted(x_sed, wl_2) + 1, 0, nsed - 1)

    # Typical delta lambda
    delta_sed = old_div((x_sed[-1] - x_sed[0]), len(x_sed))
    delta_res = old_div((x_res[-1] - x_res[0]), len(x_res))

    # Change resolution of filter
    if delta_res > delta_sed:
        x_r = arange(x_res[0], x_res[-1] + delta_sed, delta_sed)
        # print 'Changing filter resolution from %.2f AA to %.2f AA' % (delta_res,delta_sed)
        r = match_resol(x_res, y_res, x_r)
        r = where(less(r, 0.), 0., r)  # Transmission must be >=0
    else:
        x_r, r = x_res, y_res

    # Operations necessary for normalization and ccd effects
    if ccd == 'yes':
        r = r * x_r
    norm_r = trapz(r, x_r)
    if units == 'nu':
        const = norm_r / trapz(r / x_r / x_r, x_r) / clight_AHz
    else:
        const = 1.

    const = old_div(const, norm_r)

    nz = len(z)
    f = zeros(nz) * 1.
    for i in range(nz):
        i1, i2 = n1[i], n2[i]
        ys_z = match_resol(x_sed[i1:i2], y_sed[i1:i2],
                           old_div(x_r, (1. + z[i])))
        if madau != 'no':
            ys_z = etau_madau(x_r, z[i]) * ys_z
        f[i] = trapz(ys_z * r, x_r) * const
    if nz == 1:
        return f[0]
    else:
        return f


def ABflux(sed, filter, madau='yes'):
    """
    Calculates a AB file like the ones used by bpz
    It will set to zero all fluxes
    which are ab_clip times smaller than the maximum flux.
    This eliminates residual flux which gives absurd
    colors at very high-z
    """

    print(sed, filter)
    ccd = 'yes'
    units = 'nu'
    madau = madau
    # zmax_ab and dz_ab are def. in bpz_tools
    z_ab = arange(0., zmax_ab, dz_ab)

    # Figure out the correct names
    if sed[-4:] != '.sed':
        sed = sed + '.sed'
    sed = sed_dir + sed
    if filter[-4:] != '.res':
        filter = filter + '.res'
    filter = fil_dir + filter

    # Get the data
    x_sed, y_sed = get_data(sed, list(range(2)))
    nsed = len(x_sed)
    x_res, y_res = get_data(filter, list(range(2)))
    nres = len(x_res)

    if not ascend(x_sed):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % sed)
        print('They should start with the shortest lambda and end with the longest')
        print('This will probably crash the program')

    if not ascend(x_res):
        print()
        print('Warning!!!')
        print('The wavelenghts in %s are not properly ordered' % filter)
        print('They should start with the shortest lambda and end with the longest')
        print('This will probably crash the program')

    if x_sed[-1] < x_res[-1]:  # The SED does not cover the whole filter interval
        print('Extrapolating the spectrum')
        # Linear extrapolation of the flux using the last 4 points
        d_extrap = old_div((x_sed[-1] - x_sed[0]), len(x_sed))
        x_extrap = arange(x_sed[-1] + d_extrap, x_res[-1] + d_extrap, d_extrap)
        extrap = lsq(x_sed[-5:], y_sed[-5:])
        y_extrap = extrap.fit(x_extrap)
        y_extrap = clip(y_extrap, 0., max(y_sed[-5:]))
        x_sed = concatenate((x_sed, x_extrap))
        y_sed = concatenate((y_sed, y_extrap))

    # Wavelenght range of interest as a function of z_ab
    wl_1 = old_div(x_res[0], (1. + z_ab))
    wl_2 = old_div(x_res[-1], (1. + z_ab))
    print('x_res[0]', x_res[0])
    print('x_res[-1]', x_res[-1])
    n1 = clip(searchsorted(x_sed, wl_1) - 1, 0, 100000)
    n2 = clip(searchsorted(x_sed, wl_2) + 1, 0, nsed - 1)

    # Typical delta lambda
    delta_sed = old_div((x_sed[-1] - x_sed[0]), len(x_sed))
    delta_res = old_div((x_res[-1] - x_res[0]), len(x_res))

    # Change resolution of filter
    if delta_res > delta_sed:
        x_r = arange(x_res[0], x_res[-1] + delta_sed, delta_sed)
        print('Changing filter resolution from %.2f AA to %.2f' %
              (delta_res, delta_sed))
        r = match_resol(x_res, y_res, x_r)
        r = where(less(r, 0.), 0., r)  # Transmission must be >=0
    else:
        x_r, r = x_res, y_res

    # Operations necessary for normalization and ccd effects
    if ccd == 'yes':
        r = r * x_r
    norm_r = trapz(r, x_r)
    if units == 'nu':
        const = norm_r / trapz(r / x_r / x_r, x_r) / clight_AHz
    else:
        const = 1.

    const = old_div(const, norm_r)

    nz_ab = len(z_ab)
    f = zeros(nz_ab) * 1.
    for i in range(nz_ab):
        i1, i2 = n1[i], n2[i]
        # if (x_sed[i1] > max(x_r/(1.+z_ab[i]))) or (x_sed[i2] < min(x_r/(1.+z_ab[i]))):
        if (x_sed[i1] > old_div(x_r[-1], (1. + z_ab[i]))) or (x_sed[i2 - 1] < old_div(x_r[0], (1. + z_ab[i]))) or (i2 - i1 < 2):
            print('bpz_tools.ABflux:')
            print("YOUR FILTER RANGE DOESN'T OVERLAP AT ALL WITH THE REDSHIFTED TEMPLATE")
            print("THIS REDSHIFT IS OFF LIMITS TO YOU:")
            print('z = ', z_ab[i])
            print(i1, i2)
            print(x_sed[i1], x_sed[i2])
            print(y_sed[i1], y_sed[i2])
            print(min(old_div(x_r, (1. + z_ab[i]))),
                  max(old_div(x_r, (1. + z_ab[i]))))
            # NOTE: x_sed[i1:i2] NEEDS TO COVER x_r(1.+z_ab[i])
            # IF THEY DON'T OVERLAP AT ALL, THE PROGRAM WILL CRASH
            # sys.exit(1)
        else:
            try:
                ys_z = match_resol(
                    x_sed[i1:i2], y_sed[i1:i2], old_div(x_r, (1. + z_ab[i])))
            except:
                print(i1, i2)
                print(x_sed[i1], x_sed[i2 - 1])
                print(y_sed[i1], y_sed[i2 - 1])
                print(min(old_div(x_r, (1. + z_ab[i]))),
                      max(old_div(x_r, (1. + z_ab[i]))))
                print(old_div(x_r[1], (1. + z_ab[i])),
                      old_div(x_r[-2], (1. + z_ab[i])))
                print(x_sed[i1:i2])
                print(old_div(x_r, (1. + z_ab[i])))
                pause()
            if madau != 'no':
                ys_z = etau_madau(x_r, z_ab[i]) * ys_z
            f[i] = trapz(ys_z * r, x_r) * const

    ABoutput = ab_dir + \
        split(sed, '/')[-1][:-4] + '.' + split(filter, '/')[-1][:-4] + '.AB'

    print('Writing AB file ', ABoutput)
    put_data(ABoutput, (z_ab, f))


def VegatoAB(m_vega, filter, Vega=Vega):
    cons = AB(f_z_sed(Vega, filter, z=0., units='nu', ccd='yes'))
    return m_vega + cons


def ABtoVega(m_ab, filter, Vega=Vega):
    cons = AB(f_z_sed(Vega, filter, z=0., units='nu', ccd='yes'))
    return m_ab - cons

# Photometric redshift functions


def likelihood(f, ef, ft_z):
    """ 
    Usage: ps[:nz,:nt]=likelihood(f[:nf],ef[:nf],ft_z[:nz,:nt,:nf])
    """
    global minchi2
    axis = ft_z.shape
    nz = axis[0]
    nt = axis[1]

    chi2 = zeros((nz, nt), float)
    ftt = zeros((nz, nt), float)
    fgt = zeros((nz, nt), float)

    ief2 = old_div(1., (ef * ef))
    fgg = add.reduce(f * f * ief2)
    factor = ft_z[:nz, :nt, :] * ief2

    ftt[:nz, :nt] = add.reduce(ft_z[:nz, :nt, :] * factor, -1)
    fgt[:nz, :nt] = add.reduce(f[:] * factor, -1)
    chi2[:nz, :nt] = fgg - old_div(power(fgt[:nz, :nt], 2), ftt[:nz, :nt])

    min_chi2 = min(chi2)
    minchi2 = min(min_chi2)
#   chi2=chi2-minchi2
    chi2 = clip(chi2, 0., -2. * eeps)

    p = where(greater_equal(chi2, -2. * eeps), 0., exp(old_div(-chi2, 2.)))

    norm = add.reduce(add.reduce(p))
    return old_div(p, norm)


class p_c_z_t(object):
    def __init__(self, f, ef, ft_z):
        self.nz, self.nt, self.nf = ft_z.shape
        # Get true minimum of the input data (excluding zero values)

        # Define likelihood quantities taking into account non-observed objects
        self.foo = add.reduce(
            where(less(old_div(f, ef), 1e-4), 0., (old_div(f, ef))**2))
        # Above was wrong: non-detections were ignored as non-observed --DC
        nonobs = greater(reshape(ef, (1, 1, self.nf)) + ft_z * 0., 1.0)
        self.fot = add.reduce(
            # where(nonobs,0.,f[NewAxis,NewAxis,:]*ft_z[:,:,:]/ef[NewAxis,NewAxis,:]**2)
            where(nonobs, 0., reshape(f, (1, 1, self.nf)) * ft_z / reshape(ef, (1, 1, self.nf))**2), -1)
        self.ftt = add.reduce(
            # where(nonobs,0.,ft_z[:,:,:]*ft_z[:,:,:]/ef[NewAxis,NewAxis,:]**2)
            where(nonobs, 0., old_div(ft_z**2, reshape(ef, (1, 1, self.nf))**2)), -1)

        # Define chi2 adding eps to the ftt denominator to avoid overflows
        self.chi2 = where(equal(self.ftt, 0.),
                          self.foo,
                          self.foo - old_div((self.fot**2), (self.ftt + eps)))
        self.chi2_minima = loc2d(self.chi2[:self.nz, :self.nt], 'min')
        self.i_z_ml = self.chi2_minima[0]
        self.i_t_ml = self.chi2_minima[1]
        self.min_chi2 = self.chi2[self.i_z_ml, self.i_t_ml]
        self.likelihood = exp(-0.5 *
                              clip((self.chi2 - self.min_chi2), 0., -2 * eeps))
        # self.likelihood=where(equal(self.chi2,1400.),0.,self.likelihood)

        # Now we add the Bayesian f_tt^-1/2 multiplicative factor to the exponential
        #(we don't multiply it by 0.5 since it is done below together with the chi^2
        # To deal with zero values of ftt we again add an epsilon value.
        self.expo = where(
            equal(self.ftt, 0.),
            self.chi2,
            self.chi2 + log(self.ftt + eps)
        )
        # Renormalize the exponent to preserve dynamical range
        self.expo_minima = loc2d(self.expo, 'min')
        self.min_expo = self.expo[self.expo_minima[0], self.expo_minima[1]]
        self.expo -= self.min_expo
        self.expo = clip(self.expo, 0., -2. * eeps)
        # Clip very low values of the probability
        self.Bayes_likelihood = where(
            equal(self.expo, -2. * eeps),
            0.,
            exp(-0.5 * self.expo))

    def bayes_likelihood(self):
        return self.Bayes_likelihood

    def various_plots(self):
        # Normalize and collapse likelihoods (without prior)
        norm = add.reduce(add.reduce(self.Bayes_likelihood))
        bl = add.reduce(old_div(self.Bayes_likelihood, norm), -1)
        norm = add.reduce(add.reduce(self.likelihood))
        l = add.reduce(old_div(self.likelihood, norm), -1)
        plo = FramedPlot()
        plo.add(Curve(arange(self.nz), bl, color='blue'))
        plo.add(Curve(arange(self.nz), l, color='red'))
        plo.show()
        ask('More?')


class p_c_z_t_color(object):
    def __init__(self, f, ef, ft_z):
        self.nz, self.nt, self.nf = ft_z.shape
        self.chi2 = add.reduce(
            #((f[NewAxis,NewAxis,:]-ft_z[:,:,:])/ef[NewAxis,NewAxis,:])**2
            (old_div((reshape(f, (1, 1, self.nf)) - ft_z), reshape(ef, (1, 1, self.nf))))**2, -1)
        self.chi2_minima = loc2d(self.chi2[:self.nz, :self.nt], 'min')
        self.i_z_ml = self.chi2_minima[0]
        self.i_t_ml = self.chi2_minima[1]
        self.min_chi2 = self.chi2[self.i_z_ml, self.i_t_ml]
        self.likelihood = exp(-0.5 *
                              clip((self.chi2 - self.min_chi2), 0., 1400.))

    def bayes_likelihood(self):
        return self.likelihood


def prior(z, m, info='hdfn', nt=6, ninterp=0, x=None, y=None):
    """Given the magnitude m, produces the prior  p(z|T,m)
    Usage: pi[:nz,:nt]=prior(z[:nz],m,info=('hdfn',nt))
    """
    if info == 'none' or info == 'flat':
        return
    # We estimate the priors at m_step intervals
    # and keep them in a dictionary, and then
    # interpolate them for other values
    m_step = 0.1
    # number of decimals kept
    accuracy = str(len(str(int(old_div(1., m_step)))) - 1)
    exec('from .prior_%s import *' % info)
    from . import prior_hdfn_gen
    global prior_dict
    try:
        len(prior_dict)
    except NameError:
        prior_dict = {}

    # The dictionary keys are values of the
    # magnitud quantized to mstep mags
    # The values of the dictionary are the corresponding
    # prior probabilities.They are only calculated once
    # and kept in the dictionary for future
    # use if needed.
    forma = '%.' + accuracy + 'f'
    m_dict = forma % m
    if m_dict not in prior_dict or info == 'lensing':  # if lensing, the magnitude alone is not enough
        if info != 'lensing':
            prior_dict[m_dict] = prior_hdfn_gen.function(z, float(m_dict), nt)
        else:
            prior_dict[m_dict] = prior_hdfn_gen.function(z, float(m_dict), nt, x, y)
        if ninterp:
            pp_i = prior_dict[m_dict]
            nz = pp_i.shape[0]
            nt = pp_i.shape[1]
            nti = nt + (nt - 1) * int(ninterp)
            tipos = arange(nt) * 1.
            itipos = arange(nti) * 1. / (1. + float(ninterp))
            buffer = zeros((nz, nti)) * 1.
            for iz in range(nz):
                buffer[iz, :] = match_resol(tipos, pp_i[iz, :], itipos)
            prior_dict[m_dict] = buffer
    return prior_dict[m_dict]


def interval(p, x, ci=.99):
    """Gives the limits of the confidence interval
       enclosing ci of the total probability
       i1,i2=limits(p,0.99)
    """
    q1 = old_div((1. - ci), 2.)
    q2 = 1. - q1
    cp = add.accumulate(p)
    if cp[-1] != 1.:
        cp = old_div(cp, cp[-1])
    i1 = searchsorted(cp, q1) - 1
    i2 = searchsorted(cp, q2)
    i2 = minimum(i2, len(p) - 1)
    i1 = maximum(i1, 0)
    return x[i1], x[i2]


def odds(p, x, x1, x2):
    """Estimate the fraction of the total probability p(x)
    enclosed by the interval x1,x2"""
    cp = add.accumulate(p)
    i1 = searchsorted(x, x1) - 1
    i2 = searchsorted(x, x2)
    if i1 < 0:
        return old_div(cp[i2], cp[-1])
    if i2 > len(x) - 1:
        return 1. - old_div(cp[i1], cp[-1])
    return old_div((cp[i2] - cp[i1]), cp[-1])


class p_bayes(object):
    # This class reads the information contained in the files produced by BPZ
    # when the option -PROBS_LITE is on
    def __init__(self, file):
        self.file = file
        dummy = get_2Darray(file)
        self.id_list = list(map(int, list(dummy[:, 0])))
        self.p = dummy[:, 1:]
        del(dummy)
        header = get_header(file)
        header = split(header, '(')[2]
        header = split(header, ')')[0]
        zmin, zmax, dz = list(map(float, tuple(split(header, ','))))
        self.z = arange(zmin, zmax, dz)

    def plot_p(self, id, limits=None):
        if type(id) != type((1,)):
            try:
                j = self.id_list.index(int(id))
                p_j = self.p[j, :]
                if limits == None:
                    connect(self.z, p_j)
                else:
                    connect(self.z, p_j, limits)
            except:
                print('Object %i not in the file %s' % (id, self.file))
            self.prob = old_div(p_j, max(p_j))
        else:
            p = FramedPlot()
            p.frame1.draw_grid = 1
            pall = self.p[0, :] * 0. + 1.
            pmax = 0.
            if limits != None:
                p.xrange = limits[0], limits[1]
                p.yrange = limits[2], limits[3]
            for i in id:
                try:
                    j = self.id_list.index(int(i))
                    p_j = self.p[j, :]
                    if max(p_j) > pmax:
                        pmax = max(p_j)
                    pall *= p_j
                    p.add(Curve(self.z, p_j))
                except:
                    print('Object %i not in the file %s' % (id, self.file))
            p.add(Curve(self.z, pall / max(pall) * pmax, color='red'))
            p.show()
            self.prob = old_div(pall, max(pall))

    def maxima(self, limits=(0., 6.5)):
        g = greater_equal(self.z, limits[0]) * less_equal(self.z, limits[1])
        z, p = multicompress(g, (self.z, self.prob))
        imax = argmax(p)
        xp = add.accumulate(p)
        xp /= xp[-1]
        self.q66 = match_resol(xp, z, array([0.17, 0.83]))
        self.q90 = match_resol(xp, z, array([0.05, 0.95]))
        # print self.q66
        # print self.q90
        return z[imax]

# Misc stuff


def get_datasex(file, cols, purge=1, mag=(2, 99.), emag=(4, .44), flag=(24, 4), detcal='none'):
    """
      Usage:
      x,y,mag,emag=get_datasex('file.cat',(0,1,24,12))
      If purge=1, the function returns the corresponding columns 
      of a SExtractor output file, excluding those objects with 
      magnitude <mag[1], magnitude error <=emag[1] and flag <=flag[1]
      mag[0],emag[0] and flag[0] indicate the columns listing 
      these quantities in the file
      detcal: a detection image was used to create the catalog
      It will be used now to determine which objects are good.
    """
    if type(cols) == type(0):
        nvar = 1
    else:
        nvar = len(cols)

    if purge:
        if nvar > 1:
            datos = get_2Darray(file, cols)
        else:
            datos = get_data(file, cols)
        if detcal == 'none':
            detcal = file
        m, em, f = get_data(detcal, (mag[0], emag[0], flag[0]))
        good = less_equal(f, flag[1]) * less_equal(em,
                                                   emag[1]) * less(m, mag[1])
        datos = compress(good, datos, 0)
        lista = []
        if nvar > 1:
            for i in range(datos.shape[1]):
                lista.append(datos[:, i])
            return tuple(lista)
        else:
            return datos
    else:
        return get_data(file, cols)


class bpz_diagnosis(object):
    def __init__(self, bpz_file='/home/txitxo/bpz/TEST/hdfn.bpz',
                 columns=(1, 4, 5, 6, 9, 10)):
        # columns correspond to the positions of the variables z_b,odds,z_ml, z_s and m_0
        # in the bpz file
        self.zb, self.tb, self.odds, self.zm, self.zs, self.mo = get_data(
            bpz_file, columns)

    def stats(self, type='rms',
              odds_min=0.99,
              mo_min=0., mo_max=99.,
              zs_min=0., zs_max=6.5,
              t_min=0, t_max=100,
              plots='yes',
              thr=.2):
        good = greater_equal(self.mo, mo_min)
        good *= less_equal(self.mo, mo_max)
        good *= greater_equal(self.zs, zs_min)
        good *= less_equal(self.zs, zs_max)
        good *= greater_equal(self.tb, t_min)
        good *= less_equal(self.tb, t_max)
        self.n_total = len(good)
        self.good = good * greater_equal(self.odds, odds_min)
        self.n_selected = sum(self.good)
        self.d = compress(self.good, old_div(
            (self.zb - self.zs), (1. + self.zs)))

        b = stat_robust(self.d, 3, 5)
        b.run()
        self.n_remaining = b.n_remaining
        self.n_outliers = b.n_outliers
        self.rms = b.rms
        self.med = b.median
        self.std_log = std_log(self.d)
        if plots == 'yes':
            # points(compress(self.good,self.zs),compress(self.good,self.zb),(0.,zs_max,0.,zs_max))
            p = FramedPlot()
            xmin = min(compress(self.good, self.zs))
            xmax = max(compress(self.good, self.zs))
            print(xmin, xmax)
            x = arange(xmin, xmax, .01)
            p.add(Curve(x, x, width=3))
            p.add(Curve(x, x + 3. * self.rms * (1. + x), width=1))
            p.add(Curve(x, x - 3. * self.rms * (1. + x), width=1))
            p.add(Points(compress(self.good, self.zs),
                         compress(self.good, self.zb)))
            p.xlabel = r"$z_{spec}$"
            p.ylabel = r"$z_b$"
            p.show()
            if ask("save plot?"):
                name = input("name?")
                p.write_eps(name)


def test():
    """ Tests some functions defined in this module"""

    test = 'flux'
    Testing(test)

    x = arange(912., 10001., .1)
    r = exp(-(x - 3500.)**2 / 2. / 200.**2)
    f = 1. + sin(old_div(x, 100.))

    e_ccd = old_div(add.reduce(f * r * x), add.reduce(r * x))
    e_noccd = old_div(add.reduce(f * r), add.reduce(r))

    r_ccd = flux(x, f, r, ccd='yes', units='lambda')
    r_noccd = flux(x, f, r, ccd='no', units='lambda')

    if abs(1. - old_div(e_ccd, r_ccd)) > 1e-6 or abs(1. - old_div(e_noccd, r_noccd)) > 1e-6:
        raise test

    nu = arange(old_div(1., x[-1]), old_div(1., x[0]),
                1. / x[0] / 1e2) * clight_AHz
    fn = (1. + sin(clight_AHz / 100. / nu)) * clight_AHz / nu / nu
    xn = old_div(clight_AHz, nu)
    rn = match_resol(x, r, xn)
    e_ccd = old_div(add.reduce(fn * rn / nu), add.reduce(old_div(rn, nu)))
    e_noccd = old_div(add.reduce(fn * rn), add.reduce(rn))
    r_ccd = flux(x, f, r, ccd='yes', units='nu')
    r_noccd = flux(x, f, r, ccd='no', units='nu')

    if abs(1. - old_div(e_ccd, r_ccd)) > 1e-6 or abs(1. - old_div(e_noccd, r_noccd)) > 1e-6:
        raise test

    test = 'AB'
    Testing(test)
    if AB(10.**(-.4 * 48.60)) != 0.:
        raise test

    test = 'flux2mag and mag2flux'
    Testing(test)
    m, f = 20., 1e-8
    if mag2flux(m) != f:
        raise test
    if flux2mag(f) != m:
        raise test

    test = 'e_frac2mag and e_mag2frac'
    Testing(test)
    f = 1e8
    df = old_div(1e7, f)
    m = flux2mag(f)
    dm = m - flux2mag(f * (1. + df))
    if abs(e_frac2mag(df) - dm) > 1e-12:
        print(abs(e_frac2mag(df) - dm))
        raise test
    if abs(e_mag2frac(dm) - df) > 1e-12:
        print(e_mag2frac(dm), df)
        raise test

    test = 'etau_madau'
    # Un posible test es generar un plot de la absorpcion a distintos redshifts
    # igual que el que viene en el paper de Madau.

    test = 'f_z_sed'
    Testing(test)
    # Estimate fluxes at different redshift for a galaxy with a f_nu\propto \nu spectrum
    # (No K correction) and check that their colors are constant
    x = arange(1., 10001., 10.)
    f = old_div(1., x)
    put_data(sed_dir + 'test.sed', (x, f))
    z = arange(0., 10., .25)
    b = f_z_sed('test', 'B_Johnson.res', z, ccd='no', units='nu', madau='no')
    v = f_z_sed('test', 'V_Johnson.res', z, ccd='no', units='nu', madau='no')
    c = array(list(map(flux2mag, old_div(b, v))))
    if(sometrue(greater(abs(c - c[0]), 1e-4))):
        print(c - c[0])
        raise test

    test = 'VegatoAB'  # To be done
    test = 'ABtoVega'
    test = 'likelihood'

    # Test: generar un catalogo de galaxias con colores, e intentar recuperar
    # sus redshifts de nuevo utilizando solo la likelihood

    test = 'p_and_minchi2'  # To be done
    test = 'prior'

    test = 'interval'
    test = 'odds'

    test = ' the accuracy of our Johnson-Cousins-Landolt Vega-based zero-points'
    Testing(test)
    # filters=['U_Johnson.res','B_Johnson.res','V_Johnson.res','R_Cousins.res',
    #'I_Cousins.res']

    filters = [
        'HST_ACS_WFC_F435W',
        'HST_ACS_WFC_F475W',
        'HST_ACS_WFC_F555W',
        'HST_ACS_WFC_F606W',
        'HST_ACS_WFC_F625W',
        'HST_ACS_WFC_F775W',
        'HST_ACS_WFC_F814W',
        'HST_ACS_WFC_F850LP'
    ]

    ab_synphot = array([
        -0.10719,
        -0.10038,
        8.743e-4,
        0.095004,
        0.174949,
        0.40119,
        0.44478,
        0.568605
    ])

    f_l_vega = array([
        6.462e-9,
        5.297e-9,
        3.780e-9,
        2.850e-9,
        2.330e-9,
        1.270e-9,
        1.111e-9,
        7.78e-10])

    print('     f_l for Vega')
    sufix = 'cgs A^-1'
    print('                               f_lambda(Vega)     synphot(IRAF)   difference %')
    for i in range(len(filters)):
        f_vega = f_z_sed(Vega, filters[i], ccd='yes')
        tupla = (ljust(filters[i], 16), f_vega,
                 f_l_vega[i], f_vega / f_l_vega[i] * 100. - 100.)
        print('     %s         %.6e       %.6e      %.4f' % tupla + "%")

    print('    ')
    print('    AB zeropoints for Vega ')
    sufix = 'cgs Hz'
    tipo = 'nu'
    print("                                AB zero point     synphot(IRAF)   difference")
    for i in range(len(filters)):
        f_vega = f_z_sed(Vega, filters[i], units=tipo, ccd='yes')
        tupla = (ljust(filters[i], 16), AB(f_vega),
                 ab_synphot[i], AB(f_vega) - ab_synphot[i])
        print('     %s         %.6f       %.6f      %.6f' % tupla)

    print('    ')
    print('    AB zeropoints for a c/lambda^2 spectrum (flat in nu)')
    sufix = 'cgs Hz'
    tipo = 'nu'
    print("                                 Result             Expected  ")
    for i in range(len(filters)):
        f_flat = f_z_sed('flat', filters[i], units=tipo, ccd='yes')
        tupla = (ljust(filters[i], 16), AB(f_flat), 0.)
        print('     %s         %.6e       %.6f' % tupla)

    print('')
    print('         Everything OK    in   bpz_tools ')
    print('')


if __name__ == '__main__':
    test()
