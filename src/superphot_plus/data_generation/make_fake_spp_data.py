import numpy as np
from scipy.stats import truncnorm

from ..utils import flux_model
from ..constants import *
from ..ztf_transient_fit import trunc_gauss, params_valid

DEFAULT_MAX_FLUX = 1.0


def create_prior(cube):
    """Creates prior for dynesty, where each side of the "cube"
    is a value sampled between 0 and 1 representing each parameter.
    Slightly altered from ztf_transient_fit.py

    Parameters
    ----------
    cube : np.ndarray
        Array of parameters.

    Returns
    -------
    np.ndarray
        Updated array of parameters.
    """
    cube[0] = DEFAULT_MAX_FLUX * 10 ** (
        trunc_gauss(cube[0], *PRIOR_A)
    )  # log-uniform for A from 1.0x to 16x of max flux
    cube[1] = trunc_gauss(cube[1], *PRIOR_BETA)  # beta UPDATED, looks more Lorentzian so widened by 1.5x
    cube[2] = 10 ** trunc_gauss(cube[2], *PRIOR_GAMMA)  # very broad Gaussian temporary solution for gamma
    cube[3] = trunc_gauss(cube[3], *PRIOR_T0)  # t0
    cube[4] = 10 ** (trunc_gauss(cube[4], *PRIOR_TAU_RISE))  # taurise, UPDATED
    cube[5] = 10 ** (trunc_gauss(cube[5], *PRIOR_TAU_FALL))  # tau fall UPDATED
    cube[6] = 10 ** (trunc_gauss(cube[6], *PRIOR_EXTRA_SIGMA))  # lognormal for extrasigma, UPDATED

    # green band
    cube[7] = trunc_gauss(cube[7], *PRIOR_A_g)  # A UPDATED
    cube[8] = trunc_gauss(cube[8], *PRIOR_BETA_g)  # beta UPDATED
    cube[9] = trunc_gauss(cube[9], *PRIOR_GAMMA_g)  # gamma, GAUSSIAN not Lorentzian
    cube[10] = trunc_gauss(cube[10], *PRIOR_T0_g)  # t0 UPDATED
    cube[11] = trunc_gauss(cube[11], *PRIOR_TAU_RISE_g)  # taurise UPDATED, Gaussian
    cube[12] = trunc_gauss(cube[12], *PRIOR_TAU_FALL_g)  # taufall UPDATED
    cube[13] = trunc_gauss(cube[13], *PRIOR_EXTRA_SIGMA_g)  # extra sigma UPDATED, Gaussian

    return cube


def ztf_noise_model(mag, band, snr_range_g=[1, 10], snr_range_r=[1, 10]):
    """A very, very simple noise model which assumes the dimmest magnitude is at SNR = 1, and the brightest mad is at SNR = 10.

    Parameters
    ----------
    mag : np.ndarray
        Observed magnitudes.
    band : np.ndarray
        Observed bands (g or r).
    snr_range_g : tuple
        Range of signal-to-noise ratios desired in g-band.
    snr_range_r : tuple
        Range of signal-to-noise ratios desired in r-band.

    Returns
    ----------
    snr : np.ndarray
        Signal-to-noise ratios (SNR) of the observations.
    """

    snr = mag * 0  # set up a dummy array for the snr
    gind_g = np.where(band == "g")  # let's do g-band first
    range_g = np.max(mag[gind_g]) - np.min(mag[gind_g])
    snr[gind_g] = (snr_range_g[1] - snr_range_g[0]) * (
        mag[gind_g] - np.min(mag[gind_g])
    ) / range_g + snr_range_g[0]

    gind_r = np.where(band == "r")  # r-band
    range_r = np.max(mag[gind_r]) - np.min(mag[gind_r])
    snr[gind_r] = (snr_range_r[1] - snr_range_r[0]) * (
        mag[gind_r] - np.min(mag[gind_r])
    ) / range_r + snr_range_r[0]
    return snr


def create_clean_models(nmodels, num_times=100):
    """Generate 'clean' (noiseless) models from the prior

    Parameters
    ----------
    nmodels : int
        The number of models you want to generate.
    num_times : int, optional
        The number of timesteps to use.

    Returns
    -------
    params : array-like of numpy arrays
        The array of parameters used to generate each model.
    lcs : array-like of numpy arrays
        The array of individual light curves for each model generated.
    """
    params = []
    lcs = []

    tdata = np.linspace(-100, 100, num_times)
    bdata = np.asarray(["g"] * num_times, dtype=str)
    edata = np.asarray([1e-6] * num_times, dtype=float)

    while len(lcs) < nmodels:
        cube = np.random.uniform(0, 1, 14)
        cube = create_prior(cube)
        A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7]  # pylint: disable=unused-variable

        # Try again if we picked invalid priors.
        if not params_valid(beta, gamma, tau_rise, tau_fall):
            continue
        params.append(cube)

        f_model = flux_model(cube, tdata, bdata)
        lcs.append(np.array([tdata, f_model, edata, bdata]))

    return params, lcs


def create_ztf_model(plot=False):
    """Generate realisitic-ish ZTF light curves from the Superphot+ prior.

    Parameters
    ----------
    plot : bool
        Whether resulting light curve is plotted and saved. Defaults to False.

    Returns
    ----------
    params : np.ndarray
        Set of parameters used to generate model.
    tdata : np.ndarray
        Time values of each datapoint.
    filter_data : np.ndarray
        Filter corresponding to each datapoint.
    dirty_model : np.ndarray
        Dirty flux values at each time value.
    sigmas : np.ndarray
        Uncertainties of each dirty flux value.
    """

    # Randomly sample from parameter space and from data/filters
    cube = np.random.uniform(0, 1, 14)

    # This is going to random simulate some observation every 2-3 days across 2 filters
    num_observations = 130
    tdata = np.random.uniform(-100, 100, num_observations)
    filter_data = np.random.choice(["g", "r"], size=num_observations)

    cube = create_prior(cube)
    A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7]  # pylint: disable=unused-variable

    found_valid = False
    num_tried = 0

    # Now re-attempts to regenerate until it gets a "good" model
    while not found_valid and num_tried < 100:
        cube = create_prior(cube, tdata)
        A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7]  # pylint: disable=unused-variable
        found_valid = params_valid(beta, gamma, tau_rise, tau_fall)
        num_tried += 1

    if not found_valid:
        return "Failure"

    f_model = flux_model(cube, tdata, filter_data)
    snr = ztf_noise_model(f_model, filter_data)

    gind = np.where(snr > 3)  # any points with SNR < 3 are ignored
    snr = snr[gind]
    f_model = f_model[gind]
    tdata = tdata[gind]
    filter_data = filter_data[gind]
    sigmas = f_model / snr

    dirty_model = f_model + np.random.normal(0, sigmas)

    if plot:
        plt.scatter(tdata, dirty_model, color=filter_data)
        plt.errorbar(tdata, dirty_model, yerr=sigmas, color="grey", alpha=0.2, linestyle="None")
        plt.xlabel("Time (days)")
        plt.ylabel("Flux (arbitrary units)")
        plt.show()
    return cube[:, 7], tdata, filter_data, dirty_model, sigmas


# Can run this with create_model(plot=True)
