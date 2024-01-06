import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.utils import flux_model, params_valid
from superphot_plus.surveys.surveys import Survey

DEFAULT_MAX_FLUX = 1.0

def trunc_gauss(quantile, clip_a, clip_b, mean, std):
    """Truncated Gaussian distribution.

    Parameters
    ----------
    quantile : float
        The quantile at which to evaluate the ppf. Should be a value
        between 0 and 1.
    clip_a : float
        Lower clip value.
    clip_b : float
        Upper clip value.
    mean : float
        Mean of the distribution.
    std : float
        Standard deviation of the distribution.

    Returns
    -------
    scipy.stats.truncnorm.ppf
        Percent point function of the truncated Gaussian.
    """
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.ppf(quantile, a, b, loc=mean, scale=std)


def create_prior(cube, priors=Survey.ZTF().priors):
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
    priors=Survey.ZTF().priors
    all_priors = priors.to_numpy().T

    # Precompute the vectors of trunc_gauss a and b values.
    tg_a = (all_priors[0] - all_priors[2]) / all_priors[3]
    tg_b = (all_priors[1] - all_priors[2]) / all_priors[3]
    
    return truncnorm.ppf(
        cube, tg_a, tg_b, loc=all_priors[2], scale=all_priors[3]
    )

def ztf_noise_model(mag, band, snr_range_g=None, snr_range_r=None):
    """A very, very simple noise model which assumes the dimmest magnitude is at SNR = 1,
    and the brightest mag is at SNR = 10.

    Parameters
    ----------
    mag : np.ndarray
        Observed magnitudes.
    band : np.ndarray
        Observed bands (g or r).
    snr_range_g : tuple
        Range of signal-to-noise ratios desired in g-band. Defaults to [1, 10]
    snr_range_r : tuple
        Range of signal-to-noise ratios desired in r-band. Defaults to [1, 10]

    Returns
    ----------
    snr : np.ndarray
        Signal-to-noise ratios (SNR) of the observations.
    """
    if not snr_range_g:
        snr_range_g = [1, 10]
    if not snr_range_r:
        snr_range_r = [1, 10]

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


def create_clean_models(nmodels, num_times=100, priors=Survey.ZTF().priors):
    """Generate 'clean' (noiseless) models from the prior

    Parameters
    ----------
    nmodels : int
        The number of models you want to generate.
    num_times : int, optional
        The number of timesteps to use. Default = 100
    bands : list, optional
        The ordered list of bands to use. Default = ['r', 'g']
    ref_band : str, optional
        The reference band. Default = 'r'

    Returns
    -------
    params : array-like of numpy arrays
        The array of parameters used to generate each model.
    lcs : array-like of numpy arrays
        The array of individual light curves for each model generated.
    """
    params = []
    lcs = []
    
    bands = priors.ordered_bands
    ref_band = priors.reference_band
    
    tdata = np.linspace(-100, 100, num_times)
    bdata = np.asarray([bands[i % len(bands)] for i in range(num_times)], dtype=str)
    edata = np.asarray([1e-6] * num_times, dtype=float)

    while len(lcs) < nmodels:
        cube = np.random.uniform(0, 1, 7 * len(bands))
        cube = create_prior(cube, priors)
        A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7]  # pylint: disable=unused-variable

        # Try again if we picked invalid priors.
        if not params_valid(beta, 10**gamma, 10**tau_rise, 10**tau_fall):
            continue
            
        params.append(cube)

        f_model = flux_model(
            cube, tdata, bdata,
            DEFAULT_MAX_FLUX, bands, ref_band
        )
        lcs.append(Lightcurve(tdata, f_model, edata, bdata))

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
    # This is going to random simulate some observation every 2-3 days across 2 filters
    num_observations = 130
    tdata = np.random.uniform(-100, 100, num_observations)
    filter_data = np.random.choice(["g", "r"], size=num_observations)

    found_valid = False
    num_tried = 0

    # Now re-attempts to regenerate until it gets a "good" model
    while not found_valid and num_tried < 100:
        cube = np.random.uniform(0, 1, 14)
        params = create_prior(np.copy(cube))
        A, beta, gamma, t0, tau_rise, tau_fall, es = params[:7]  # pylint: disable=unused-variable
        found_valid = params_valid(
            beta, 10**gamma, 10**tau_rise, 10**tau_fall
        )
        num_tried += 1

    if not found_valid:
        return "Failure"

    f_model = flux_model(
        params, tdata, filter_data,
        DEFAULT_MAX_FLUX,
        ["r", "g"], "r"
    )
    snr = ztf_noise_model(f_model, filter_data)

    gind = np.where(snr > 3)  # any points with SNR < 3 are ignored
    snr = snr[gind]
    f_model = f_model[gind]
    tdata = tdata[gind]
    filter_data = filter_data[gind]
    print(f_model)
    sigmas = f_model / snr

    dirty_model = f_model + np.random.normal(0, sigmas)

    if plot:
        plt.scatter(tdata, dirty_model, color=filter_data)
        plt.errorbar(tdata, dirty_model, yerr=sigmas, color="grey", alpha=0.2, linestyle="None")
        plt.xlabel("Time (days)")
        plt.ylabel("Flux (arbitrary units)")
        plt.show()
    return params[:7], tdata, filter_data, dirty_model, sigmas


# Can run this with create_model(plot=True)
if __name__ == "__main__":
    create_ztf_model(plot=True)