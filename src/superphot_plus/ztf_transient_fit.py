import contextlib
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from dynesty import NestedSampler
from scipy.optimize import curve_fit
from scipy.stats import truncnorm

from superphot_plus.constants import DLOGZ, MAX_ITER, NLIVE
from superphot_plus.file_utils import get_posterior_filename, has_posterior_samples
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.plotting import plot_sampling_lc_fit
from superphot_plus.priors.fitting_priors import MultibandPriors, CurvePriors
from superphot_plus.utils import flux_model

from superphot_plus.file_paths import FIT_PLOTS_FOLDER


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(
        joblib.parallel.BatchCompletionCallBack
    ):  # pylint: disable=missing-class-docstring
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def params_valid(beta, gamma, tau_rise, tau_fall):
    """Check if parameters are valid given certain model constraints.

    Parameters
    ----------
    beta : float
        Parameter beta.
    gamma : float
        Parameter gamma.
    tau_rise : float
        Parameter tau_rise.
    tau_fall : float
        Parameter tau_fall.

    Returns
    -------
    bool
        True if parameters are valid, False otherwise.
    """
    if tau_fall > 1.0 / beta:
        return False

    if gamma > (1.0 - beta * tau_fall) / beta:
        return False

    if tau_rise * (1.0 + np.exp(gamma / tau_rise)) < tau_fall:
        return False

    return True


def run_mcmc(lc, t0_lim=None, plot=False, rstate=None, telescope="ZTF"):
    """Runs dynesty importance nested sampling on a single light curve; returns set
    of equally weighted posteriors (sets of fit parameters).

    Parameters
    ----------
    lc : Lightcurve object
        The lightcurve of interest
    t0_lim : float, optional
        Upper limit for t0. Defaults to None.
    plot : bool, optional
        Flag to enable/disable plotting. Defaults to False.
    rstate : int, optional
        Random state that is seeded. if none, use machine entropy.
    telescope : str, optional
        Determines band and prior information. Defaults to ZTF.

    Returns
    -------
    np.ndarray or None
        Numpy array containing the equally weighted posteriors, or None
        if the data is invalid.
    """
    if telescope == "ZTF":
        all_priors_cls = MultibandPriors.load_ztf_priors()
        all_priors = all_priors_cls.to_numpy().T
        ref_band = "r" # maybe define within MultibandPriors class

    n_params = len(all_priors.T)
    unique_bands = all_priors_cls.ordered_bands
    ref_band_idx = np.argmax(unique_bands == ref_band)

    # Require data in both the g and r bands.
    for ub in unique_bands:
        if lc.obs_count(ub) == 0:
            return None

    tdata = lc.times
    fdata = lc.fluxes
    ferrdata = lc.flux_errors
    bdata = lc.bands

    # Precompute the information about the maximum flux in the r band.
    where_ref_band = (bdata == ref_band)
    max_index = np.argmax(lc.fluxes[where_ref_band] - np.abs(lc.flux_errors[where_ref_band]))
    max_flux = lc.fluxes[where_ref_band][max_index] - np.abs(lc.flux_errors[where_ref_band][max_index])
    max_flux_loc = tdata[where_ref_band][max_index]
    
    start_idx = 7*ref_band_idx

    # Create copies of the prior vectors with the value for t0 overwritten for the
    # current lightcurve.
    prior_clip_a = np.copy(all_priors[0])
    prior_clip_a[start_idx+3] = np.amin(tdata) - 50.0

    prior_clip_b = np.copy(all_priors[1])
    prior_clip_b[start_idx+3] = np.amax(tdata) + 50.0

    prior_mean = np.copy(all_priors[2])
    prior_mean[start_idx+3] = max_flux_loc

    prior_std = np.copy(all_priors[3])
    prior_std[start_idx+3] = 20.0

    # Precompute the vectors of trunc_gauss a and b values.
    tg_a = (prior_clip_a - prior_mean) / prior_std
    tg_b = (prior_clip_b - prior_mean) / prior_std

    def create_prior(cube):
        """Creates prior for pymultinest, where each side of the "cube"
        is a value sampled between 0 and 1 representing each parameter.

        Parameters
        ----------
        cube : np.ndarray
            Array of parameters.

        Returns
        -------
        np.ndarray
            Updated array of parameters.
        """
        # Compute the truncated Gaussian distribution for all values at once.
        tg_vals = truncnorm.ppf(cube, tg_a, tg_b, loc=prior_mean, scale=prior_std)

        cube[start_idx] = max_flux * 10 ** tg_vals[start_idx]  # log-uniform for A from 1.0x to 16x of max flux
        cube[start_idx+1] = tg_vals[start_idx+1]  # beta UPDATED, looks more Lorentzian so widened by 1.5x
        cube[start_idx+2] = 10 ** tg_vals[start_idx+2]  # very broad Gaussian temporary solution for gamma
        cube[start_idx+3] = tg_vals[start_idx+3]
        cube[start_idx+4:start_idx+7] = 10 ** tg_vals[start_idx+4:start_idx+7]  # taurise, UPDATED

        # all other bands
        cube[:start_idx] = tg_vals[:start_idx]  # all non-ref params
        cube[start_idx+7:] = tg_vals[start_idx+7:]
    
        return cube

    def create_logL(cube):
        """Define the log-likelihood function.

        Is proportional to chi-squared of data's fit to generated flux
        model.

        Parameters
        ----------
        cube : np.ndarray
            Array of parameters.

        Returns
        -------
        float
            Log-likelihood value.
        """
        f_model = flux_model(cube, tdata, bdata, unique_bands, ref_band)
        extra_sigma_arr = np.ones(len(tdata)) * cube[6] * max_flux
        
        for band_idx, ordered_band in enumerate(unique_bands):
            if ordered_band == ref_band:
                continue
            extra_sigma_arr[bdata == ordered_band] *= cube[7*band_idx+6]

        sigma_sq = ferrdata**2 + extra_sigma_arr**2
        logL = np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq)) - 0.5 * (f_model - fdata) ** 2 / sigma_sq)
        return logL

    st = time.time()  # pylint: disable=unused-variable

    sampler = NestedSampler(
        create_logL, create_prior, n_params, sample="rwalk", bound="single", nlive=NLIVE, rstate=rstate
    )
    sampler.run_nested(maxiter=MAX_ITER, dlogz=DLOGZ, print_progress=False)
    res = sampler.results

    red_chisq = res.logl / len(tdata)
    samples = res.samples
    eq_wt_samples = res.samples_equal(rstate=rstate)

    orig_idxs = np.array([np.argmin(np.sum((e - samples) ** 2, axis=1)) for e in eq_wt_samples])
    eq_wt_red_chisq = red_chisq[orig_idxs]

    eq_wt_samples = np.append(eq_wt_samples, eq_wt_red_chisq[np.newaxis, :].T, 1)

    if plot:  # pragma: no cover
        if lc.name is None:
            raise ValueError("Missing file name for plotting files.")
        plot_sampling_lc_fit(lc.name, FIT_PLOTS_FOLDER, tdata, fdata, ferrdata, bdata, eq_wt_samples)

    return eq_wt_samples


def run_curve_fit(filename, output_dir, plot=True):
    """Run curve fit on data file.

    Parameters
    ----------
    filename : str
        Name of the data file.
    output_dir : str
        Directory to place plots, if generated.
    plot : boolean
        Whether or not to draw lightcurve fit plots.

    Returns
    -------
    tuple or None
        Tuple containing the fitted parameters for each
        bands, or None if the required data is missing.
    """
    lightcurve = Lightcurve.from_file(filename)

    tdata = lightcurve.times
    fdata = lightcurve.fluxes
    ferrdata = lightcurve.flux_errors
    bdata = lightcurve.bands

    mb_priors = MultibandPriors.load_ztf_priors()
    ref_band = mb_priors.reference_band

    ## Exit early if we don't some data points in each band.
    for band in mb_priors.bands_in_order():
        if (tdata[bdata == band] is None) or (len(tdata[bdata == band]) == 0):
            return None

    max_flux = np.max(fdata[bdata == ref_band] - np.abs(ferrdata[bdata == ref_band]))

    max_flux_loc = tdata[bdata == ref_band][
        np.argmax(fdata[bdata == ref_band] - np.abs(ferrdata[bdata == ref_band]))
    ]

    bounds = (
        [max_flux * 10 ** (-0.2), 0.0, -2.0, np.amin(tdata) - 50.0, -1.0, 0.5],
        [max_flux * 10 ** (1.2), 0.02, 2.5, np.amax(tdata) + 50.0, 3.0, 4.0],
    )

    def flux_model_smooth(t_data, A, beta, gamma, t0, tau_rise, tau_fall):
        """Tests the smooth model implemented in ALERCE's classifier.

        Parameters
        ----------
        t_data : array-like
            Time data.
        A : float
            Parameter A.
        beta : float
            Parameter beta.
        gamma : float
            Parameter gamma.
        t0 : float
            Parameter t0.
        tau_rise : float
            Parameter tau_rise.
        tau_fall : float
            Parameter tau_fall.

        Returns
        -------
        np.ndarray
            Flux model.
        """
        gamma = 10.0**gamma
        tau_rise = 10.0**tau_rise
        tau_fall = 10.0**tau_fall

        sigma_arg = (t_data - gamma - t0) / 3.0
        sigma = 1.0 / (1.0 + np.exp(-sigma_arg))
        if not params_valid(beta, gamma, tau_rise, tau_fall):
            return 1e10 * np.ones(len(t_data))

        phase = t_data - t0
        f_model = (
            A
            / (1.0 + np.exp(-phase / tau_rise))
            * (1.0 - beta * gamma)
            * np.exp((gamma - phase) / tau_fall)
            * sigma
        )
        f_model += A / (1.0 + np.exp(-phase / tau_rise)) * (1.0 - beta * phase) * (1.0 - sigma)

        return f_model

    opts = {}

    ## Get parameters for the reference band.
    priors = mb_priors.bands[ref_band]

    prior_array = [
        max_flux,
        priors.beta.mean,
        priors.gamma.mean,
        max_flux_loc,
        priors.tau_rise.mean,
        priors.tau_fall.mean,
    ]

    ref_results, _ = curve_fit(  # pylint: disable=unused-variable,unbalanced-tuple-unpacking
        flux_model_smooth,
        tdata[bdata == ref_band],
        fdata[bdata == ref_band],
        p0=prior_array,
        sigma=ferrdata[bdata == ref_band],
        bounds=bounds,
        maxfev=100000,
    )

    opts[ref_band] = ref_results

    for band, priors in mb_priors.bands.items():
        if band == ref_band:
            continue

        prior_array = [
            priors.amp.mean,
            priors.beta.mean,
            priors.gamma.mean,
            priors.t_0.mean,
            priors.tau_rise.mean,
            priors.tau_fall.mean,
        ]
        prior_array = prior_array * ref_results

        popt, _ = curve_fit(  # pylint: disable=unused-variable,unbalanced-tuple-unpacking
            flux_model_smooth,
            tdata[bdata == band],
            fdata[bdata == band],
            p0=prior_array,
            sigma=ferrdata[bdata == band],
            bounds=bounds,
            maxfev=100000,
        )

        opts[band] = popt

    if plot:
        trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

        for band in mb_priors.bands_in_order():
            plt.errorbar(
                tdata[bdata == band],
                fdata[bdata == band],
                yerr=ferrdata[bdata == band],
                c=band,
                label=band,
                fmt="o",
            )
            plt.plot(trange_fine, flux_model_smooth(trange_fine, *opts[band]), c=band, lw=1)

        plt.xlabel("MJD")
        plt.ylabel("Flux")
        plt.title(lightcurve.name)

        plt.savefig(os.path.join(output_dir, f"{lightcurve.name}.png"))

    return opts


def dynesty_single_curve(lc, output_dir, skip_if_exists=True, rstate=None):
    """Perform model fitting using dynesty on a single light curve.

    This function runs the dynesty importance nested sampling algorithm
    on a single light curve. It saves the resulting equally weighted
    posterior samples to a compressed NumPy archive file.

    Parameters
    ----------
    lc : Lightcurve object
        The light curve of interest.
    output_dir : str
        The directory where the output file will be saved.
    skip_if_exists : bool, optional
        Flag indicating whether to skip fitting if the output file
        already exists. Defaults to true.
    rstate : int, optional
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    sample_mean: numpy array
        Return the mean of the MCMC samples or None if the fitting is
        skipped or encounters an error.
    """
    if lc.name is None or lc.name == "":
        raise ValueError("Empty light curve name.")

    os.makedirs(output_dir, exist_ok=True)
    if skip_if_exists and has_posterior_samples(lc_name=lc.name, fits_dir=output_dir, sampler="dynesty"):
        return None

    eq_samples = run_mcmc(lc, plot=False, rstate=rstate)
    if eq_samples is None:
        return None
    sample_mean = np.mean(eq_samples, axis=0)
    print(sample_mean)

    posterior_filename = get_posterior_filename(lc.name, output_dir, "dynesty")

    np.savez_compressed(posterior_filename, eq_samples)
    return sample_mean


def dynesty_single_file(test_fn, output_dir, skip_if_exists=True, rstate=None, t0_lim=None):
    """Perform model fitting using dynesty on a single data file.

    This function runs the dynesty importance nested sampling algorithm
    on a single data file. It saves the resulting equally weighted
    posterior samples to a compressed NumPy archive file.

    Parameters
    ----------
    test_fn : str
        The path of the data file to be analyzed.
    output_dir : str
        The directory where the output file will be saved.
    skip_if_exists : bool, optional
        Flag indicating whether to skip fitting if the output file
        already exists. Defaults to true.
    rstate : int, optional
        Random state that is seeded. if none, use machine entropy.
    t0_lim : float, optional
        Upper limit for t0. Defaults to None.

    Returns
    -------
    sample_mean: numpy array
        Return the mean of the MCMC samples or None if the fitting is
        skipped or encounters an error.
    """
    lc = Lightcurve.from_file(test_fn)
    sample_mean = dynesty_single_curve(lc, output_dir, skip_if_exists, rstate)
    return sample_mean
