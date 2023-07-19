import contextlib
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from dynesty import NestedSampler
from dynesty import utils as dyfunc
from scipy.optimize import curve_fit
from scipy.stats import truncnorm

from superphot_plus.file_utils import read_single_lightcurve
from superphot_plus.plotting import plot_sampling_lc_fit
from superphot_plus.utils import flux_model

from .constants import *  # pylint: disable=wildcard-import
from .file_paths import FIT_PLOTS_FOLDER


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


def run_mcmc(filename, t0_lim=None, plot=False, rstate=None):
    """Runs dynesty importance nested sampling on datafile; returns set
    of equally weighted posteriors (sets of fit parameters).

    Parameters
    ----------
    filename : str
        Data file name.
    t0_lim : float, optional
        Upper limit for t0. Defaults to None.
    plot : bool, optional
        Flag to enable/disable plotting. Defaults to False.
    rstate : int, optional
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    np.ndarray or None
        Numpy array containing the equally weighted posteriors, or None
        if the data is invalid.
    """
    ref_band_idx = 1  # red band # pylint: disable=unused-variable

    prefix = filename.split("/")[-1][:-4]

    print(prefix)
    n_params = 14

    tdata, fdata, ferrdata, bdata = read_single_lightcurve(filename, t0_lim)

    if (tdata[bdata == "r"] is None) or (len(tdata[bdata == "r"]) == 0):
        return None
    if (tdata[bdata == "g"] is None) or (len(tdata[bdata == "g"]) == 0):
        return None

    max_flux = np.max(fdata[bdata == "r"] - np.abs(ferrdata[bdata == "r"]))

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

        cube[0] = max_flux * 10 ** (
            trunc_gauss(cube[0], *PRIOR_A)
        )  # log-uniform for A from 1.0x to 16x of max flux
        cube[1] = trunc_gauss(cube[1], *PRIOR_BETA)  # beta UPDATED, looks more Lorentzian so widened by 1.5x
        cube[2] = 10 ** trunc_gauss(cube[2], *PRIOR_GAMMA)  # very broad Gaussian temporary solution for gamma
        max_flux_loc = tdata[np.argmax(fdata[bdata == "r"] - np.abs(ferrdata[bdata == "r"]))]
        cube[3] = trunc_gauss(cube[3], np.amin(tdata) - 50.0, np.amax(tdata) + 50.0, max_flux_loc, 20.0)  # t0
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
        f_model = flux_model(cube, tdata, bdata)
        extra_sigma_arr = np.ones(len(tdata)) * cube[6] * max_flux
        extra_sigma_arr[bdata == "g"] *= cube[13]

        sigma_sq = ferrdata**2 + extra_sigma_arr**2
        logL = np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq)) - 0.5 * (f_model - fdata) ** 2 / sigma_sq)
        return logL

    st = time.time()  # pylint: disable=unused-variable

    sampler = NestedSampler(
        create_logL, create_prior, n_params, sample="rwalk", bound="single", nlive=NLIVE, rstate=rstate
    )
    sampler.run_nested(maxiter=MAX_ITER, dlogz=DLOGZ, print_progress=False)
    res = sampler.results

    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])

    eq_wt_samples = dyfunc.resample_equal(samples, weights, rstate=rstate)

    if plot:
        plot_sampling_lc_fit(prefix, FIT_PLOTS_FOLDER, tdata, fdata, ferrdata, bdata, eq_wt_samples)
    return eq_wt_samples


def dynesty_single_file(test_fn, output_dir, skip_if_exists=True, rstate=None):
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

    Returns
    -------
    None
        Returns None if the fitting is skipped or encounters an error.
    """
    # try:

    os.makedirs(output_dir, exist_ok=True)
    prefix = test_fn.split("/")[-1][:-4]
    if skip_if_exists and os.path.exists(os.path.join(output_dir, f"{prefix}_eqwt.npz")):
        return None

    eq_samples = run_mcmc(test_fn, plot=False, rstate=rstate)
    if eq_samples is None:
        return None
    print(np.mean(eq_samples, axis=0))

    np.savez_compressed(os.path.join(output_dir, f"{prefix}_eqwt_dynesty.npz"), eq_samples)
