"""MCMC sampling using dynesty."""

from typing import List

import numpy as np
from dynesty import NestedSampler
from scipy.stats import truncnorm

from superphot_plus.constants import DLOGZ, MAX_ITER, NLIVE
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.samplers.sampler import Sampler
from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import flux_model


class DynestySampler(Sampler):
    """ "MCMC sampling using dynesty."""

    def __init__(self):
        pass

    def run_single_curve(
        self, lightcurve: Lightcurve, priors: MultibandPriors, rstate=None, **kwargs
    ) -> PosteriorSamples:
        """Perform model fitting using dynesty on a single light curve.

        This function runs the dynesty importance nested sampling algorithm
        on a single light curve.

        Parameters
        ----------
        lightcurve : Lightcurve object
            The light curve of interest.
        rstate : int, optional
            Random state that is seeded. if none, use machine entropy.
        plot : bool, optional
            Flag to enable/disable plotting. Defaults to False.
        rstate : int, optional
            Random state that is seeded. if none, use machine entropy.

        Returns
        -------
        samples: PosteriorSamples
            Return the MCMC samples or None if the fitting is
            skipped or encounters an error.
        """
        return run_mcmc(lightcurve, priors=priors, rstate=rstate)

    def run_multi_curve(self, lightcurves, priors, **kwargs) -> List[PosteriorSamples]:
        """Not yet implemented."""
        raise NotImplementedError


def run_mcmc(lightcurve, priors=Survey.ZTF().priors, rstate=None):
    """Runs dynesty importance nested sampling on a single light curve; returns set
    of equally weighted posteriors (sets of fit parameters).

    Parameters
    ----------
    lightcurve : Lightcurve object
        The lightcurve of interest
    priors : str, optional
        Prior information. Defaults to ZTF.
    rstate : int, optional
        Random state that is seeded. if none, use machine entropy.

    Returns
    -------
    PosteriorSamples or None
        Equally weighted posteriors, or None if the data is invalid.
    """
    all_priors = priors.to_numpy().T
    ref_band = priors.reference_band

    n_params = len(all_priors.T)
    unique_bands = priors.ordered_bands
    ref_band_idx = np.argmax(unique_bands == ref_band)

    # Require data in all bands
    for band in unique_bands:
        if lightcurve.obs_count(band) == 0:
            return None

    # Precompute the information about the maximum flux in the reference band.
    max_flux, max_flux_loc = lightcurve.find_max_flux(band=ref_band)

    start_idx = 7 * ref_band_idx

    # Create copies of the prior vectors with the value for t0 overwritten for the
    # current lightcurve.
    prior_clip_a = np.copy(all_priors[0])
    prior_clip_a[start_idx + 3] = np.amin(lightcurve.times) - 50.0

    prior_clip_b = np.copy(all_priors[1])
    prior_clip_b[start_idx + 3] = np.amax(lightcurve.times) + 50.0

    prior_mean = np.copy(all_priors[2])
    prior_mean[start_idx + 3] = max_flux_loc

    prior_std = np.copy(all_priors[3])
    prior_std[start_idx + 3] = 20.0

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

        cube[start_idx] = (
            max_flux * 10 ** tg_vals[start_idx]
        )  # log-uniform for A from 1.0x to 16x of max flux
        cube[start_idx + 1] = tg_vals[start_idx + 1]  # beta UPDATED, looks more Lorentzian so widened by 1.5x
        cube[start_idx + 2] = 10 ** tg_vals[start_idx + 2]  # very broad Gaussian temporary solution for gamma
        cube[start_idx + 3] = tg_vals[start_idx + 3]
        cube[start_idx + 4 : start_idx + 7] = 10 ** tg_vals[start_idx + 4 : start_idx + 7]  # taurise, UPDATED

        # all other bands
        cube[:start_idx] = tg_vals[:start_idx]  # all non-ref params
        cube[start_idx + 7 :] = tg_vals[start_idx + 7 :]

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
        f_model = flux_model(cube, lightcurve.times, lightcurve.bands, unique_bands, ref_band)
        extra_sigma_arr = np.ones(len(lightcurve.times)) * cube[6] * max_flux

        for band_idx, ordered_band in enumerate(unique_bands):
            if ordered_band == ref_band:
                continue
            extra_sigma_arr[lightcurve.bands == ordered_band] *= cube[7 * band_idx + 6]

        sigma_sq = lightcurve.flux_errors**2 + extra_sigma_arr**2
        logL = np.sum(
            np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
            - 0.5 * (f_model - lightcurve.fluxes) ** 2 / sigma_sq
        )
        return logL

    sampler = NestedSampler(
        create_logL, create_prior, n_params, sample="rwalk", bound="single", nlive=NLIVE, rstate=rstate
    )
    sampler.run_nested(maxiter=MAX_ITER, dlogz=DLOGZ, print_progress=False)
    res = sampler.results

    red_chisq = res.logl / len(lightcurve.times)  # pylint: disable=no-member
    samples = res.samples  # pylint: disable=no-member
    eq_wt_samples = res.samples_equal(rstate=rstate)

    orig_idxs = np.array([np.argmin(np.sum((e - samples) ** 2, axis=1)) for e in eq_wt_samples])
    eq_wt_red_chisq = red_chisq[orig_idxs]

    eq_wt_samples = np.append(eq_wt_samples, eq_wt_red_chisq[np.newaxis, :].T, 1)

    return PosteriorSamples(
        eq_wt_samples, name=lightcurve.name, sampling_method="dynesty", sn_class=lightcurve.sn_class
    )
