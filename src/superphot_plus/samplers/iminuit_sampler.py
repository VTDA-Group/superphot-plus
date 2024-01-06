"""Gradient slope fitting using iminuit."""

from typing import List

import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import truncnorm

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.samplers.sampler import Sampler
from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import (
    calculate_chi_squareds, flux_model, villar_fit_constraint
)


class IminuitSampler(Sampler):
    """Negative log-likelihood optimization with iminuit's migrad."""

    def __init__(self):
        pass

    def run_single_curve(
        self, lightcurve: Lightcurve, priors: MultibandPriors, rstate=None, **kwargs
    ) -> PosteriorSamples:
        """Perform model fitting using iminuit on a single light curve.

        This function runs a gradient slope fitting algorithm with iminuit package.
        It returns 100 parameter cubes sampled from a multivariate Gaussian distribution
        centered at the best-fit parameters and with a covariance matrix given by the
        iminuit.

        Parameters
        ----------
        lightcurve : Lightcurve object
            The light curve of interest.
        priors : MultibandPriors
            Prior distribution.
        rstate : int, optional

        Returns
        -------
        samples: PosteriorSamples
            Return the samples or None if the fitting is
            skipped or encounters an error.
        """
        return run_fit(lightcurve, priors=priors, rstate=rstate)

    def run_multi_curve(self, lightcurves, priors, **kwargs) -> List[PosteriorSamples]:
        """Not yet implemented."""
        raise NotImplementedError


def run_fit(lightcurve, priors=Survey.ZTF().priors, rstate=None):
    """Runs iminuit fit for a single light curve

    Parameters
    ----------
    lightcurve : Lightcurve object
        The lightcurve of interest
    priors : str, optional
        Prior information. Defaults to ZTF.
    rstate : int, optional

    Returns
    -------
    PosteriorSamples or None
        Equally weighted posteriors, or None if the data is invalid.
    """
    prior_clip_a, prior_clip_b, prior_mean, prior_std = priors.to_numpy().T.copy()
    ref_band = priors.reference_band

    n_params = prior_clip_a.shape[0]
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
    #prior_clip_a[start_idx + 3] += np.amin(lightcurve.times)
    #prior_clip_b[start_idx + 3] += np.amax(lightcurve.times)
    prior_clip_a[start_idx + 3] += max_flux_loc
    prior_clip_b[start_idx + 3] += max_flux_loc
    prior_mean[start_idx + 3] += max_flux_loc
    #prior_std[start_idx + 3] = 20.0

    # Precompute the vectors of trunc_gauss a and b values.
    tg_a = (prior_clip_a - prior_mean) / prior_std
    tg_b = (prior_clip_b - prior_mean) / prior_std

    """
    idx_to_normalize = np.concatenate(
        [np.array([0, 2, 4, 5, 6]) + 7 * band_idx for band_idx in range(len(unique_bands))]
    )
    """

    def ln_prior(cube):
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
        probs = truncnorm.pdf(cube, tg_a, tg_b, loc=prior_mean, scale=prior_std)
        # replace zeros with small number to avoid log(0)
        probs[probs < 1e-300] = 1e-300
        return np.sum(np.log(probs))

    def ln_like(cube):
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
        logL = 0
        # Re-normalize the cube
        cube = cube.copy()
        beta = cube[7*ref_band_idx + 1]
        gamma = 10**cube[7*ref_band_idx + 2]
        tau_rise, tau_fall, extra_sigma = 10**cube[7*ref_band_idx + 4:7*ref_band_idx + 7]
        
        logL += -1000. * villar_fit_constraint([beta, gamma, tau_rise, tau_fall])
        #cube[idx_to_normalize] = 10 ** cube[idx_to_normalize]

        f_model = flux_model(cube, lightcurve.times, lightcurve.bands, max_flux, unique_bands, ref_band)
        extra_sigma_arr = np.ones(len(lightcurve.times)) * extra_sigma * max_flux

        for band_idx, ordered_band in enumerate(unique_bands):
            if ordered_band == ref_band:
                continue
            beta_b = beta * 10**cube[7 * band_idx + 1]
            gamma_b = gamma * 10**cube[7 * band_idx + 2]
            tau_rise_b = tau_rise * 10**cube[7 * band_idx + 4]
            tau_fall_b = tau_fall * 10**cube[7 * band_idx + 5]
            extra_sigma_b = extra_sigma * 10**cube[7 * band_idx + 6]
            
            logL += villar_fit_constraint([beta_b, gamma_b, tau_rise_b, tau_fall_b])
            logL += -1000. * np.maximum(extra_sigma_b - 10**(-0.8), 0.)
            
            extra_sigma_arr[lightcurve.bands == ordered_band] = extra_sigma_b

        sigma_sq = lightcurve.flux_errors**2 + extra_sigma_arr**2
        logL += np.sum(
            np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
            - 0.5 * (f_model - lightcurve.fluxes) ** 2 / sigma_sq
        )
        return logL

    def ln_l(_, *cube):
        cube = np.array(list(cube))
        return np.array(ln_like(cube) + ln_prior(cube))

    names = [f'cube_{i:0{int(np.ceil(np.log10(n_params)))}d}' for i in range(n_params)]
    # It looks like iminuit has a bug which requires this extra mock parameter
    parameters = {'__mock': None}
    # Here we assign boundaries to the parameters
    parameters.update({name: (a, b) for name, a, b in zip(names, prior_clip_a, prior_clip_b)})
    ln_l._parameters = parameters  # pylint: disable=protected-access

    # We have no data to pass, because the lightcurve is already in the ln_l function
    cost = UnbinnedNLL([], ln_l, log=True)
    minuit = Minuit(cost, **dict(zip(names, prior_mean)))
    minuit.migrad()

    rng = np.random.default_rng(rstate)
    if minuit.valid:
        sample_mean = np.asarray(minuit.values)
        # Sample from multi-variate Gaussian distribution
        samples = rng.multivariate_normal(sample_mean, minuit.covariance, size=100)
    else:
        sample_mean = prior_mean
        samples = truncnorm.rvs(
            tg_a,
            tg_b,
            loc=prior_mean,
            scale=prior_std,
            random_state=rng,
            size=(100, n_params),
        )

    """
    sample_mean[idx_to_normalize] = 10.0 ** sample_mean[idx_to_normalize]
    sample_mean[7 * ref_band_idx] = max_flux * sample_mean[7 * ref_band_idx]
    samples[:, idx_to_normalize] = 10.0 ** samples[:, idx_to_normalize]
    samples[:, 7 * ref_band_idx] = max_flux * samples[:, 7 * ref_band_idx]
    """
    red_chisq_mean = calculate_chi_squareds(
        sample_mean.reshape(1, -1),
        lightcurve.times,
        lightcurve.fluxes,
        lightcurve.flux_errors,
        lightcurve.bands,
        max_flux,
        ordered_bands=priors.ordered_bands,
        ref_band=priors.reference_band,
    )
    posterior_cube_mean = np.concatenate([sample_mean, red_chisq_mean])
    red_chisq = calculate_chi_squareds(
        samples,
        lightcurve.times,
        lightcurve.fluxes,
        lightcurve.flux_errors,
        lightcurve.bands,
        max_flux,
        ordered_bands=priors.ordered_bands,
        ref_band=priors.reference_band,
    )
    posterior_cubes = np.hstack((samples, red_chisq[:, np.newaxis]))

    return PosteriorSamples(
        posterior_cubes, sample_mean=posterior_cube_mean, name=lightcurve.name, sampling_method="iminuit",
        sn_class=lightcurve.sn_class
    )
