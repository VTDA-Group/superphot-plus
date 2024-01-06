"""MCMC sampling using the light-curve package."""

from typing import List

import light_curve as licu
import numpy as np

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.samplers.sampler import Sampler
from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.utils import calculate_chi_squareds

__all__ = ["LiCuSampler"]


class LiCuSampler(Sampler):
    """Fit light curves using the light-curve package's VillarFit

    Parameters
    ----------
    **licu_kwargs : dict
        Keyword arguments to pass to the light-curve package's VillarFit,
        most notably 'algorithm', which can be 'ceres', 'lmsder', 'mcmc',
        'mcmc-ceres' and 'mcmc-lmsder'. Default is 'ceres'. Please also
        consider setting 'mcmc_niter' which is the number of MCMC iterations
        defaults to 128.
        Boundaries and initial guesses are set by this code, so you don't
        need to set them yourself.
    """

    def __init__(self, **licu_kwargs):
        kwargs = {'algorithm': 'ceres', 'ceres_niter': 10_000}
        kwargs.update(licu_kwargs)
        self.licu_kwargs = kwargs

    def run_single_curve(
        self, lightcurve: Lightcurve, priors: MultibandPriors, **kwargs
    ) -> PosteriorSamples:
        """Perform model fitting using light-curve on a single light curve.


        Parameters
        ----------
        lightcurve : Lightcurve object
            The light curve of interest.
        priors : MultibandPriors object
            The priors to use for fitting.
        Returns
        -------
        samples: PosteriorSamples
            Return the MCMC samples or None if the fitting is
            skipped or encounters an error.
        """
        max_flux, max_flux_loc = lightcurve.find_max_flux(band=priors.reference_band)
        reference_idx = priors.ref_band_index
        priors_clip_a, priors_clip_b, priors_mean, _priors_std = (
            transform_priors_to_physical_values(a, max_flux, max_flux_loc, reference_idx=reference_idx)
            for a in priors.to_numpy().T
        )

        cube = []
        for band, clip_a, clip_b, mean in zip(
                priors.ordered_bands,
                priors_clip_a.reshape(-1, 7),
                priors_clip_b.reshape(-1, 7),
                priors_mean.reshape(-1, 7)
        ):
            lc = lightcurve.filter_by_band([band,], in_place=False)
            cube.extend(fit_single_band(lc, clip_a, clip_b, mean, **self.licu_kwargs))
        cube = np.asarray(cube)
        # Fix for out of the limits parameters and for NaNs
        cube = np.clip(cube, priors_clip_a, priors_clip_b)
        
        # Scale parameters to the reference band
        non_ref_idx = np.delete(np.arange(len(priors.ordered_bands)), reference_idx)
        param_matrix = cube.reshape(-1, 7)
        param_matrix[non_ref_idx, [0,1,2,4,5,6]] /= param_matrix[reference_idx, [0,1,2,4,5,6]]
        
        # revert back to log space for saving
        param_matrix[reference_idx][[0,6]] /= max_flux
        param_matrix[reference_idx][[0, 2, 4, 5, 6]] = np.log10(param_matrix[reference_idx][[0, 2, 4, 5, 6]])
        param_matrix[non_ref_idx, [0, 1, 2, 4, 5, 6]] = np.log10(param_matrix[non_ref_idx, [0, 1, 2, 4, 5, 6]])
        param_matrix[non_ref_idx, 3] -= param_matrix[reference_idx, 3]
        param_matrix[:, 3] -= max_flux_loc

        cube = param_matrix.flatten()
        cube = np.where(np.isnan(cube), priors.to_numpy().T[2], cube)

        red_chisq = calculate_chi_squareds(
            cube.reshape(1, -1),
            lightcurve.times,
            lightcurve.fluxes,
            lightcurve.flux_errors,
            lightcurve.bands,
            max_flux,
            ordered_bands=priors.ordered_bands,
            ref_band=priors.reference_band,
        )
        sample = np.concatenate([cube, red_chisq])

        samples = sample.reshape(1, -1)

        return PosteriorSamples(
            samples,
            name=lightcurve.name,
            sampling_method="light-curve-package",
            sn_class=lightcurve.sn_class,
        )

    def run_multi_curve(self, lightcurves, priors, **kwargs) -> List[PosteriorSamples]:
        """Not yet implemented."""
        raise NotImplementedError


def fit_single_band(lc, prior_clip_a, prior_clip_b, prior_mean, **licu_kwargs):
    """Fit a single band light curve with light_curve.VillarFit

    Parameters
    ----------
    lc : Lightcurve
        The light curve with a single band
    prior_clip_a : array-like
        The lower bound of the prior
    prior_clip_b : array-like
        The upper bound of the prior
    prior_mean : array-like
        The mean of the prior
    licu_kwargs : dict
        Keyword arguments to pass to the light-curve package's VillarFit
    """
    left_bound = transform_to_licu(*prior_clip_a)
    # change baseline to be not exactly zero
    left_bound[1] = -1e-3 * prior_clip_a[0]

    right_bound = transform_to_licu(*prior_clip_b)
    # change baseline to be not exactly zero
    right_bound[1] = 1e-3 * prior_clip_a[0]
    # relative plateau amplitude must be less than 1
    right_bound[5] = np.min([right_bound[5], 0.99])

    initial_guess = transform_to_licu(*prior_mean)

    villar_fit = licu.VillarFit(  # pylint: disable=no-member
        init=initial_guess,
        bounds=list(zip(left_bound, right_bound)),
        **licu_kwargs
    )

    # We have nothing better to do than just use the mean of the prior
    extra_sigma = prior_mean[6]

    # light-curve package parameters
    # Reduced Chi^2 definition is different from the one in superphot+
    try:
        *features, _reduced_chi2 = villar_fit(lc.times, lc.fluxes, lc.flux_errors)
    except ValueError:
        features = initial_guess

    cube = transform_from_licu(*features)
    # Use mean of prior for extra_sigma
    cube[6] = extra_sigma

    return cube


def transform_priors_to_physical_values(cube, max_flux, max_flux_loc, *, reference_idx):
    """Some priors are for log-params, this transforms them back to linear"""
    cube = cube.reshape(-1, 7)
    output = cube.copy()

    # Scale log-priors to linear values
    output[reference_idx, [0, 2, 4, 5, 6]] = 10 ** output[reference_idx, [0, 2, 4, 5, 6]]

    # amplitude and extra_sigma must be scaled by the maximum flux
    output[reference_idx, [0, 6]] *= max_flux

    # Non-reference band priors must be scaled by the reference band prior
    non_ref_idx = np.delete(np.arange(output.shape[0]), reference_idx)
    output[non_ref_idx, [0,1,2,4,5,6]] = 10**output[non_ref_idx, [0,1,2,4,5,6]]
    output[non_ref_idx, [0,1,2,4,5,6]] *= output[reference_idx, [0,1,2,4,5,6]]

    # Reference time should not be scaled and should be set separately
    output[non_ref_idx, 3] += output[reference_idx, 3]
    output[:, 3] += max_flux_loc

    return output.reshape(-1)


def transform_to_licu(amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma):
    """Transforms superphot+ parameters to light-curve package parameters"""
    del extra_sigma  # no extra_sigma in light-curve package
    amplitude = amp
    baseline = 0.0  # no baseline in superphot+
    reference_time = t_0
    rise_time = tau_rise
    fall_time = tau_fall
    plateau_rel_amplitude = beta * gamma
    plateau_duration = gamma
    return np.array([
        amplitude,
        baseline,
        reference_time,
        rise_time,
        fall_time,
        plateau_rel_amplitude,
        plateau_duration,
    ])


def transform_from_licu(
        amplitude,
        baseline,
        reference_time,
        rise_time,
        fall_time,
        plateau_rel_amplitude,
        plateau_duration,
):
    """Transforms light-curve package parameters to superphot+ parameters"""
    del baseline  # no baseline in superphot+
    amp = amplitude
    beta = plateau_rel_amplitude / plateau_duration
    gamma = plateau_duration
    t_0 = reference_time
    tau_rise = rise_time
    tau_fall = fall_time
    extra_sigma = 0.  # no extra_sigma in light-curve package
    return np.array([amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma])
