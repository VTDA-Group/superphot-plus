"""MCMC sampling using the light-curve package."""

import light_curve as licu
import numpy as np
from numpy.typing import NDArray
from snapi import SamplerResult
import pandas as pd

from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.samplers.superphot_sampler import SuperphotSampler


__all__ = ["LiCuSampler"]

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

class LiCuSampler(SuperphotSampler):
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

    def __init__(
            self,
            priors: MultibandPriors,
            **licu_kwargs
        ):
        super().__init__(priors)
        kwargs = {'algorithm': 'ceres', 'ceres_niter': 10_000}
        kwargs.update(licu_kwargs)
        self.licu_kwargs = kwargs
        self._priors_clip_a, self._priors_clip_b, self._priors_mean, _ = (
            self.transform_priors_to_physical_values(a)
            for a in priors.to_numpy().T
        )
        self._orig_prior_mean = priors.to_numpy().T[2]

    def fit(
            self, X: NDArray[np.object_], # pylint: disable=invalid-name
            y: NDArray[np.float32],
        ) -> None: 
        """Fit the data.

        Parameters
        ----------
        X : np.ndarray
            The X data to fit. If 1d, will be reshaped to 2d.
            First column = times, second column = bands, third column = errors.
        y : np.ndarray
            The y data to fit.
        """
        super().fit(X, y)
        

        cube = []
        for band, clip_a, clip_b, mean in zip(
                self._ordered_bands,
                self._priors_clip_a.reshape(-1, 7),
                self._priors_clip_b.reshape(-1, 7),
                self._priors_mean.reshape(-1, 7)
        ):
            X_red = self._X[self._X[:, 1] == band]
            y_red = self._y[self._X[:, 1] == band]
            cube.extend(
                self.fit_single_band(X_red, y_red, clip_a, clip_b, mean)
            )

        cube = np.asarray(cube)
        # Fix for out of the limits parameters and for NaNs
        cube = np.clip(cube, self._priors_clip_a, self._priors_clip_b)
        
        # Scale parameters to the reference band
        non_ref_idx = np.delete(np.arange(len(self._ordered_bands)), self._ref_band_idx)
        param_matrix = cube.reshape(-1, 7)
        param_matrix[non_ref_idx, [0,1,2,4,5,6]] /= param_matrix[self._ref_band_idx, [0,1,2,4,5,6]]
        
        # revert back to log space for saving
        param_matrix[self._ref_band_idx][[0, 2, 4, 5, 6]] = np.log10(param_matrix[self._ref_band_idx][[0, 2, 4, 5, 6]])
        param_matrix[non_ref_idx, [0, 1, 2, 4, 5, 6]] = np.log10(param_matrix[non_ref_idx, [0, 1, 2, 4, 5, 6]])
        param_matrix[non_ref_idx, 3] -= param_matrix[self._ref_band_idx, 3]

        cube = param_matrix.flatten()
        cube = np.where(np.isnan(cube), self._orig_prior_mean, cube)
        samples_df = pd.DataFrame(cube.reshape(-1, 7), columns=self._create_param_names())

        self.result = SamplerResult(samples_df, sampler_name="superphot_licu")
        self.result.score = self.score(self._X, self._y)


    def fit_single_band(self, X, y, prior_clip_a, prior_clip_b, prior_mean):
        """Fit a single band light curve with light_curve.VillarFit

        Parameters
        ----------
        X : np.ndarray
            First column = times, second column = bands, third column = errors.
        y : np.ndarray
            The flux data to fit.
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
            *self.licu_kwargs
        )

        # We have nothing better to do than just use the mean of the prior
        extra_sigma = prior_mean[6]
        try:
            *features, _ = villar_fit(X[:,0], y, X[:,2])
        except ValueError:
            features = initial_guess

        cube = transform_from_licu(*features)
        # Use mean of prior for extra_sigma
        cube[6] = extra_sigma
        return cube

    def transform_priors_to_physical_values(self, cube):
        """Some priors are for log-params, this transforms them back to linear"""
        cube = cube.reshape(-1, 7)
        output = cube.copy()

        # Scale log-priors to linear values
        output[self._ref_band_idx, [0, 2, 4, 5, 6]] = 10 ** output[: [0, 2, 4, 5, 6]]

        # Non-reference band priors must be scaled by the reference band prior
        non_ref_idx = np.delete(np.arange(output.shape[0]), self._ref_band_idx)
        output[non_ref_idx, [0,1,2,4,5,6]] = 10 ** output[non_ref_idx, [0,1,2,4,5,6]]
        output[non_ref_idx, [0,1,2,4,5,6]] *= output[self._ref_band_idx, [0,1,2,4,5,6]]

        # Reference time should not be scaled and should be set separately
        output[non_ref_idx, 3] += output[self._ref_band_idx, 3]

        return output.reshape(-1)
