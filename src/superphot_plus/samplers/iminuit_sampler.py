"""Gradient slope fitting using iminuit."""
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import truncnorm
from snapi import SamplerResult
from sklearn.utils import check_random_state

from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.utils import (
    flux_model, villar_fit_constraint
)
from superphot_plus.samplers.superphot_sampler import SuperphotSampler

class IminuitSampler(SuperphotSampler):
    """Negative log-likelihood optimization with iminuit's migrad."""

    def __init__(
            self, priors: MultibandPriors,
            random_state: int = None,
        ):
        prior_clip_a, prior_clip_b, self._prior_mean, self._prior_std = priors.to_numpy().T.copy()
        random_state = check_random_state(random_state)
        self._rng = np.random.default_rng(random_state)

        # Precompute the vectors of trunc_gauss a and b values.
        self._tg_a = (prior_clip_a - self._prior_mean) / self._prior_std
        self._tg_b = (prior_clip_b - self._prior_mean) / self._prior_std

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
            probs = truncnorm.pdf(cube, self._tg_a, self._tg_b, loc=self._prior_mean, scale=self._prior_std)
            # replace zeros with small number to avoid log(0)
            probs[probs < 1e-300] = 1e-300
            return np.sum(np.log(probs))
        
        self._prior_func = ln_prior

        self._param_names = self._create_param_names()
        # It looks like iminuit has a bug which requires this extra mock parameter
        self._parameters = {'__mock': None}
        # Here we assign boundaries to the parameters
        self._parameters.update({name: (a, b) for name, a, b in zip(self._param_names, prior_clip_a, prior_clip_b)})

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
        # Require data in all bands
        for band in self._unique_bands:
            if band not in self._X[:, 1]:
                return None

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
            beta = cube[7*self._ref_band_idx + 1]
            gamma = 10**cube[7*self._ref_band_idx + 2]
            tau_rise, tau_fall, extra_sigma = 10**cube[7*self._ref_band_idx + 4:7*self._ref_band_idx + 7]
            
            logL += -1000. * villar_fit_constraint([beta, gamma, tau_rise, tau_fall])

            f_model = flux_model(cube, self._X[:,0], self._X[:,1], self._unique_bands, self._ref_band)
            extra_sigma_arr = np.ones(self._X.shape[0]) * extra_sigma

            for band_idx, ordered_band in enumerate(self._unique_bands):
                if ordered_band == self._ref_band:
                    continue
                beta_b = beta * 10**cube[7 * band_idx + 1]
                gamma_b = gamma * 10**cube[7 * band_idx + 2]
                tau_rise_b = tau_rise * 10**cube[7 * band_idx + 4]
                tau_fall_b = tau_fall * 10**cube[7 * band_idx + 5]
                extra_sigma_b = extra_sigma * 10**cube[7 * band_idx + 6]
                
                logL += villar_fit_constraint([beta_b, gamma_b, tau_rise_b, tau_fall_b])
                logL += -1000. * np.maximum(extra_sigma_b - 10**(-0.8), 0.)
                
                extra_sigma_arr[self._X[:,1] == ordered_band] = extra_sigma_b

            sigma_sq = self._X[:,2]**2 + extra_sigma_arr**2
            logL += np.sum(
                np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
                - 0.5 * (f_model - self._y) ** 2 / sigma_sq
            )
            return logL

        def ln_l(_, *cube):
            cube = np.array(list(cube))
            return np.array(ln_like(cube) + self._prior_func(cube))
        
        ln_l._parameters = self._parameters  # pylint: disable=protected-access

        # We have no data to pass, because the lightcurve is already in the ln_l function
        cost = UnbinnedNLL([0], ln_l, log=True)
        minuit = Minuit(cost, **dict(zip(self._param_names, self._prior_mean)))
        minuit.migrad()

        if minuit.valid:
            sample_mean = np.asarray(minuit.values)
            # Sample from multi-variate Gaussian distribution
            samples = self._rng.multivariate_normal(sample_mean, minuit.covariance, size=100)
        else:
            sample_mean = self._prior_mean
            samples = truncnorm.rvs(
                self._tg_a,
                self._tg_b,
                loc=self._prior_mean,
                scale=self._prior_std,
                random_state=self._rng,
                size=(100, (self._nparams + 1) * len(self._unique_bands)),
            )

        samples_df = pd.DataFrame(samples, columns=self._param_names)

        self.result = SamplerResult(
            samples_df, sampler_name='superphot_iminuit'
        )
        self.result.score = self.score(self._X, self._y)