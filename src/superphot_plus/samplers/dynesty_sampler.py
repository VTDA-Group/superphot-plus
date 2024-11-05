"""MCMC sampling using dynesty."""

from typing import List, Optional

import numpy as np
from dynesty import NestedSampler
from scipy.stats import truncnorm
from snapi import SamplerResult
import pandas as pd

from superphot_plus.constants import DLOGZ, MAX_ITER, NLIVE
from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.utils import flux_model, params_valid
from superphot_plus.samplers.superphot_sampler import SuperphotSampler



class DynestySampler(SuperphotSampler):
    """ "MCMC sampling using dynesty."""

    def __init__(
            self,
            priors: MultibandPriors,
            random_state: Optional[int]=None,
            max_iter: int=MAX_ITER,
            dlogz: float=DLOGZ,
            bound: str='single',
            sample_strategy: str='rwalk',
            nlive: int=NLIVE,
            verbose: bool=False
        ):
        """Initialize the DynestySampler object.

        Parameters
        ----------
        priors : MultibandPriors
            The priors for the fit.
        random_state : int, optional
            The random state for the fit.
        max_iter : int, optional
            The maximum number of iterations.
        dlogz : float, optional
            The dlogz value.
        bound : str, optional
            The bound type.
        sample_strategy : str, optional
            The sample strategy.
        nlive : int, optional
            The number of live points.
        verbose : bool, optional
            Whether to print progress.
        """
        super().__init__(priors)

        # set all parameters
        self._rng = np.random.default_rng(random_state)
        if max_iter < 1:
            raise ValueError("max_iter must be greater than 0.")
        if dlogz <= 0:
            raise ValueError("dlogz must be greater than 0.")
        self._max_iter = max_iter
        self._dlogz = dlogz
        self._bound = bound
        self._sample_strategy = sample_strategy
        self._nlive = nlive
        self._verbose = verbose
        self._sampler_name = 'superphot_dynesty'

        # Precompute the vectors of trunc_gauss a and b values.
        tg_a = (self._all_priors[0] - self._all_priors[2]) / self._all_priors[3]
        tg_b = (self._all_priors[1] - self._all_priors[2]) / self._all_priors[3]

        def create_prior(cube):
            """Define the prior function.

            Parameters
            ----------
            cube : np.ndarray
                Array of parameters.

            Returns
            -------
            np.ndarray
                Array of parameters.
            """
            x = truncnorm.ppf(cube, tg_a, tg_b, loc=self._all_priors[2], scale=self._all_priors[3])
            return x
        
        self._prior_func = create_prior

    def fit(self, X, y):
        """Runs dynesty importance nested sampling on a set of light curves; saves set
        of equally weighted posteriors (sets of fit parameters).

        Parameters
        ----------
        X: np.ndarray
            Array of light curve times, bands, and flux errors (in that order).
        y: np.ndarray
            Array of light curve fluxes.

        Returns
        -------
        SamplerResult or None
            Stores info on equally weighted posteriors, or None if the data is invalid.
        """
        super().fit(X, y)
        
        # Require data in all bands
        for band in self._unique_bands:
            if band not in self._X[:, 1]:
                return None

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
            beta = cube[self._start_idx+1]
            gamma = 10**cube[self._start_idx+2]
            tau_rise = 10**cube[self._start_idx+4]
            tau_fall = 10**cube[self._start_idx+5]
            
            if not params_valid(beta, gamma, tau_rise, tau_fall):
                return -np.inf
            
            f_model = flux_model(cube, self._X[:,0].astype(np.float32), self._X[:,1], self._unique_bands, self._ref_band)
            extra_sigma_arr = np.ones(self._X.shape[0]) * 10**cube[7*self._ref_band_idx + 6]

            for band_idx, ordered_band in enumerate(self._unique_bands):
                if ordered_band == self._ref_band:
                    continue
                if cube[7 * band_idx + 6] + cube[7*self._ref_band_idx + 6] > -0.8:
                    return -np.inf
                
                beta_g = beta + cube[7*band_idx+1]
                gamma_g = gamma * 10**cube[7*band_idx+2]
                tau_rise_g = tau_rise * 10**cube[7*band_idx+4]
                tau_fall_g = tau_fall * 10**cube[7*band_idx+5]
                                                    
                if not params_valid(beta_g, gamma_g, tau_rise_g, tau_fall_g):
                    return -np.inf
                
                extra_sigma_arr[self._X[:,1] == ordered_band] *= 10**cube[7 * band_idx + 6]

            sigma_sq = self._X[:,2].astype(np.float32)**2 + extra_sigma_arr**2
            logL = np.sum(
                np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
                - 0.5 * (f_model - self._y) ** 2 / sigma_sq
            )
            return logL

        sampler = NestedSampler(
            create_logL, self._prior_func, (self._nparams + 1) * len(self._unique_bands),
            sample=self._sample_strategy, bound=self._bound, nlive=self._nlive,
            rstate=self._rng
        )
        sampler.run_nested(maxiter=self._max_iter, dlogz=self._dlogz, print_progress=self._verbose)
        res = sampler.results

        samples_df = pd.DataFrame(
            res.samples_equal(rstate=self._rng),
            columns=self._create_param_names()
        )
        self.result = SamplerResult(samples_df, sampler_name=self._sampler_name)
        self._is_fitted = True
        self.result.score = self.score(self._X, self._y)
