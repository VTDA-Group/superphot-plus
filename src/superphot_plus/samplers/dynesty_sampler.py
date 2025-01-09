"""MCMC sampling using dynesty."""

from typing import List, Optional
from functools import partial

import numpy as np
from dynesty import DynamicNestedSampler, NestedSampler
from snapi.analysis import SamplerResult, SamplerPrior
import pandas as pd

from superphot_plus.constants import DLOGZ, MAX_ITER, NLIVE
from superphot_plus.utils import flux_model, params_valid
from superphot_plus.samplers.superphot_sampler import SuperphotSampler


class DynestySampler(SuperphotSampler):
    """ "MCMC sampling using dynesty."""

    def __init__(
            self,
            priors: SamplerPrior,
            random_state: Optional[int]=None,
            max_iter: int=MAX_ITER,
            dlogz: float=DLOGZ,
            bound: str='single',
            sample_strategy: str='rwalk',
            nlive: int=NLIVE,
            dynamic: bool=False,
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
        dynamic: bool, optional
            Whether to use dynamic sampling. Defaults to false.
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
        self._verbose = verbose
        self._sampler_name = 'superphot_dynesty'
        self._prior_func = partial(self._priors.sample, use_numpyro=False)
        self._param_map = None
        self._dynamic = dynamic

        if dynamic:
            self._nested_sampler = DynamicNestedSampler(
                self._logL, self._prior_func, (self._nparams + 3) * len(self._unique_bands),
                sample=sample_strategy, bound=bound, nlive=nlive,
                rstate=self._rng, #walks=50,
            )
        else:
            self._nested_sampler = NestedSampler(
                self._logL, self._prior_func, (self._nparams + 3) * len(self._unique_bands),
                sample=sample_strategy, bound=bound, nlive=nlive,
                rstate=self._rng, #walks=50
            )

    def _logL(self, cube):
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
        if self._param_map is None:
            return -1.0 # placeholder
        
        new_cube = self._reformat_cube(cube)

        if not params_valid(new_cube):
            return -np.inf
        
        f_model = flux_model(new_cube, self._t, self._X[:,1])
        extra_sigma_arr = new_cube[-1]
        sigma_sq = self._err**2 + extra_sigma_arr**2

        logL = np.sum(
            np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
            - 0.5 * (f_model - self._y) ** 2 / sigma_sq
        )
        return logL
    
    def reset(self):
        """Reset the nested sampler."""
        self._nested_sampler.loglikelihood.pool = None # post-pickling fix
        self._nested_sampler.reset()


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
        self._t = self._X[:,0].astype(np.float32)
        self._err = self._X[:,2].astype(np.float32)
        
        # map time steps to param values
        self._param_map = np.zeros((self._nparams+3, len(self._X)), dtype=int)
        for i, param in enumerate(self._base_params):
            for b in self._unique_bands:
                b_idxs = self._X[:,1] == b
                self._param_map[i,b_idxs] = np.where(self._params == f'{param}_{b}')[0][0]
        
        self._param_map = np.array(self._param_map)

        # Require data in all bands
        for band in self._unique_bands:
            if band not in self._X[:, 1]:
                return None
    
        self.reset()

        if self._dynamic:
            self._nested_sampler.run_nested(
                maxiter_init=self._max_iter,
                n_effective=10_000, dlogz_init=self._dlogz,
                print_progress=self._verbose
            )
        else:
            self._nested_sampler.run_nested(
                maxiter=self._max_iter,
                dlogz=self._dlogz,
                print_progress=self._verbose
            )
        res = self._nested_sampler.results
        samples_equal = res.samples_equal(rstate=self._rng)

        samples_df = pd.DataFrame(
            samples_equal,
            columns=self._params
        )
        self._is_fitted = True
        self.result = SamplerResult(samples_df, sampler_name=self._sampler_name)
        self.result.score = self.score(self._X, self._y)
