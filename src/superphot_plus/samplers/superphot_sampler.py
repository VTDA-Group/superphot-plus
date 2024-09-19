from typing import Optional

import numpy as np
from snapi import Sampler

from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import flux_model

class SuperphotSampler(Sampler):
    """Subclass of SNAPI Sampler for Superphot+."""
    def __init__(
        self,
        *args,
        priors: Optional[MultibandPriors]=Survey.ZTF().priors,
        **kwargs
    ):
        super().__init__()
        self._nparams = 6
        self._all_priors = priors.to_numpy().T
        self._ref_band = priors.reference_band
        self._unique_bands = priors.ordered_bands
        self._ref_band_idx = np.argmax(self._unique_bands == self._ref_band)
        self._start_idx = 7 * self._ref_band_idx
        self._is_fitted = False
        self.result = None


    def _create_param_names(self):
        """Creates the parameter names."""
        param_names = ['log_A', 'beta', 'log_gamma', 't0', 'log_tau_rise', 'log_tau_fall', 'log_extra_sigma']
        for band in self._unique_bands:
            if band == self._ref_band:
                continue
            param_names += [
                f'log_A_{band}', f'beta_{band}', f'log_gamma_{band}', f't0_{band}',
                f'log_tau_rise_{band}', f'log_tau_fall_{band}', f'log_extra_sigma_{band}'
            ]
        return param_names

    def _eff_variance(self, X):
        """Calculates the effective variance of the model."""
        log_extra_sigma_arr = np.repeat(self.result.fit_parameters['log_extra_sigma'].to_numpy()[:,np.newaxis], X.shape[0], axis=1)
        
        for ordered_band in self._unique_bands:
            if ordered_band == self._ref_band:
                continue
            log_extra_sigma_arr_band = np.repeat(self.result.fit_parameters[f'log_extra_sigma_{ordered_band}'].to_numpy()[:,np.newaxis], X.shape[0], axis=1)
            log_extra_sigma_arr[:,X[:,1] == ordered_band] += log_extra_sigma_arr_band[:,X[:,1] == ordered_band]
        
        return X[:,2:3].T.astype(np.float32)**2 + (10**log_extra_sigma_arr)**2

    def predict(self, X):
        """Predicts the flux of a light curve using the model."""
        super().predict(X)

        return flux_model(
            self.result.samples,
            X[:, 0], X[:, 1],
            self._unique_bands,
            self._ref_band
        )