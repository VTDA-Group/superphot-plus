from typing import Optional

import numpy as np

from snapi.analysis import Sampler, SamplerPrior
from superphot_plus.utils import flux_model

class SuperphotSampler(Sampler):
    """Subclass of SNAPI Sampler for Superphot+."""
    def __init__(
        self,
        priors: SamplerPrior,
        *args,
        **kwargs
    ):
        super().__init__()
        self._nparams = 4 # effective DOF using first piecewise part
        self._priors = priors
        self._params = self._priors.dataframe['param'].to_numpy()
        self._unique_bands = []
        for c in self._params:
            if c[0] == 'A':
                self._unique_bands.append(c[2:])
        self._base_params = []
        for c in self._params:
            if self._unique_bands[0] in c:
                self._base_params.append(c.replace("_" +self._unique_bands[0], ""))
        self.result = None

        print(self._params, self._base_params)

    def fit(self, X, y, event_indices=None):
        """Remove elements where filter not included in priors.
        """
        mask = np.isin(X[:,1], self._unique_bands)

        if event_indices is not None:
            # Prepare new index ranges
            cumsum_mask = np.cumsum(mask)

            # Prepare new index ranges
            new_index_ranges = []

            for original_start, original_end in event_indices:
                # Adjust the start and end indices based on the cumulative mask
                num_retained_before_start = cumsum_mask[original_start - 1] if original_start > 0 else 0
                num_retained_before_end = cumsum_mask[original_end - 1]

                # Append the updated range directly
                new_index_ranges.append((num_retained_before_start, num_retained_before_end))

            super().fit(X[mask], y[mask], new_index_ranges)

        else:
            super().fit(X[mask], y[mask])

    def _reformat_cube(self, cube):
        """Reformat cube based on self._param_map"""
        return cube[self._param_map]
            
    def _eff_variance(self, X):
        """Calculates the effective variance of the model."""
        fit_param_numpy = self.result.fit_parameters[self._params].to_numpy().T # each entry is a parameter
        extra_sigma_arr = self._reformat_cube(fit_param_numpy)[-1] # (num_times, num_fits)
        return X[:,2:3].T.astype(np.float32)**2 + (extra_sigma_arr.T)**2
    
    def flux_model(self, cube, t, b):
        return flux_model(cube, t, b)

    def predict(self, X, num_fits=None):
        """Predicts the flux of a light curve using the model."""
        mask = np.isin(X[:,1], self._unique_bands)
        if np.all(~mask):
            return np.array([]), np.array([])
        _, val_x = super().predict(X[mask])
        self._param_map = np.zeros((self._nparams+3, len(val_x)), dtype=int)
        
        for i, param in enumerate(self._base_params):
            for b in self._unique_bands:
                b_idxs = val_x[:,1] == b
                self._param_map[i,b_idxs] = np.where(self._params == f'{param}_{b}')[0][0]
                
        fit_param_numpy = self.result.fit_parameters[self._params].to_numpy().T # each entry is a parameter
        cube = self._reformat_cube(fit_param_numpy) # (num_params, num_times, num_fits)
        
        if num_fits:
            return self.flux_model(
                cube[:,:,-1*num_fits:],
                val_x[:, 0].astype(np.float32), val_x[:, 1]
            ), val_x

        return self.flux_model(
            cube,
            val_x[:, 0].astype(np.float32), val_x[:, 1]
        ), val_x