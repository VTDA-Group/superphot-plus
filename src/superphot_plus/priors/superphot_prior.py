import pandas as pd
import jax.numpy as jnp
import jax
import numpy as np
import numpyro.distributions as dist
import numpyro
from scipy.stats import truncnorm
from snapi.analysis import SamplerPrior

#jax.config.update("jax_disable_jit", True)
jax.config.update('jax_platform_name', 'cpu')

class SuperphotPrior(SamplerPrior):
    """Stores prior information for sampler. Only supports Gaussianity
    or log-Gaussianity. If log-Gaussianity, parameters are assumed of logged
    distribution."""
    
    def __init__(
        self,
        prior_info: pd.DataFrame
    ):
        """Stores prior information for the Sampler."""
        super().__init__(prior_info)
        for k in ['param', 'mean', 'stddev', 'min', 'max', 'logged', 'relative', 'relative_op']:
            if k not in prior_info:
                raise ValueError(f"column {k} not in prior_info!")
        self._df.loc[:,'logged'] = self._df.loc[:,'logged'].astype(bool)
        self.update()
        
    def update(self) -> None:
        """Rearrange priors so correlated priors are sampled
        in the right order.
        """
        rerun = False
        while rerun:
            rerun = False
            ordered_indices = {p: i for i, p in enumerate(self._df['param'])}
            for r in self._df.loc[~pd.isna(self._df['relative'])]:
                if ordered_indices[r['relative']] > r.index:
                    # move r to back of array
                    self._df.pop(r, inplace=True)
                    self._df.append(r, inplace=True)
                    rerun = True
                    
        self._relative_mask = self._df['relative'].notna().to_numpy()
        self._relative_idxs = []

        # get shuffled relative idxs
        for rel_val in self._df.loc[self._relative_mask, 'relative']:
            # Find the index where the 'param' column matches the 'relative' value
            index = self._df[self._df['param'] == rel_val].index
            self._relative_idxs.append(index[0])  # Storing the first matching index
        
        
        self._df.loc[
            self._relative_mask, 'min'
        ] = self._df.iloc[self._relative_idxs]['min'].to_numpy()
        
        self._df.loc[
            self._relative_mask, 'max'
        ] = self._df.iloc[self._relative_idxs]['max'].to_numpy()
        
                    
        self._tga = ((self._df['min'] - self._df['mean']) / self._df['stddev']).to_numpy()
        self._tgb = ((self._df['max'] - self._df['mean']) / self._df['stddev']).to_numpy()
            
        # faster sample calls
        self._logged = self._df['logged'].to_numpy()
        self._mean = self._df['mean'].to_numpy()
        self._std = self._df['stddev'].to_numpy()
        self._numpyro_sample_arr = jnp.array(
            self._df[['min', 'max', 'mean', 'stddev']].to_numpy().astype(float).T
        )
        
        self._relative_idxs_jax = jnp.array(self._relative_idxs)
        self._logged_jax = jnp.where(self._logged)[0]
        self._static_base_mask = jnp.where(~self._relative_mask)[0]
        self._static_relative_mask = jnp.where(self._relative_mask)[0]
        
        self._params = self._df['param'].to_numpy()
        
    @property
    def dataframe(self):
        """Return all prior info in a dataframe."""
        return self._df.copy()
    
    def _trunc_norm(self, fields):
        """Provides keyword parameters to numpyro's TruncatedNormal, using the fields in PriorFields.

        Parameters
        ----------
        fields : PriorFields
            The (low, high, mean, standard deviation) fields of the truncated normal distribution.

        Returns
        -------
        numpyro.distributions.TruncatedDistribution
            A truncated normal distribution.
        """
        return dist.TruncatedNormal(
            loc=fields[2], scale=fields[3], 
            low=fields[0], high=fields[1],
            validate_args=False
        )
    
    def sample(self, cube, use_numpyro=False):
        """Sample from priors. If numpyro=True, then
        use the numpyro framework.
        """
        if use_numpyro:
            min_vals, max_vals, init_loc, init_scale = self._numpyro_sample_arr[:,self._static_base_mask]
            
            with numpyro.plate("base_params", len(min_vals)):
                base_vals = numpyro.sample(
                    "base_samples",
                    dist.TruncatedNormal(
                        loc=init_loc,
                        scale=init_scale,
                        low=min_vals,
                        high=max_vals
                    )
                )

            # Compute the adjustment only for relative ones
            relative_shifts = base_vals[self._relative_idxs_jax]

            min_vals, max_vals, init_loc, init_scale = self._numpyro_sample_arr[:,self._static_relative_mask]
            adjusted_locs = init_loc + relative_shifts

            # Reapply the constraints to adjusted_locs to make sure they stay within bounds
            adjusted_locs_constrained = jnp.clip(adjusted_locs, min_vals + 1e-6, max_vals - 1e-6)

            with numpyro.plate("relative_params", len(min_vals)):
                # Re-sample using the adjusted means only for relative parameters
                resampled_vals = numpyro.sample(
                    "relative_samples",
                    dist.TruncatedNormal(
                        loc=adjusted_locs_constrained,
                        scale=init_scale,
                        low=min_vals,
                        high=max_vals,
                    )
                )
            
            vals = jnp.concatenate([
                base_vals,
                resampled_vals
            ])

            vals = vals.at[self._logged_jax].set(10**vals[self._logged_jax])
            
        else:
            if cube is None:
                cube = self._rng.uniform(size=len(self._df))
            vals = np.zeros(len(cube))
            
            vals[~self._relative_mask] = truncnorm.ppf(
                cube[~self._relative_mask],
                self._tga[~self._relative_mask],
                self._tgb[~self._relative_mask],
                loc=self._mean[~self._relative_mask],
                scale=self._std[~self._relative_mask],
            )
            
            vals[self._relative_mask] = truncnorm.ppf(
                cube[self._relative_mask],
                self._tga[self._relative_mask],
                self._tgb[self._relative_mask],
                loc=self._mean[self._relative_mask] + vals[self._relative_idxs],
                scale=self._std[self._relative_mask]
            )
            
            # log transformations
            vals[self._logged] = 10**vals[self._logged]
        return vals
    
    
    def jax_guide(self):
        """Guide for numpyro-based samplers."""
        
        min_vals, max_vals, init_loc, init_scale = self._numpyro_sample_arr[:,self._static_base_mask]
            
        with numpyro.plate("base_params", len(min_vals)):
            # Create learnable parameters with initial constraints
            svi_loc_base = numpyro.param(
                "loc_base",
                init_value=init_loc,
                constraint=dist.constraints.interval(min_vals, max_vals)
            )

            svi_scale_base = numpyro.param(
                "scale_base", 
                init_value=init_scale / 10.,
                constraint=dist.constraints.positive
            )
            
            base_vals = numpyro.sample(
                "base_samples",
                dist.TruncatedNormal(
                    loc=svi_loc_base,
                    scale=svi_scale_base,
                    low=min_vals,
                    high=max_vals
                )
            )

        # Compute the adjustment only for relative ones
        relative_shifts = base_vals[self._relative_idxs_jax]
        
        min_vals, max_vals, init_loc, init_scale = self._numpyro_sample_arr[:,self._static_relative_mask]
        adjusted_locs = init_loc[self._static_relative_mask] + relative_shifts
        adjusted_locs_constrained = jnp.clip(adjusted_locs, min_vals + 1e-6, max_vals - 1e-6)
        

        with numpyro.plate("relative_params", len(min_vals)):
            # Re-sample using the adjusted means only for relative parameters
            svi_loc = numpyro.param(
                "loc_relative",
                init_value=adjusted_locs_constrained,
                constraint=dist.constraints.interval(min_vals, max_vals)
            )

            svi_scale = numpyro.param(
                "scale_relative", 
                init_value=init_scale / 10.,
                constraint=dist.constraints.positive
            )
            
            resampled_vals = numpyro.sample(
                "relative_samples",
                dist.TruncatedNormal(
                    loc=svi_loc,
                    scale=svi_scale,
                    low=min_vals,
                    high=max_vals,
                )
            )
            
    
    def transform(self, samples):
        """Transform relative and log-Gaussian samples
        from gaussian-sampled values.
        """
        samples.loc[:,self._params[self._logged]] = 10**samples.loc[:,self._params[self._logged]]
        return samples
    
    def reverse_transform(self, samples: pd.DataFrame):
        """From relative, log-Gaussian samples, return original
        uncorrelated Gaussian samples.
        """
        samples_shuffled = samples.loc[:,self._params]
        samples_shuffled.iloc[:,self._logged] = np.log10(samples_shuffled.iloc[:,self._logged])
        
        # un-relative samples
        samples_shuffled.iloc[:,self._relative_mask] = samples_shuffled.iloc[:,self._relative_mask].sub(
            samples_shuffled.iloc[:,self._relative_idxs].to_numpy()
        )
        return samples_shuffled