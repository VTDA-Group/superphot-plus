import pandas as pd
import jax.numpy as jnp
import jax
from jax import debug
import numpy as np
import numpyro.distributions as dist
import numpyro
from scipy.stats import truncnorm
from snapi.analysis import SamplerPrior

#jax.config.update("jax_disable_jit", True)
jax.config.update('jax_platform_name', 'cpu')
#jax.config.update("jax_debug_nans", True)

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
        
        """
        self._df.loc[
            self._relative_mask, 'min'
        ] = self._df.iloc[self._relative_idxs]['min'].to_numpy()
        
        self._df.loc[
            self._relative_mask, 'max'
        ] = self._df.iloc[self._relative_idxs]['max'].to_numpy()
        """
                    
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
    
    def sample(self, cube, use_numpyro=False, num_events=None):
        """Sample from priors. If numpyro=True, then
        use the numpyro framework.
        """
        
        if use_numpyro:
            min_vals_base, max_vals_base, init_loc_base, init_scale_base = self._numpyro_sample_arr[:,self._static_base_mask]
            min_vals_rel, max_vals_rel, init_loc_rel, init_scale_rel = self._numpyro_sample_arr[:, self._static_relative_mask]

            if num_events:
                # Global priors for base parameters
                global_mu_base = numpyro.sample(
                    "global_mu_base",
                    dist.TruncatedNormal(
                        loc=init_loc_base,
                        scale=init_scale_base,
                        low=min_vals_base,
                        high=max_vals_base,
                    )
                )
                global_sigma_base = numpyro.sample("global_sigma_base", dist.HalfNormal(init_scale_base))

                # Global priors for relative parameters
                global_mu_rel = numpyro.sample(
                    "global_mu_rel",
                    dist.TruncatedNormal(
                        loc=init_loc_rel,
                        scale=init_scale_rel,
                        low=min_vals_rel,
                        high=max_vals_rel
                    )
                )
                global_sigma_rel = numpyro.sample("global_sigma_rel", dist.HalfNormal(init_scale_rel))

                with numpyro.plate("events", num_events, dim=-2):   # Assume self.num_events defines the number of events
                    with numpyro.plate("base_params", len(init_loc_base), dim=-1):
                        # Base values for each event
                        global_mu_base_arr = jnp.tile(global_mu_base, (num_events, 1))
                        base_vals = numpyro.sample(
                            "base_samples",
                            dist.TruncatedNormal(
                                loc=global_mu_base_arr,
                                scale=global_sigma_base,
                                low=min_vals_base,
                                high=max_vals_base
                            )
                        )

                    # Compute relative shifts
                    relative_shifts = base_vals[:,self._relative_idxs_jax]

                    # Adjust locations based on relative shifts
                    adjusted_locs = global_mu_rel + relative_shifts
                    # Constrain adjusted locations within bounds
                    adjusted_locs_constrained = jnp.clip(
                        adjusted_locs, min_vals_base[self._relative_idxs_jax] + 1e-6,
                        max_vals_base[self._relative_idxs_jax] - 1e-6
                    )
                    with numpyro.plate("relative_params", len(min_vals_rel), dim=-1):
                        resampled_vals = numpyro.sample(
                            "relative_samples",
                            dist.TruncatedNormal(
                                loc=adjusted_locs_constrained,
                                scale=global_sigma_rel,
                                low=min_vals_base[self._relative_idxs_jax],
                                high=max_vals_base[self._relative_idxs_jax],
                            )
                        )

                    # Combine base and relative values for each event
                    vals = jnp.concatenate([
                        base_vals,
                        resampled_vals
                    ], axis=1)

                    # Apply transformations to logged values if necessary
                    vals = vals.at[:,self._logged_jax].set(10 ** vals[:,self._logged_jax])


            else:
                base_vals = numpyro.sample(
                    "base_samples",
                    dist.TruncatedNormal(
                        loc=init_loc_base,
                        scale=init_scale_base,
                        low=min_vals_base,
                        high=max_vals_base,
                    )
                )

                # Compute the adjustment only for relative ones
                relative_shifts = base_vals[self._relative_idxs_jax]
                adjusted_locs = init_loc_rel + relative_shifts

                # Reapply the constraints to adjusted_locs to make sure they stay within bounds
                adjusted_locs_constrained = jnp.clip(adjusted_locs, min_vals_rel + 1e-6, max_vals_rel - 1e-6)

                with numpyro.plate("relative_params", len(min_vals_rel)):
                    # Re-sample using the adjusted means only for relative parameters
                    resampled_vals = numpyro.sample(
                        "relative_samples",
                        dist.TruncatedNormal(
                            loc=adjusted_locs_constrained,
                            scale=init_scale_rel,
                            low=min_vals_rel,
                            high=max_vals_rel,
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
    
    
    def jax_guide(self, num_events=None):
        """Guide for numpyro-based samplers."""
        
        min_vals_base, max_vals_base, init_loc_base, init_scale_base = self._numpyro_sample_arr[:,self._static_base_mask]
        min_vals_rel, max_vals_rel, init_loc_rel, init_scale_rel = self._numpyro_sample_arr[:, self._static_relative_mask]

        if num_events:
            global_mu_base_loc = numpyro.param(
                "global_mu_base_loc",
                init_value=init_loc_base,
                constraint=dist.constraints.interval(min_vals_base, max_vals_base)
            )

            global_mu_base_sigma = numpyro.param(
                "global_mu_base_sigma",
                init_value=init_scale_base / 10.,
                constraint=dist.constraints.positive
            )

            # Sample global base parameters in the guide
            numpyro.sample(
                "global_mu_base",
                dist.TruncatedNormal(
                    loc=global_mu_base_loc,
                    scale=global_mu_base_sigma,
                    low=min_vals_base,
                    high=max_vals_base
                )
            )

            global_sigma_base_loc = numpyro.param(
                "global_sigma_base_loc",
                init_value=init_scale_base,
                constraint=dist.constraints.positive
            )

            global_sigma_base_sigma = numpyro.param(
                "global_sigma_base_sigma",
                init_value=init_scale_base / 10.,
                constraint=dist.constraints.positive
            )

            numpyro.sample("global_sigma_base", dist.Normal(global_sigma_base_loc, global_sigma_base_sigma))

            # global relative
            global_mu_rel_loc = numpyro.param(
                "global_mu_rel_loc",
                init_value=init_loc_rel,
                constraint=dist.constraints.interval(min_vals_rel, max_vals_rel)
            )

            global_mu_rel_sigma = numpyro.param(
                "global_mu_rel_sigma",
                init_value=init_scale_rel / 10.,
                constraint=dist.constraints.positive
            )

            # Sample global base parameters in the guide
            numpyro.sample(
                "global_mu_rel",
                dist.TruncatedNormal(
                    loc=global_mu_rel_loc,
                    scale=global_mu_rel_sigma,
                    low=min_vals_rel,
                    high=max_vals_rel
                )
            )

            global_sigma_rel_loc = numpyro.param(
                "global_sigma_rel_loc",
                init_value=init_scale_rel,
                constraint=dist.constraints.positive
            )

            global_sigma_rel_sigma = numpyro.param(
                "global_sigma_rel_sigma",
                init_value=init_scale_rel / 10.,
                constraint=dist.constraints.positive
            )

            numpyro.sample("global_sigma_rel", dist.Normal(global_sigma_rel_loc, global_sigma_rel_sigma))

            """
            debug.print("Global mu base: {}", global_mu_base_loc)
            debug.print("Global sigma base: {}", global_mu_base_sigma)
            debug.print("Global mu rel: {}", global_mu_rel_loc)
            debug.print("Global sigma rel: {}", global_mu_rel_sigma)
            """
 
            with numpyro.plate("events", num_events, dim=-2):
                with numpyro.plate("base_params", len(init_loc_base), dim=-1):

                    init_loc_base_arr = jnp.tile(init_loc_base, (num_events, 1))
                    init_scale_base_arr = jnp.tile(init_scale_base, (num_events, 1))

                    # Define learnable parameters for base values per event
                    svi_loc_base = numpyro.param(
                        "loc_base",
                        init_value=init_loc_base_arr,
                        constraint=dist.constraints.interval(min_vals_base, max_vals_base)
                    )

                    svi_scale_base = numpyro.param(
                        f"scale_base",
                        init_value=init_scale_base_arr / 10.0,
                        constraint=dist.constraints.positive
                    )

                    # Sample base values per event
                    base_vals = numpyro.sample(
                        f"base_samples",
                        dist.TruncatedNormal(
                            loc=svi_loc_base,
                            scale=svi_scale_base,
                            low=min_vals_base,
                            high=max_vals_base,
                        )
                    )

                    #debug.print("{}", base_vals)

                # Compute the shifts for relative parameters
                relative_shifts = svi_loc_base[:,self._relative_idxs_jax]
                adjusted_locs = init_loc_rel + relative_shifts
                adjusted_locs_constrained = jnp.clip(
                    adjusted_locs,
                    min_vals_rel[jnp.newaxis,:] + 1e-6,
                    max_vals_rel[jnp.newaxis,:] - 1e-6
                )

                with numpyro.plate("relative_params", len(init_loc_rel), dim=-1):
                    init_scale_rel_arr = jnp.tile(init_scale_rel, (num_events, 1))

                    svi_loc_relative = numpyro.param(
                        "loc_relative",
                        init_value=adjusted_locs_constrained,
                        constraint=dist.constraints.interval(
                            min_vals_base[self._relative_idxs_jax],
                            max_vals_base[self._relative_idxs_jax]
                        )
                    )

                    svi_scale_relative = numpyro.param(
                        "scale_relative",
                        init_value=init_scale_rel_arr / 10.0,
                        constraint=dist.constraints.positive
                    )

                    # Sample relative values per event
                    relative_vals = numpyro.sample(
                        "relative_samples",
                        dist.TruncatedNormal(
                            loc=svi_loc_relative,
                            scale=svi_scale_relative,
                            low=min_vals_base[self._relative_idxs_jax],
                            high=max_vals_base[self._relative_idxs_jax]
                        )
                    )



        else:
            with numpyro.plate("base_params", len(min_vals_base)):
                # Create learnable parameters with initial constraints
                svi_loc_base = numpyro.param(
                    "loc_base",
                    init_value=init_loc_base,
                    constraint=dist.constraints.interval(min_vals_base, max_vals_base)
                )

                svi_scale_base = numpyro.param(
                    "scale_base", 
                    init_value=init_scale_base / 10.,
                    constraint=dist.constraints.positive
                )
                
                base_vals = numpyro.sample(
                    "base_samples",
                    dist.TruncatedNormal(
                        loc=svi_loc_base,
                        scale=svi_scale_base,
                        low=min_vals_base,
                        high=max_vals_base
                    )
                )

            # Compute the adjustment only for relative ones
            relative_shifts = base_vals[self._relative_idxs_jax]
            
            adjusted_locs = init_loc_rel + relative_shifts
            adjusted_locs_constrained = jnp.clip(adjusted_locs, min_vals_rel + 1e-6, max_vals_rel - 1e-6)
            
            with numpyro.plate("relative_params", len(min_vals_rel)):
                # Re-sample using the adjusted means only for relative parameters
                svi_loc = numpyro.param(
                    "loc_relative",
                    init_value=adjusted_locs_constrained,
                    constraint=dist.constraints.interval(min_vals_rel, max_vals_rel)
                )

                svi_scale = numpyro.param(
                    "scale_relative", 
                    init_value=init_scale_rel / 10.,
                    constraint=dist.constraints.positive
                )
                
                resampled_vals = numpyro.sample(
                    "relative_samples",
                    dist.TruncatedNormal(
                        loc=svi_loc,
                        scale=svi_scale,
                        low=min_vals_rel,
                        high=max_vals_rel,
                    )
                )
            
    
    def transform(self, samples, relative=False):
        """Transform relative and log-Gaussian samples
        from gaussian-sampled values.
        """
        samples_copy = samples.loc[:,self._params]
        if relative:
            samples_copy.iloc[:,self._relative_mask] = samples_copy.iloc[:,self._relative_mask].add(
                samples_copy.iloc[:,self._relative_idxs].to_numpy()
            )
        samples_copy.loc[:,self._params[self._logged]] = 10**samples_copy.loc[:,self._params[self._logged]]
        return samples_copy
    
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