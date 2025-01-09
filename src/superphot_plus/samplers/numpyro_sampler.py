"""MCMC sampling using numpyro."""
from typing import Optional
from functools import partial

from numpy.typing import NDArray
#from numpyro.distributions import constraints
#from numpyro.distributions.transforms import biject_to
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import (
    _without_rsample_stop_gradient,
    get_importance_trace,
    is_identically_one,
    log_density,
)
from numpyro.util import _validate_model, check_model_guide_match, find_stack_level
import pandas as pd
import jax.numpy as jnp
from jax import random, lax, jit, vmap, config, debug, grad
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.elbo import ELBO
from numpyro.infer.svi import _make_loss_fn, SVIState
from numpyro.handlers import replay, seed, substitute, trace
from snapi.analysis import SamplerPrior, SamplerResult
from sklearn.utils import check_random_state

from superphot_plus.samplers.superphot_sampler import SuperphotSampler
from superphot_plus.utils import villar_fit_constraint

#numpyro.set_host_device_count(1)
#config.update("jax_enable_x64", True)
#config.update("jax_disable_jit", True)
#config.update("jax_debug_nans", True)
#numpyro.enable_x64()

class Debug_Trace_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide.

    This is the most basic implementation of the Evidence Lower Bound, which is the
    fundamental objective in Variational Inference. This implementation has various
    limitations (for example it only supports random variables with reparameterized
    samplers) but can be used as a template to build more sophisticated loss
    objectives.

    For more details, refer to http://pyro.ai/examples/svi_part_i.html.

    **References:**

    1. *Automated Variational Inference in Probabilistic Programming*,
       David Wingate, Theo Weber
    2. *Black Box Variational Inference*,
       Rajesh Ranganath, Sean Gerrish, David M. Blei

    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators.
    :param vectorize_particles: Whether to use `jax.vmap` to compute ELBOs over the
        num_particles-many particles in parallel. If False use `jax.lax.map`.
        Defaults to True. You can also pass a callable to specify a custom vectorization
        strategy, for example `jax.pmap`.
    :param multi_sample_guide: Whether to make an assumption that the guide proposes
        multiple samples.
    """

    def __init__(
        self, num_particles=1, vectorize_particles=True, multi_sample_guide=False
    ):
        self.multi_sample_guide = multi_sample_guide
        super().__init__(
            num_particles=num_particles, vectorize_particles=vectorize_particles
        )

    def loss_with_mutable_state(
            self,
            rng_key,
            param_map,
            model,
            guide,
            *args,
            **kwargs,
        ):
            def single_particle_elbo(rng_key):
                params = param_map.copy()
                model_seed, guide_seed = random.split(rng_key)
                seeded_guide = seed(guide, guide_seed)
                guide_log_density, guide_trace = log_density(
                    seeded_guide, args, kwargs, param_map
                )
                debug.print("Guide log density: {}", guide_log_density)

                seeded_model = seed(model, model_seed)
                replay_model = replay(seeded_model, guide_trace)
                model_log_density, model_trace = log_density(
                    replay_model, args, kwargs, params
                )
                debug.print("Model log density: {}", model_log_density)
                check_model_guide_match(model_trace, guide_trace)
                _validate_model(model_trace, plate_warning="loose")
                
                # log p(z) - log q(z)
                elbo_particle = model_log_density - guide_log_density
                debug.print("ELBO: {}",elbo_particle)

                return elbo_particle, None

            # Return (-elbo) since by convention we do gradient descent on a loss and
            # the ELBO is a lower bound that needs to be maximized.
            elbo, mutable_state = single_particle_elbo(rng_key)
            return {"loss": -elbo, "mutable_state": mutable_state}
            

def lax_helper_function(svi, svi_state, num_iters, *args, **kwargs):
    """Helper function using LAX to speed up SVI state updates."""
    @jit
    def update_svi(s, i):
        u, l = svi.stable_update(s, *args, **kwargs)
        return (u,l)
        
    u, losses = lax.scan(update_svi, svi_state, jnp.arange(num_iters), length=num_iters)
    return u, losses

class NumpyroSampler(SuperphotSampler):
    """Samplers which use numpyro."""

    def single_hierarchical_event(
        self, event_idx, t, obsflux, uncertainties, parameter_map,
        start_idx, end_idx, cube
    ):
        max_length = 300
        # parameter_map.shape = (7, len(t)), cube.shape = (28,)
        new_cube_all = jnp.take(cube, parameter_map)
        
        # Create a mask for the current event
        mask = jnp.arange(max_length) < (end_idx - start_idx)
        
        # Extract data for the current event using the mask
        t_event = jnp.where(mask, lax.dynamic_slice(t, (start_idx,), (max_length,)), 0.)
        obsflux_event = jnp.where(mask, lax.dynamic_slice(obsflux, (start_idx,), (max_length,)), 0.)
        uncertainties_event = jnp.where(mask, lax.dynamic_slice(uncertainties, (start_idx,), (max_length,)), 1.)

        new_cube = jnp.where(
            mask[None, :],
            lax.dynamic_slice(new_cube_all, (0, start_idx), (7, max_length)),
            jnp.zeros((7, max_length))
        )

        updated_column1 = jnp.where(mask, new_cube[4], jnp.ones(max_length))
        updated_column2 = jnp.where(mask, new_cube[5], jnp.ones(max_length))

        new_cube = new_cube.at[4,:].set(updated_column1)
        new_cube = new_cube.at[5,:].set(updated_column2)

        fit_constraint = jnp.max(villar_fit_constraint(new_cube))

        amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = new_cube
        phase = jnp.clip(t_event - t_0, min=-50.*tau_rise, max=None)
        phase = jnp.clip(phase, min=-50.*tau_fall + gamma, max=None)
        flux_const = amp / (1.0 + jnp.exp(-phase / tau_rise))

        flux = flux_const * jnp.where(
            gamma - phase >= 0,
            (1 - beta * phase),
            (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
        )

        sigma_tot = jnp.sqrt(uncertainties_event**2 + extra_sigma**2)

        #print(jnp.sum((flux - obsflux_event)**2 / sigma_tot**2) / jnp.sum(mask))

        return flux, sigma_tot, mask, fit_constraint, obsflux_event

    def create_hierarchical_jax_model(
        self,
        t=None,
        obsflux=None,
        uncertainties=None,
        parameter_map=None,
        start_idxs=None,
        end_idxs=None
    ):
        cube_all_events = self._prior_func()

        if t is None:
            return None
                
        num_events = len(start_idxs)
        index_array = jnp.arange(num_events)

        fluxes, sigma_tots, masks, factors, obsfluxes = vmap(self.single_hierarchical_event, in_axes=(0, None, None, None, None, 0, 0, 0))(
            index_array, t, obsflux, uncertainties, parameter_map,
            start_idxs, end_idxs, cube_all_events
        )

        factors = factors[:, jnp.newaxis]
        event_idx = index_array[:, jnp.newaxis]

        with numpyro.plate("events", num_events, dim=-2):
            numpyro.factor(
                f"vf_constraint_{event_idx}",
                -1000. * factors
            )
            numpyro.sample(f"obs_{event_idx}", dist.Normal(fluxes, sigma_tots).mask(masks), obs=obsfluxes)

            
    def create_jax_model(
        self,
        t=None,
        obsflux=None,
        uncertainties=None,
        parameter_map=None,
    ):  # pylint: disable=too-many-locals
        """Create a JAX model for MCMC.

        Parameters
        ----------
        t : array-like, optional
            Time values. Defaults to None.
        obsflux : array-like, optional
            Observed flux values. Defaults to None.
        uncertainties : array-like, optional
            Flux uncertainties. Defaults to None.
        max_flux : float, optional
            Maximum flux value. Defaults to None.
        priors : MultibandPriors
            priors for all bands in lightcurves
        """
        cube = self._prior_func()
        
        if t is None:
            return None
        
        # convert cube to (num_params x len(t))
        new_cube = cube[parameter_map]
        
        constraint = villar_fit_constraint(new_cube)
        
        numpyro.factor(
            "vf_constraint",
            -1000. * jnp.max(constraint)
        )

        (
            amp, beta, gamma, t_0,
            tau_rise, tau_fall, extra_sigma
        ) = new_cube

        phase = jnp.clip(t - t_0, min=-50.*tau_rise, max=None)
        phase = jnp.clip(phase, min=-50.*tau_fall + gamma, max=None)
        flux_const = amp / (1.0 + jnp.exp(-phase / tau_rise))

        flux = flux_const * jnp.where(
            gamma - phase >= 0,
            (1 - beta * phase),
            (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
        )
        
        sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)
        
        numpyro.sample("obs", dist.Normal(flux, sigma_tot), obs=obsflux)
        
    def create_jax_guide(
            self,
            num_events=None,
            t=None,
            obsflux=None,
            uncertainties=None,
            parameter_map=None,
            start_idxs=None,
            end_idxs=None
        ):
        """JAX guide function for MCMC.
        """
        self._priors.jax_guide(num_events=num_events)
            
    def __init__(
            self,
            priors: SamplerPrior,
            random_state: int,
            num_events=None,
            *args,
            **kwargs,
        ):
        super().__init__(priors)
        self._rng = random.key(random_state)
        self._prior_func = partial(self._priors.sample, cube=None, use_numpyro=True, num_events=num_events)
        if num_events:
            self._jax_model = self.create_hierarchical_jax_model
        else:
            self._jax_model = self.create_jax_model
        self._jax_guide = partial(self.create_jax_guide, num_events=num_events)
        self._orig_num_times: Optional[int] = None
        self._X = jnp.array([])
        self._y = jnp.array([])
        
        self.priors = None
        self.random_state = None # filler for repr

    def fit(
            self, X: NDArray[jnp.object_], # pylint: disable=invalid-name
            y: NDArray[jnp.float32],
            orig_num_times=None,
            event_indices=None,
        ) -> None: 
        """Fit the data.

        Parameters
        ----------
        X : np.ndarray
            The X data to fit. If 1d, will be reshaped to 2d.
            First column = times, second column = bands, third column = errors.
        y : np.ndarray
            The y data to fit.
        orig_num_times : the original number of datapoints. Important when calculating
            a score based on DOF with an artificially padded input. Defaults to the
            length of X.
        """
        super().fit(X,y,event_indices=event_indices)

        """
        _, band_counts = np.unique(X[:, 1], return_counts=True)
        print(band_counts)
        if not jnp.all(jnp.diff(band_counts) == 0): # if different counts
            raise ValueError("There must be same number of points in each band.")
        """
        if orig_num_times is not None:
            self._orig_num_times = orig_num_times
        
        self._y = jnp.array(self._y, dtype=jnp.float32)
        
        self._param_map = jnp.zeros((self._nparams+3, len(self._X)), dtype=int)
        for i, param in enumerate(self._base_params):
            for b in self._unique_bands:
                b_idxs = self._X[:,1] == b
                self._param_map = self._param_map.at[i,b_idxs].set(
                    jnp.where(self._params == f'{param}_{b}')[0][0]
                )
                
    def _process_samples(self, samples_df):
        """Convert parameter dict from numpyro to SamplerResult."""
        # transform log-Gaussian and relative params
        samples_df = self._priors.transform(samples_df)
        self.result = SamplerResult(samples_df, sampler_name=self._sampler_name)
        self._is_fitted = True
        self.result.score = np.array(
            self.score(self._X, self._y, orig_num_times=self._orig_num_times)
        )

    def _process_samples_hierarchical(
            self, prior_loc_samples,
            prior_scale_samples, indiv_samples
        ):
        """Convert parameter dict from numpyro to SamplerResult."""
        self.result_arr = []
        self._is_fitted = True

        # first handle global priors
        prior_mu_df = pd.DataFrame(np.array(prior_loc_samples), columns=self._params)
        prior_mu_transformed = self._priors.transform(prior_mu_df, relative=True)
        self.result = SamplerResult(prior_mu_transformed, sampler_name=self._sampler_name)
        self.result.score = np.nan * np.ones(len(prior_loc_samples))
        self.result_arr.append(self.result)

        prior_sigma_df = pd.DataFrame(np.array(prior_scale_samples), columns=self._params)
        prior_sigma_transformed = self._priors.transform(prior_sigma_df, relative=True)
        self.result = SamplerResult(prior_sigma_transformed, sampler_name=self._sampler_name)
        self.result.score = np.nan * np.ones(len(prior_scale_samples))
        self.result_arr.append(self.result)

        for i, s in enumerate(indiv_samples):
            samples_df = pd.DataFrame(np.array(s), columns=self._params)
            s_transformed = self._priors.transform(samples_df)

            print(prior_mu_transformed.mean(axis=0).sub(s_transformed.mean(axis=0)) / s_transformed.std(axis=0))
            print(prior_mu_transformed.mean(axis=0).sub(s_transformed.mean(axis=0)) / prior_sigma_transformed.mean(axis=0))

            self.result = SamplerResult(s_transformed, sampler_name=self._sampler_name)
            
            self.result.score = np.array(
                self.score(
                    self._X[self._idxs[i,0]:self._idxs[i,1]],
                    self._y[self._idxs[i,0]:self._idxs[i,1]],
                    orig_num_times=self._orig_num_times[i]
                )
            )
            self.result_arr.append(self.result)

        self.result = self.result_arr        
        

    def _reduced_chi_squared(self, X, y, y_pred, orig_num_times: Optional[int]=None):
        """Returns the reduced chi-squared value of the model.

        Parameters
        ----------
        X : np.ndarray
            The x data to score.
        y : np.ndarray
            The y data to score.
        y_pred : np.ndarray
            The predicted y data.
        orig_num_times: int, optional
            length of y before padding. If None,
            defaults to len(y)

        Returns
        -------
        float
            The reduced chi-squared value.
        """
        if orig_num_times is None:
            orig_num_times = len(y)
            
        return jnp.sum(
            (y[jnp.newaxis,:] - y_pred) ** 2 / self._eff_variance(X) / (orig_num_times - self._nparams),
            axis=1,
        )

class NUTSSampler(NumpyroSampler):
    """NUTS sampling using numpyro."""

    def __init__(
            self,
            priors: SamplerPrior,
            num_warmup: int=10_000,
            num_samples: int=10_000,
            num_chains: int=4,
            random_state: int=42,
        ):
        super().__init__(priors, random_state)
        self._sampler_name = 'superphot_nuts'

        kernel = NUTS(self._jax_model, init_strategy=init_to_uniform)

        self._mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method="parallel",
            jit_model_args=True,
        )

    def fit(
            self, X: NDArray[jnp.object_], # pylint: disable=invalid-name
            y: NDArray[jnp.float32],
            orig_num_times: Optional[int] = None,
        ) -> None: 
        """Fit the data.

        Parameters
        ----------
        X : np.ndarray
            The X data to fit. If 1d, will be reshaped to 2d.
            First column = times, second column = bands, third column = errors.
        y : np.ndarray
            The y data to fit.
        orig_num_times : the original number of datapoints. Important when calculating
            a score based on DOF with an artificially padded input. Defaults to the
            length of X.
        """
        super().fit(X,y,orig_num_times)
        
        self._mcmc.run(
            self._rng,
            obsflux=self._y,
            t=jnp.array(self._X[:,0], dtype=jnp.float32), # type: ignore
            uncertainties=jnp.array(self._X[:,2], dtype=jnp.float32), # type: ignore
            parameter_map=self._param_map,
        )
        params = self._mcmc.get_samples()
        params_concat = np.append(params['base_samples'], params['relative_samples'], axis=1)

        self._process_samples(pd.DataFrame(params_concat, columns=self._params))


class SVISampler(NumpyroSampler):
    """SVI sampling using numpyro."""

    def __init__(
            self,
            priors: SamplerPrior,
            num_iter=10_000,
            step_size=0.001,
            random_state: int = 42,
            num_events=None,
        ):
        super().__init__(priors, random_state, num_events)
        self._sampler_name = 'superphot_svi'
        self.step_size = step_size
        self.num_iter = num_iter
        
        optimizer = numpyro.optim.Adam(self.step_size)
        self._svi = SVI(self._jax_model, self._jax_guide, optimizer, loss=Trace_ELBO())
        
        self._lax_jit = jit(lax_helper_function, static_argnums=(0, 2))
        self._svi_state = None
        

    def reset(self):
        """Reset sampler, in the case it gets stuck in poor local minima."""
        self._svi_state = self._svi.init(self._rng)
        
    def fit(
            self, X: NDArray[jnp.object_], # pylint: disable=invalid-name
            y: NDArray[jnp.float32],
            orig_num_times: Optional[int] = None,
            event_indices = None
        ) -> None: 
        """Fit the data.

        Parameters
        ----------
        X : np.ndarray
            The X data to fit. If 1d, will be reshaped to 2d.
            First column = times, second column = bands, third column = errors.
        y : np.ndarray
            The y data to fit.
        orig_num_times : the original number of datapoints. Important when calculating
            a score based on DOF with an artificially padded input. Defaults to the
            length of X.
        """
        
        if event_indices is not None:
            # pad end of arrays for masking later
            X_padding = np.repeat([[999_999, self._unique_bands[0], 1.0],], 1000, axis=0)
            X_pad = np.concatenate([X, X_padding], axis=0)
            y_padding = np.zeros(1000)
            y_pad = np.append(y, y_padding)
        else:
            X_pad = X
            y_pad = y

        super().fit(
            X_pad,
            y_pad,
            orig_num_times=orig_num_times,
            event_indices=event_indices
        )

        if self._svi_state is None:
            self.reset()

        if event_indices is not None: #hierarchical
            self._svi_state, elbo_losses = self._lax_jit(
                self._svi,
                self._svi_state,
                self.num_iter,
                obsflux=self._y,
                t=jnp.array(self._X[:,0], dtype=jnp.float32),
                uncertainties=jnp.array(self._X[:,2], dtype=jnp.float32),
                start_idxs=jnp.array(self._idxs[:,0], dtype=int),
                end_idxs=jnp.array(self._idxs[:,1], dtype=int),
                parameter_map=self._param_map,
            )

        else:
            self._svi_state, elbo_losses = self._lax_jit(
                self._svi,
                self._svi_state,
                self.num_iter,
                obsflux=self._y,
                t=jnp.array(self._X[:,0], dtype=jnp.float32),
                uncertainties=jnp.array(self._X[:,2], dtype=jnp.float32),
                parameter_map=self._param_map,
            )

        params = self._svi.get_params(self._svi_state)

        if event_indices is not None:
            params_loc = jnp.concatenate([
                params['loc_base'],
                params['loc_relative']
            ], axis=1)
            params_scale = jnp.concatenate([
                params['scale_base'],
                params['scale_relative']
            ], axis=1)

            global_mu_loc = jnp.concatenate([
                params['global_mu_base_loc'],
                params['global_mu_rel_loc']
            ])
            global_mu_scale = jnp.concatenate([
                params['global_mu_base_sigma'],
                params['global_mu_rel_sigma']
            ])

            global_scale_loc = jnp.concatenate([
                params['global_sigma_base_loc'],
                params['global_sigma_rel_loc']
            ])
            global_scale_scale = jnp.concatenate([
                params['global_sigma_base_sigma'],
                params['global_sigma_rel_sigma']
            ])

            global_mu_arr = global_mu_loc + random.normal(
                key=self._rng, shape=(1000,)
            )[:,jnp.newaxis] * global_mu_scale
            global_scale_arr = global_scale_loc + random.normal(
                key=self._rng, shape=(1000,)
            )[:,jnp.newaxis] * global_scale_scale

            indiv_param_arr = params_loc[:,jnp.newaxis,:] + random.normal(
                key=self._rng, shape=(len(event_indices), 1000)
            )[:,:,jnp.newaxis] * params_scale[:,jnp.newaxis,:]

            self._process_samples_hierarchical(global_mu_arr, global_scale_arr, indiv_param_arr)

        else:
            params_loc = jnp.concatenate([
                params['loc_base'],
                params['loc_relative']
            ])
            params_scale = jnp.concatenate([
                params['scale_base'],
                params['scale_relative']
            ])

            param_arr = params_loc + random.normal(
                key=self._rng, shape=(1000,)
            )[:,jnp.newaxis] * params_scale

            posterior_samples = pd.DataFrame(np.array(param_arr), columns=self._params)
            self._process_samples(posterior_samples)