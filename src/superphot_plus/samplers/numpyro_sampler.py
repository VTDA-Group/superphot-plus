"""MCMC sampling using numpyro."""
from typing import Optional
from functools import partial

from numpy.typing import NDArray
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import jax.numpy as jnp
from jax import jit, lax, random, value_and_grad, vmap, config, debug
from numpyro.infer.svi import SVIState
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_uniform
from snapi.analysis import SamplerPrior, SamplerResult

from superphot_plus.samplers.superphot_sampler import SuperphotSampler
from superphot_plus.utils import villar_fit_constraint


config.update('jax_platform_name', 'cpu')

def _make_loss_fn(
    elbo,
    rng_key,
    constrain_fn,
    model,
    guide,
    args,
    kwargs,
    static_kwargs,
):
    def loss_fn(params):
        params = constrain_fn(params)
        return elbo.loss(
            rng_key, params, model, guide, *args, **kwargs, **static_kwargs
        )

    return loss_fn


def stablish_update(svi, svi_state, *args, forward_mode_differentiation=False, **kwargs):
    """
    Take a single step of SVI (possibly on a batch / minibatch of data),
    using the optimizer.

    :param svi_state: current state of SVI.
    :param args: arguments to the model / guide (these can possibly vary during
        the course of fitting).
    :param forward_mode_differentiation: boolean flag indicating whether to use forward mode differentiation.
        Defaults to False.
    :param kwargs: keyword arguments to the model / guide (these can possibly vary
        during the course of fitting).
    :return: tuple of `(svi_state, loss)`.
    """
    rng_key, rng_key_step = random.split(svi_state.rng_key)
    loss_fn = _make_loss_fn(
        svi.loss,
        rng_key_step,
        svi.constrain_fn,
        svi.model,
        svi.guide,
        args,
        kwargs,
        svi.static_kwargs,
    )
    state = svi_state.optim_state
    params = svi.optim.get_params(state)
    #debug.print("Params: {}", params)
    loss_val, grads = value_and_grad(loss_fn)(params)

    optim_state = lax.cond(
        jnp.isfinite(ravel_pytree(grads)[0]).all(),
        lambda _: svi.optim.update(grads, state),
        lambda _: state,
        None,
    )
    #debug.print("Optim: {}", optim_state)
    return SVIState(optim_state, None, rng_key), loss_val


def lax_helper_function(svi, svi_state, num_iters, *args, **kwargs):
    """Helper function using LAX to speed up SVI state updates."""
    @jit
    def update_svi(s, i):
        return stablish_update(svi, s, *args, **kwargs)

    """
    for i in range(num_iters):
        svi_state, _ = update_svi(svi_state, i)
    return svi_state
    """
        
    u, _ = lax.scan(update_svi, svi_state, jnp.arange(num_iters), length=num_iters)
    return u

class NumpyroSampler(SuperphotSampler):
    """Samplers which use numpyro."""

    def single_hierarchical_event(
        self, t, obsflux, uncertainties, parameter_map,
        start_idx, end_idx, cube
    ):
        max_length = self.max_length
        
        # parameter_map.shape = (7, len(t)), cube.shape = (28,)
        new_cube_all = jnp.take(cube, parameter_map)
        
        # Create a mask for the current event
        # Add padding mask to existing event mask
        mask = jnp.arange(max_length) < (end_idx - start_idx)

        # Extract data for the current event using the mask
        sliced_t = lax.dynamic_slice(t, (start_idx,), (max_length,))
        t_event = jnp.where(mask, sliced_t, jnp.zeros(max_length))
        sliced_obsflux = lax.dynamic_slice(obsflux, (start_idx,), (max_length,))
        obsflux_event = jnp.where(mask, sliced_obsflux, jnp.zeros(max_length))
        sliced_unc = lax.dynamic_slice(uncertainties, (start_idx,), (max_length,))
        uncertainties_event = jnp.where(mask, sliced_unc, jnp.ones(max_length))

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
        gamma = jnp.clip(gamma, min=0.0, max=None)
        phase = jnp.clip(t_event - t_0, min=-50.*tau_rise, max=None)

        flux_const = amp / (1.0 + jnp.exp(-phase / tau_rise))

        flux = flux_const * jnp.where(
            gamma - phase >= 0,
            (1 - beta * phase),
            (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
        )

        sigma_tot = jnp.sqrt(uncertainties_event**2 + extra_sigma**2)

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

        batch_size = 100
        num_events = len(start_idxs)
        num_batches = (num_events + batch_size - 1) // batch_size

        # Pre-pad arrays to ensure fixed shapes for JIT
        pad_size = num_batches * batch_size - num_events
        start_idxs_padded = jnp.pad(start_idxs, (0, pad_size))
        end_idxs_padded = jnp.pad(end_idxs, (0, pad_size))
        cube_padded = jnp.pad(cube_all_events, ((0, pad_size), (0, 0)))
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            # No need for min() since we pre-padded
            batch_slice = slice(start, start + batch_size)
            
            # Process each batch with vmap
            fluxes, sigma_tots, masks, factors, obsfluxes = vmap(self.single_hierarchical_event, in_axes=(None, None, None, None, 0, 0, 0))(
                t, obsflux, uncertainties, parameter_map,
                start_idxs_padded[batch_slice], 
                end_idxs_padded[batch_slice],
                cube_padded[batch_slice]
            )
            batch_indices = jnp.arange(batch_size)

            with numpyro.plate(f"obs_plate_{batch_idx}", batch_size, dim=-2):
                numpyro.factor(
                    f"vf_constraint_{batch_idx}_{batch_indices}",
                    -10_000. * factors
                )

                numpyro.sample(f"obs_{batch_idx}_{batch_indices}", dist.Normal(fluxes, sigma_tots).mask(masks), obs=obsfluxes)

            
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
            -10_000. * jnp.max(constraint)
        )

        amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = new_cube
        gamma = jnp.clip(gamma, min=0.0, max=None)
        phase = jnp.clip(t - t_0, min=-50.*tau_rise, max=None)

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
            max_length=1000,
            *args,
            **kwargs,
        ):
        super().__init__(priors)
        self.max_length = max_length
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
        self._generate_param_map()
        
    
    def _generate_param_map(self):
        """Generate param map."""
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
            max_length=1000,
        ):
        super().__init__(priors, random_state, num_events, max_length)
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
        #self._svi_state = init(self._svi, self._rng)
        
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
            self._svi_state = self._lax_jit(
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
            self._svi_state = self._lax_jit(
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
                key=self._rng, shape=(1000, len(params_loc))
            ) * params_scale

            posterior_samples = pd.DataFrame(np.array(param_arr), columns=self._params)
            self._process_samples(posterior_samples)