"""MCMC sampling using numpyro."""
from typing import Optional
from functools import partial

from numpy.typing import NDArray
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import jax.numpy as jnp
from jax import random, lax, jit, value_and_grad, grad
from jax.core import concrete_or_error
from jax._src import config
import traceback
import pickle
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_uniform
from snapi.analysis import SamplerPrior, SamplerResult
from sklearn.utils import check_random_state

from superphot_plus.samplers.superphot_sampler import SuperphotSampler
from superphot_plus.utils import villar_fit_constraint


numpyro.set_host_device_count(1)
#config.update("jax_enable_x64", True)
#config.update("jax_disable_jit", True)
#config.update("jax_debug_nans", True)
#numpyro.enable_x64()


def lax_helper_function(svi, svi_state, num_iters, *args, **kwargs):
    """Helper function using LAX to speed up SVI state updates."""
    @jit
    def update_svi(s, i):
        return svi.update(s, *args, **kwargs)

        """
        # Extract parameters
        params = svi.get_params(s)

        # Define a function to compute the ELBO manually
        def elbo_fn(param_dict):
            return -svi.loss(rng_key=s.rng_key, param_map=param_dict, model=svi.model, guide=svi.guide, *args, **kwargs)
        
        # Compute the gradient of the ELBO with respect to params
        grad_fn = value_and_grad(elbo_fn)
        elbo, gradients = grad_fn(params)
        
        
        # Print ELBO and gradients at certain intervals
        print(f"Step {i}, ELBO: {elbo}")
        print("Gradients:", gradients)
            
        return svi.update(s, *args, **kwargs)
        """

        
    u, losses = lax.scan(update_svi, svi_state, jnp.arange(num_iters), length=num_iters)
    return u, losses

class NumpyroSampler(SuperphotSampler):
    """Samplers which use numpyro."""
            
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
        
        constraint = villar_fit_constraint(new_cube) # FIX THIS
        
        numpyro.factor(
            "vf_constraint",
            -1000. * constraint
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
        
    def create_jax_guide(self, t=None, obsflux=None, uncertainties=None, parameter_map=None):
        """JAX guide function for MCMC.
        """
        self._priors.jax_guide()
            
    def __init__(
            self,
            priors: SamplerPrior,
            random_state: int,
            *args,
            **kwargs,
        ):
        super().__init__(priors)
        self._rng = random.key(random_state)
        self._prior_func = partial(self._priors.sample, cube=None, use_numpyro=True)
        self._jax_model = self.create_jax_model
        self._jax_guide = self.create_jax_guide
        self._orig_num_times: Optional[int] = None
        self._padded_len = 0
        self._X = jnp.array([])
        self._y = jnp.array([])
        
        self.priors = None
        self.random_state = None # filler for repr

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
        super().fit(X,y)
        _, band_counts = np.unique(X[:, 1], return_counts=True)
        if not jnp.all(jnp.diff(band_counts) == 0): # if different counts
            raise ValueError("There must be same number of points in each band.")

        self._padded_len = band_counts[0]

        if orig_num_times is not None:
            self._orig_num_times = orig_num_times
        
        self._y = jnp.array(self._y, dtype=jnp.float32)
        
        self._param_map = jnp.zeros((self._nparams+1, len(self._X)), dtype=int)
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
        self.result.score = self.score(self._X, self._y, orig_num_times=self._orig_num_times)

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
            param_map=self._param_map,
        )
        params = self._mcmc.get_samples()

        self._process_samples(pd.DataFrame(params))


class SVISampler(NumpyroSampler):
    """SVI sampling using numpyro."""

    def __init__(
            self,
            priors: SamplerPrior,
            num_iter=10_000,
            step_size=0.001,
            random_state: int = 42,
        ):
        super().__init__(priors, random_state)
        self._sampler_name = 'superphot_svi'
        self.step_size= step_size
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

        if self._svi_state is None:
            self.reset()
    
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
        
        params_loc = jnp.concatenate([
            params['loc_base'],
            params['loc_relative']
        ])
        params_scale = jnp.concatenate([
            params['scale_base'],
            params['scale_relative']
        ])
        
        #print(elbo_losses)
        #print(params_loc, params_scale)

        param_arr = params_loc + random.normal(
            key=self._rng, shape=(100,)
        )[:,jnp.newaxis] * params_scale

        posterior_samples = pd.DataFrame(np.array(param_arr), columns=self._params)
        self._process_samples(posterior_samples)