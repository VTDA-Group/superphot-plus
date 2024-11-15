"""MCMC sampling using numpyro."""
from typing import Optional

from numpy.typing import NDArray
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import jax.numpy as jnp
from jax import random, lax, jit, value_and_grad
from jax.core import concrete_or_error
from jax._src import config
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_uniform
from snapi.analysis import SamplerPrior, SamplerResult
from sklearn.utils import check_random_state

from superphot_plus.samplers.superphot_sampler import SuperphotSampler
from superphot_plus.utils import (
    get_numpyro_cube,
    villar_fit_constraint
)

numpyro.set_host_device_count(1)
#config.update("jax_enable_x64", True)
#config.update("jax_disable_jit", True)
#config.update("jax_debug_nans", True)
#numpyro.enable_x64()


def lax_helper_function(svi, svi_state, num_iters, *args, **kwargs):
    """Helper function using LAX to speed up SVI state updates."""

    @jit
    def update_svi(s, i):
        print(i)
        return svi.update(s, *args, **kwargs)
    u, _ = lax.scan(update_svi, svi_state, jnp.arange(num_iters), length=num_iters)
    return u

class NumpyroSampler(SuperphotSampler):
    """Samplers which use numpyro."""
    def __init__(
            self,
            priors: SamplerPrior,
            random_state: int,
            pad_size: int,
            *args,
            **kwargs,
        ):
        super().__init__(priors)
        self._rng = random.key(random_state)
        self._pad_size = concrete_or_error(int, pad_size, "pad_size must be static")
        self._ref_priors = self._priors[self._ref_params]

        def create_jax_model(
            t=None,
            obsflux=None,
            uncertainties=None,
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
            (
                amp, beta, gamma, t_0,
                tau_rise, tau_fall, extra_sigma
            ) = prior_helper(self._ref_priors)

            constraint = villar_fit_constraint([beta, gamma, tau_rise, tau_fall])

            numpyro.factor(
                "vf_constraint",
                -1000. * constraint
            )
                
            phase = t - t_0
            flux_const = amp / (1.0 + jnp.exp(-phase / tau_rise))

            flux = flux_const * jnp.where(
                gamma - phase >= 0,
                (1 - beta * phase),
                (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
            )
            
            sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)

            # auxiliary bands
            for b_idx, uniq_b in enumerate(self._unique_bands):
                if uniq_b == self._ref_band:
                    continue
                    
                b_priors = priors.bands[uniq_b]

                (
                    amp_ratio,
                    beta_shift,
                    gamma_ratio,
                    t0_shift,
                    tau_rise_ratio,
                    tau_fall_ratio,
                    extra_sigma_ratio,
                ) = prior_helper(b_priors, uniq_b)

                amp_b = amp * amp_ratio
                beta_b = beta + beta_shift
                gamma_b = gamma * gamma_ratio
                t0_b = t_0 + t0_shift
                tau_rise_b = tau_rise * tau_rise_ratio
                tau_fall_b = tau_fall * tau_fall_ratio
                extra_sigma_b = extra_sigma * extra_sigma_ratio
                
                constraint = villar_fit_constraint([beta_b, gamma_b, tau_rise_b, tau_fall_b])

                numpyro.factor(
                    f"vf_constraint_{uniq_b}",
                    -1000. * constraint
                )
                
                numpyro.factor(
                    f"sigma_constraint_{uniq_b}",
                    -1000. * jnp.maximum(extra_sigma_b - 10**(-0.8), 0.)
                )

                # base inc_band_ix on ordered_bands
                # ISSUE: CHANGE FOR LOOP TO LAX.SCAN OR FORILOOP (SEE GPT RESPONSE)
                inc_band_ix = jnp.arange(b_idx * self._pad_size, (b_idx+1) * self._pad_size)

                phase_b = (t - t0_b)[inc_band_ix]
                flux_const_b = amp_b / (1.0 + jnp.exp(-phase_b / tau_rise_b))
                #z = jnp.clip(10.0 * (gamma_b - phase_b), -88.0, 88.0)  # exp(88) is near the max float32 can handle
                #sigmoid_b = 1.0 / (1.0 + jnp.exp(z))

                flux = flux.at[inc_band_ix].set(
                    flux_const_b * jnp.where(
                        gamma_b - phase_b >= 0,
                        (1 - beta_b * phase_b),
                        (1 - beta_b * gamma_b) * jnp.exp(-(phase_b - gamma_b) / tau_fall_b)
                    )
                )

                sigma_tot = sigma_tot.at[inc_band_ix].set(
                    jnp.sqrt(uncertainties[inc_band_ix] ** 2 + extra_sigma_b**2)
                )
            _ = numpyro.sample("obs", dist.Normal(flux, sigma_tot), obs=obsflux)

        def create_jax_guide(t=None, obsflux=None, uncertainties=None):
            """JAX guide function for MCMC.
            """

            def numpyro_sample(prefix: str, fields: PriorFields, param_constraint: float):
                param_mu = numpyro.param(
                    f"{prefix}_mu",
                    fields.mean,
                    constraint=constraints.interval(fields.clip_a, fields.clip_b),
                )
                param_sigma = numpyro.param(f"{prefix}_sigma", param_constraint, constraint=constraints.positive)
                out = numpyro.sample(prefix, dist.Normal(param_mu, param_sigma, validate_args=True))

            ref_priors = priors.bands[priors.reference_band]
            numpyro_sample("logA", ref_priors.amp, 1e-3)
            numpyro_sample("beta", ref_priors.beta, 1e-5)
            numpyro_sample("log_gamma", ref_priors.gamma, 1e-3)
            numpyro_sample("t0", ref_priors.t_0, 1e-3)
            numpyro_sample("log_tau_rise", ref_priors.tau_rise, 1e-3)
            numpyro_sample("log_tau_fall", ref_priors.tau_fall, 1e-3)
            numpyro_sample("log_extra_sigma", ref_priors.extra_sigma, 1e-3)

            # aux bands
            for uniq_b in self._unique_bands:
                if uniq_b == self._ref_band:
                    continue
                b_priors = priors.bands[uniq_b]
                numpyro_sample("A_" + uniq_b, b_priors.amp, 1e-3)
                numpyro_sample("beta_" + uniq_b, b_priors.beta, 1e-5)
                numpyro_sample("gamma_" + uniq_b, b_priors.gamma, 1e-3)
                numpyro_sample("t0_" + uniq_b, b_priors.t_0, 1e-3)
                numpyro_sample("tau_rise_" + uniq_b, b_priors.tau_rise, 1e-3)
                numpyro_sample("tau_fall_" + uniq_b, b_priors.tau_fall, 1e-3)
                numpyro_sample("extra_sigma_" + uniq_b, b_priors.extra_sigma, 1e-3)
        self._jax_model = create_jax_model
        self._jax_guide = create_jax_guide
        self._orig_num_times: Optional[int] = None
        self._padded_len = 0
        self._X = jnp.array([])
        self._y = jnp.array([])

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
        
        self._rearrange_inputs()
        self._y = jnp.array(self._y, dtype=jnp.float32)

    def _rearrange_inputs(self):
        """Rearrange X and y so padded band order matches self._unique_bands"""
        rearranged_X = np.zeros(self._X.shape, dtype=self._X.dtype)
        rearranged_y = np.zeros(self._y.shape, dtype=self._y.dtype)
        for i, b in enumerate(self._unique_bands):
            b_mask = self._X[:,1] == b
            rearranged_X[self._padded_len*i:self._padded_len*(i+1)] = self._X[b_mask]
            rearranged_y[self._padded_len*i:self._padded_len*(i+1)] = self._y[b_mask]

        self._X = rearranged_X
        self._y = rearranged_y

    def _process_samples(self, params):
        """Convert parameter dict from numpyro to SamplerResult."""
        posterior_cube = get_numpyro_cube(
            params, self._ref_band, self._unique_bands
        )[0]

        samples_df = pd.DataFrame(
            posterior_cube,
            columns=self._create_param_names()
        )

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
        return jnp.median(
            jnp.sum(
                (y[jnp.newaxis,:] - y_pred) ** 2 / self._eff_variance(X) / (orig_num_times - self._nparams - 1),
                axis=1,
            )
        )

class NUTSSampler(NumpyroSampler):
    """NUTS sampling using numpyro."""

    def __init__(
            self,
            priors: SamplerPrior,
            pad_size: int,
            num_warmup: int=10_000,
            num_samples: int=10_000,
            num_chains: int=4,
            random_state: int=42,
        ):
        super().__init__(priors, random_state, pad_size)
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
        )
        params = self._mcmc.get_samples()

        self._process_samples(params)


class SVISampler(NumpyroSampler):
    """SVI sampling using numpyro."""

    def __init__(
            self,
            priors: SamplerPrior,
            pad_size: int,
            num_iter=10_000,
            step_size=0.001,
            random_state: int = 42,
        ):
        super().__init__(priors, random_state, pad_size)
        self._sampler_name = 'superphot_svi'

        optimizer = numpyro.optim.Adam(step_size=step_size)
        self._svi = SVI(self._jax_model, self._jax_guide, optimizer, loss=Trace_ELBO())
        self._num_iter = num_iter
        self._lax_jit = jit(lax_helper_function, static_argnums=(0, 2))
        self._svi_state = None

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
            self._svi_state = self._svi.init(
                self._rng,
                obsflux=self._y,
                t=jnp.array(self._X[:,0], dtype=jnp.float32),
                uncertainties=jnp.array(self._X[:,2], dtype=jnp.float32),
            )

        new_state, loss = self._svi.update(
            self._svi_state,
            obsflux=self._y,
            t=jnp.array(self._X[:,0], dtype=jnp.float32),
            uncertainties=jnp.array(self._X[:,2], dtype=jnp.float32),
        )

    
        self._svi_state = self._lax_jit(
            self._svi,
            self._svi_state,
            self._num_iter,
            obsflux=self._y,
            t=jnp.array(self._X[:,0], dtype=jnp.float32),
            uncertainties=jnp.array(self._X[:,2], dtype=jnp.float32),
        )

        params = self._svi.get_params(self._svi_state)

        posterior_samples = {}
        for param in params:
            if param[-2:] == "mu":
                mu = params[param][()]
                sig = params[param[:-2] + 'sigma'][()]
                posterior_samples[param[:-3]] = mu + sig * random.normal(
                    key=self._rng, shape=(100,)
                )

        self._process_samples(posterior_samples)