"""MCMC sampling using numpyro."""
from typing import Optional

from numpy.typing import NDArray
import numpyro
import numpyro.distributions as dist
import pandas as pd
import jax.numpy as jnp
from jax import random, lax, jit
from jax._src import config
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_uniform
from snapi import SamplerResult
from sklearn.utils import check_random_state

from superphot_plus.constants import PAD_SIZE
from superphot_plus.surveys.fitting_priors import MultibandPriors, PriorFields
from superphot_plus.samplers.superphot_sampler import SuperphotSampler
from superphot_plus.utils import (
    get_numpyro_cube,
    villar_fit_constraint
)
#config.update("jax_enable_x64", True)
#numpyro.enable_x64()

def prior_helper(priors, aux_b=None):
    """Helper function to sample prior values. If aux_b is not None,
    appends aux_b to value names.

    Parameters
    ----------
    priors : CurvePriors
        The priors for one band
    max_flux : float
        Max flux of the light curve.
    aux_b : str, optional
        The name of the auxiliary band, if it is auxiliary. Defaults to None, which
        assumes it's the base band.
    """
    if aux_b is None:
        amp = 10 ** numpyro.sample("logA", trunc_norm(priors.amp))
        beta = numpyro.sample(
            "beta",
            trunc_norm(priors.beta, high = 1.0/(10**priors.tau_fall.clip_a + 10**priors.gamma.clip_a)),
        )
        t_0 = numpyro.sample("t0", trunc_norm(priors.t_0))
        tau_rise = 10 ** numpyro.sample("log_tau_rise", trunc_norm(priors.tau_rise))
        tau_fall = 10 ** numpyro.sample(
            "log_tau_fall",
            trunc_norm(priors.tau_fall, high = jnp.log10(1./beta - 10**priors.gamma.clip_a))
        )
        gamma = 10 ** numpyro.sample(
            "log_gamma",
            trunc_norm(priors.gamma, high=jnp.log10((1.0 - beta * tau_fall) / beta))
        )
        extra_sigma = 10 ** numpyro.sample("log_extra_sigma", trunc_norm(priors.extra_sigma))
        
    else:
        suffix = "_" + str(aux_b)
        amp = 10**numpyro.sample(f"A{suffix}", trunc_norm(priors.amp))
        beta = 10**numpyro.sample(f"beta{suffix}", trunc_norm(priors.beta))
        gamma = 10**numpyro.sample(f"gamma{suffix}", trunc_norm(priors.gamma))
        t_0 = numpyro.sample(f"t0{suffix}", trunc_norm(priors.t_0))
        tau_rise = 10**numpyro.sample(f"tau_rise{suffix}", trunc_norm(priors.tau_rise))
        tau_fall = 10**numpyro.sample(f"tau_fall{suffix}", trunc_norm(priors.tau_fall))
        extra_sigma = 10**numpyro.sample(f"extra_sigma{suffix}", trunc_norm(priors.extra_sigma))
    
    return amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma


def lax_helper_function(svi, svi_state, num_iters, *args, **kwargs):
    """Helper function using LAX to speed up SVI state updates."""

    @jit
    def update_svi(s, _):
        return svi.stable_update(s, *args, **kwargs)

    u = svi_state
    u, _ = lax.scan(update_svi, svi_state, None, length=num_iters)
    return u


def trunc_norm(fields: PriorFields, low=None, high=None):
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
    if high is None:
        high = fields.clip_b
    else:
        high = jnp.minimum(high, fields.clip_b)
    if low is None:
        low = fields.clip_a
    else:
        low = jnp.maximum(low, fields.clip_b)
        
    return dist.TruncatedNormal(
        loc=fields.mean, scale=fields.std, low=low, high=high, validate_args=True
    )


def create_jax_model(
    priors,
    t=None,
    obsflux=None,
    uncertainties=None,
    max_flux=None,
    ref_params=None
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
    ref_priors = priors.bands[priors.reference_band]

    if ref_params is not None:
        (
            amp, beta, gamma, t_0,
            tau_rise, tau_fall, extra_sigma
        ) = ref_params
    else:
        (
            amp, beta, gamma, t_0,
            tau_rise, tau_fall, extra_sigma
        ) = prior_helper(ref_priors)

    constraint = villar_fit_constraint([beta, gamma, tau_rise, tau_fall])
    numpyro.factor(
        "vf_constraint",
        -1000. * constraint
    )
        
    phase = t - t_0
    flux_const = max_flux * amp / (1.0 + jnp.exp(-phase / tau_rise))
    sigmoid = 1 / (1 + jnp.exp(10.0 * (gamma - phase)))

    flux = flux_const * (
        (1 - sigmoid) * (1 - beta * phase)
        + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
    )
    
    sigma_tot = jnp.sqrt(uncertainties**2 + max_flux**2 * extra_sigma**2)

    # auxiliary bands
    for b_idx, uniq_b in enumerate(priors.ordered_bands):
        if uniq_b == priors.reference_band:
            continue
            
        b_priors = priors.bands[uniq_b]

        (
            amp_ratio,
            beta_ratio,
            gamma_ratio,
            t0_shift,
            tau_rise_ratio,
            tau_fall_ratio,
            extra_sigma_ratio,
        ) = prior_helper(b_priors, uniq_b)

        amp_b = max_flux * amp * amp_ratio
        beta_b = beta * beta_ratio
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
        inc_band_ix = jnp.arange(b_idx * PAD_SIZE, (b_idx + 1) * PAD_SIZE)

        phase_b = (t - t0_b)[inc_band_ix]
        flux_const_b = amp_b / (1.0 + jnp.exp(-phase_b / tau_rise_b))
        sigmoid_b = 1 / (1 + jnp.exp(10.0 * (gamma_b - phase_b)))

        flux = flux.at[inc_band_ix].set(
            flux_const_b
            * (
                (1 - sigmoid_b) * (1 - beta_b * phase_b)
                + sigmoid_b * (1 - beta_b * gamma_b) * jnp.exp(-(phase_b - gamma_b) / tau_fall_b)
            )
        )

        sigma_tot = sigma_tot.at[inc_band_ix].set(
            jnp.sqrt(uncertainties[inc_band_ix] ** 2 + (max_flux*extra_sigma_ratio*extra_sigma)**2)
        )

    _ = numpyro.sample("obs", dist.Normal(flux, sigma_tot), obs=obsflux)


def create_jax_guide(priors, t=None, obsflux=None, uncertainties=None, max_flux=None, ref_params=None):
    """JAX guide function for MCMC.

    Parameters
    ----------
    priors : MultibandPriors
        priors for all bands in lightcurves
    """

    def numpyro_sample(prefix: str, fields: PriorFields, param_constraint: float):
        param_mu = numpyro.param(
            f"{prefix}_mu",
            fields.mean,
            constraint=constraints.interval(fields.clip_a, fields.clip_b),
        )
        param_sigma = numpyro.param(f"{prefix}_sigma", param_constraint, constraint=constraints.positive)
        numpyro.sample(prefix, dist.Normal(param_mu, param_sigma))

    ref_priors = priors.bands[priors.reference_band]
    numpyro_sample("logA", ref_priors.amp, 1e-3)
    numpyro_sample("beta", ref_priors.beta, 1e-5)
    numpyro_sample("log_gamma", ref_priors.gamma, 1e-3)
    numpyro_sample("t0", ref_priors.t_0, 1e-3)
    numpyro_sample("log_tau_rise", ref_priors.tau_rise, 1e-3)
    numpyro_sample("log_tau_fall", ref_priors.tau_fall, 1e-3)
    numpyro_sample("log_extra_sigma", ref_priors.extra_sigma, 1e-3)

    # aux bands
    for uniq_b in priors.aux_bands:
        b_priors = priors.bands[uniq_b]
        numpyro_sample("A_" + uniq_b, b_priors.amp, 1e-3)
        numpyro_sample("beta_" + uniq_b, b_priors.beta, 1e-3)
        numpyro_sample("gamma_" + uniq_b, b_priors.gamma, 1e-3)
        numpyro_sample("t0_" + uniq_b, b_priors.t_0, 1e-3)
        numpyro_sample("tau_rise_" + uniq_b, b_priors.tau_rise, 1e-3)
        numpyro_sample("tau_fall_" + uniq_b, b_priors.tau_fall, 1e-3)
        numpyro_sample("extra_sigma_" + uniq_b, b_priors.extra_sigma, 1e-3)


class NumpyroSampler(SuperphotSampler):
    """Samplers which use numpyro."""
    def __init__(
            self,
            priors: MultibandPriors,
            random_state: int,
            *args,
            **kwargs,
        ):
        super().__init__(priors)
        self._random_state = check_random_state(random_state)
        self._rng = jnp.random.key(self._random_state)

        def jax_model(t=None, obsflux=None, uncertainties=None, max_flux=None, ref_params=None):
            create_jax_model(priors, t, obsflux, uncertainties, max_flux, ref_params)

        def jax_guide(**kwargs):  # pylint: disable=unused-argument
            create_jax_guide(priors)

        self._jax_model = jax_model
        self._jax_guide = jax_guide
        self._orig_num_times = None
        self._padded_len = 0
        self._X = None
        self._y = None

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
        _, band_counts = jnp.unique(X[:, 1], return_counts=True)
        if not jnp.all(jnp.diff(band_counts) == 0): # if different counts
            raise ValueError("There must be same number of points in each band.")

        self._padded_len = band_counts[0]

        if orig_num_times is not None:
            self._orig_num_times = orig_num_times
        
        self._rearrange_inputs()

    def _rearrange_inputs(self):
        """Rearrange X and y so padded band order matches self._unique_bands"""
        rearranged_X = jnp.zeros(self._X.shape)
        rearranged_y = jnp.zeros(self._y.shape)
        for i, b in enumerate(self._unique_bands):
            b_mask = self._X[:,1] == b
            rearranged_X[self._padded_len*(i+1):self._padded_len*(i+2)] = self._X[b_mask]
            rearranged_y[self._padded_len*(i+1):self._padded_len*(i+2)] = self._y[b_mask]

        self._X = rearranged_X
        self._y = rearranged_y

    def _process_samples(self, params):
        """Convert parameter dict from numpyro to SamplerResult."""
        posterior_samples = {}
        for param in params:
            if param[-2:] == "mu":
                posterior_samples[param[:-3]] = self._rng.normal(
                    loc=params[param], scale=params[param[:-2] + "sigma"], size=100
                )
        posterior_cube = get_numpyro_cube(
            posterior_samples, 1,
            self.priors.reference_band, self.priors.ordered_bands
        )[0]

        samples_df = pd.DataFrame(
            posterior_cube,
            columns=self._create_param_names()
        )

        self.result = SamplerResult(samples_df, sampler_name=self._sampler_name)
        self._is_fitted = True
        self.result.score = self.score(self._X, self._y, self._orig_num_times)

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
            priors: MultibandPriors,
            num_warmup: int=10_000,
            num_samples: int=10_000,
            num_chains: int=4,
            random_state: int = None,
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
            obsflux=y,
            t=X[:,0],
            uncertainties=X[:,2],
            max_flux=1,
        )
        params = self._mcmc.get_samples()

        self._process_samples(params)


class SVISampler(NumpyroSampler):
    """SVI sampling using numpyro."""

    def __init__(
            self,
            priors: MultibandPriors,
            num_iter=10_000,
            step_size=0.001,
            random_state: int = None,
        ):
        super().__init__(priors, random_state)

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
                random.PRNGKey(1),
                obsflux=y,
                t=X[:,0],
                uncertainties=X[:,2],
            )
    
        self._svi_state = self._lax_jit(
            self._svi,
            self._svi_state,
            self._num_iter,
            obsflux=y,
            t=X[:,0],
            uncertainties=X[:,2],
        )

        params = self._svi.get_params(self._svi_state)
        self._process_samples(params)