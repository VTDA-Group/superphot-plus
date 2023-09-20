"""Sampling using FlowMC"""

from functools import partialmethod
from typing import Callable, List

import jax
import jax.config as config
import jax.numpy as jnp
import numpy as np
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.LocalSampler_Base import LocalSamplerBase
from flowMC.sampler.Sampler import Sampler as FlowSampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from jax import jit
from jax.config import config
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm

from superphot_plus.constants import *
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.samplers.sampler import Sampler
from superphot_plus.surveys.fitting_priors import MultibandPriors

# MODIFIED FROM FLOWMC'S MALA TO HANDLE ARRAY OF DIFFERENT TIME STEPS PER PARAM
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


class MALA(LocalSamplerBase):
    """
    Metropolis-adjusted Langevin algorithm sampler class builiding the mala_sampler method

    Args:
        logpdf: target logpdf function
        jit: whether to jit the sampler
        params: dictionary of parameters for the sampler
    """

    def __init__(
        self, logpdf: Callable, jit: bool, params: dict, verbose: bool = False, use_autotune=False
    ) -> Callable:
        super().__init__(logpdf, jit, params)
        self.params = params
        self.logpdf = logpdf
        self.logpdf_vmap = jax.vmap(logpdf, in_axes=(0, None))
        self.verbose = verbose
        self.kernel = None
        self.kernel_vmap = None
        self.update = None
        self.update_vmap = None
        self.sampler = None
        self.use_autotune = use_autotune

    def make_kernel(self, return_aux=False) -> Callable:
        """
        Make a MALA kernel for a given logpdf.

        Args:
            logpdf : (Callable) The logpdf of the target distribution.

        Returns:
            mala_kernel (Callable) A MALA kernel.
        """

        def body(carry, this_key):
            this_position, dt, data = carry
            dt2 = dt * dt
            this_log_prob, this_d_log = jax.value_and_grad(self.logpdf)(this_position, data)
            proposal = this_position + dt2 * this_d_log / 2
            proposal += dt * jax.random.normal(this_key, shape=this_position.shape)
            return (proposal, dt, data), (proposal, this_log_prob, this_d_log)

        def mala_kernel(rng_key, position, log_prob, data, params={"step_size": 0.1}):
            """
            Metropolis-adjusted Langevin algorithm kernel.
            This function make a proposal and accept/reject it.

            Args:
                rng_key (n_chains, 2): random key
                position (n_chains, n_dim): current position
                log_prob (n_chains, ): log-probability of the current position
                data: data to be passed to the logpdf
                params: dictionary of parameters for the sampler

            Returns:
                position (n_chains, n_dim): the new poisiton of the chain
                log_prob (n_chains, ): the log-probability of the new position
                do_accept (n_chains, ): whether to accept the new position

            """
            key1, key2 = jax.random.split(rng_key)

            dt = params["step_size"]
            dt2 = dt * dt

            _, (proposal, logprob, d_logprob) = jax.lax.scan(
                body, (position, dt, data), jnp.array([key1, key1])
            )
            # print(jnp.dot(dt2, d_logprob[0]))
            ratio = logprob[1] - logprob[0]
            cov = jnp.diag(dt2)
            ratio -= multivariate_normal.logpdf(proposal[0], position + jnp.dot(dt2, d_logprob[0]) / 2, cov)
            ratio += multivariate_normal.logpdf(position, proposal[0] + jnp.dot(dt2, d_logprob[1]) / 2, cov)

            log_uniform = jnp.log(jax.random.uniform(key2))
            do_accept = log_uniform < ratio

            position = jnp.where(do_accept, proposal[0], position)
            log_prob = jnp.where(do_accept, logprob[1], logprob[0])
            return position, log_prob, do_accept

        return mala_kernel

    def make_update(self) -> Callable:
        """
        Make a MALA update function for multiple steps
        """
        if self.kernel is None:
            raise ValueError("Kernel not defined. Please run make_kernel first.")

        def mala_update(i, state):
            key, positions, log_p, acceptance, data, params = state
            _, key = jax.random.split(key)
            new_position, new_log_p, do_accept = self.kernel(
                key, positions[i - 1], log_p[i - 1], data, params
            )
            positions = positions.at[i].set(new_position)
            log_p = log_p.at[i].set(new_log_p)
            acceptance = acceptance.at[i].set(do_accept)
            return (key, positions, log_p, acceptance, data, params)

        return mala_update

    def make_sampler(self) -> Callable:
        """
        Make a MALA sampler for multiple chains given initial positions
        """

        if self.update is None:
            raise ValueError("Update function not defined. Please run make_update first.")

        def mala_sampler(rng_key, n_steps, initial_position, data, verbose):
            logp = self.logpdf_vmap(initial_position, data)
            n_chains = rng_key.shape[0]
            acceptance = jnp.zeros((n_chains, n_steps))
            all_positions = (jnp.zeros((n_chains, n_steps) + initial_position.shape[-1:])) + initial_position[
                :, None
            ]
            all_logp = jnp.zeros((n_chains, n_steps)) + logp[:, None]
            state = (rng_key, all_positions, all_logp, acceptance, data, self.params)
            if verbose:
                iterator_loop = tqdm(range(1, n_steps), desc="Sampling Locally", miniters=int(n_steps / 10))
            else:
                iterator_loop = range(1, n_steps)
            for i in iterator_loop:
                state = self.update_vmap(i, state)
            return state[:-2]

        self.sampler = mala_sampler
        return mala_sampler

@jit
def prior_eval(cube, _):
    """
    From a parameter cube, evaluates the associated
    prior probability.
    """
    return -0.5 * jnp.sum((cube - PRIOR_MEANS) ** 2 / PRIOR_SIGMAS**2)


@jit
def posterior_eval(cube, data_stacked):
    """
    Extracts the parameter cube and evaluates
    the associated likelihood.
    """
    # return -0.5 * jnp.linalg.norm((cube - PRIOR_MEANS - 1.)/PRIOR_SIGMAS)

    t, obsflux, uncertainties = data_stacked

    # return -50. * jnp.linalg.norm(cube - 100.)

    max_flux = np.max(obsflux - uncertainties)

    # return prior_eval(cube, max_flux)

    (
        A,
        beta,
        gamma,
        t0,
        tau_rise,
        tau_fall,
        extra_sigma,
        A_g,
        beta_g,
        gamma_g,
        t0_g,
        tau_rise_g,
        tau_fall_g,
        extra_sigma_g,
    ) = cube

    A = max_flux * 10**A
    gamma = 10**gamma
    tau_rise = 10**tau_rise
    tau_fall = 10**tau_fall
    extra_sigma = 10**extra_sigma

    phase = t - t0
    flux_const = A / (1.0 + jnp.exp(-phase / tau_rise))

    sigmoid = 1 / (1 + jnp.exp(10.0 * (gamma - phase)))

    # return -jnp.sum( (obsflux[:14] - flux_const)**2 )
    flux = flux_const * (
        (1 - sigmoid) * (1 - beta * phase)
        + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
    )

    inc_band_ix = np.arange(0, PAD_SIZE)

    A_b = A * A_g  # pylint: disable=unused-variable
    beta_b = beta * beta_g
    gamma_b = gamma * gamma_g
    t0_b = t0 * t0_g
    tau_rise_b = tau_rise * tau_rise_g
    tau_fall_b = tau_fall * tau_fall_g

    # g band
    phase_b = (t - t0_b)[inc_band_ix]
    flux_const_b = A / (1.0 + jnp.exp(-phase_b / tau_rise_b))
    sigmoid_b = 1 / (1 + jnp.exp(10.0 * (gamma_b - phase_b)))

    flux = flux.at[inc_band_ix].set(
        flux_const_b
        * (
            (1 - sigmoid_b) * (1 - beta_b * phase_b)
            + sigmoid_b * (1 - beta_b * gamma_b) * jnp.exp(-(phase_b - gamma_b) / tau_fall_b)
        )
    )

    sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)
    sigma_tot = sigma_tot.at[inc_band_ix].set(
        jnp.sqrt(uncertainties[inc_band_ix] ** 2 + extra_sigma_g**2 * extra_sigma**2)
    )

    return (
        prior_eval(cube, None)
        - 0.5 * jnp.sum((flux - obsflux) ** 2 / sigma_tot**2)
        - jnp.sum(jnp.log(jnp.sqrt(2 * jnp.pi) * sigma_tot))
    )


def run_flowMC(lightcurve, priors, n_chains=4, rseed=42):
    """
    Run flowMC on one light curve.
    """
    if lightcurve.times is None:
        return None

    data_stacked = jnp.array([lightcurve.times, lightcurve.fluxes, lightcurve.flux_errors])

    all_priors = priors.to_numpy().T
    rng_key_set = initialize_rng_keys(n_chains, seed=rseed)
    n_dim = len(all_priors.T)
    initial_position = jnp.tile(all_priors[2], (n_chains, 1))

    print(jax.value_and_grad(prior_eval)(initial_position[0], None))
    print(jax.value_and_grad(posterior_eval)(initial_position[0], data_stacked))

    model = MaskedCouplingRQSpline(n_dim, 4, [32, 32], 8, jax.random.PRNGKey(10))

    n_loop_training = 10
    n_loop_production = 10
    n_local_steps = 100
    n_global_steps = 100
    num_epochs = 1

    learning_rate = 0.005
    momentum = 0.9
    batch_size = 500

    sampler = MALA(posterior_eval, True, {"step_size": all_priors[3] / 100.0}, use_autotune=True)  # {"})
    nf_sampler = FlowSampler(
        n_dim=n_dim,
        rng_key_set=rng_key_set,
        local_sampler=sampler,
        data=data_stacked,
        nf_model=model,
        n_loop_training=n_loop_training,
        n_loop_production=n_loop_production,
        n_local_steps=n_local_steps,
        n_global_steps=n_global_steps,
        n_chains=n_chains,
        n_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        batch_size=batch_size,
        use_global=True,
    )
    nf_sampler.sample(initial_position, data_stacked)

    out_prod = nf_sampler.get_sampler_state()  # default training=False
    chains = np.reshape(out_prod["chains"][:, -100:], (100 * n_chains, n_dim))

    return chains

class FlowMCSampler(Sampler):
    """Sampling using FlowMC."""

    def __init__(self):
        pass

    def run_single_curve(
        self, lightcurve: Lightcurve, priors: MultibandPriors, rng_seed=None, **kwargs
    ) -> PosteriorSamples:
        lightcurve.pad_bands(priors.ordered_bands, PAD_SIZE)
        eq_wt_samples = run_flowMC(lightcurve, rseed=rng_seed, priors=priors)
        if eq_wt_samples is None:
            return None
        return PosteriorSamples(
            eq_wt_samples, name=lightcurve.name, sampling_method="flowMC", sn_class=lightcurve.sn_class
        )

    def run_multi_curve(self, lightcurves, priors, **kwargs) -> List[PosteriorSamples]:
        """Not yet implemented."""
        raise NotImplementedError
