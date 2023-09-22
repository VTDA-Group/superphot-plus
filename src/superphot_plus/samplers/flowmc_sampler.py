"""Sampling using FlowMC MALA """

from functools import partialmethod
from typing import Callable, List

import jax
import jax.config as config
import jax.numpy as jnp
import numpy as np
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.MALA import MALA
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

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

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
    if data_stacked.shape[0] != 3:
        return -50. * jnp.linalg.norm(cube - 100.)

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
    print("n_dim", n_dim)
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

    sampler = MALA(posterior_eval, True, {"step_size": all_priors[3] / 100.0})#, use_autotune=True)  # {"})
    nf_sampler = FlowSampler(
        n_dim=n_dim,
        rng_key_set=rng_key_set,
        local_sampler=sampler,
        # data = jnp.zeros(n_dim),
        # data=lightcurve.times,
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