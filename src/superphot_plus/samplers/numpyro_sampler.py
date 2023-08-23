"""MCMC sampling using numpyro."""

from typing import List

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.config import config
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_uniform

from superphot_plus.constants import PAD_SIZE
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.samplers.sampler import Sampler
from superphot_plus.surveys.fitting_priors import MultibandPriors, PriorFields
from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import calculate_neg_chi_squareds, get_numpyro_cube

config.update("jax_enable_x64", True)
numpyro.enable_x64()


class NumpyroSampler(Sampler):
    """MCMC sampling using numpyro."""

    def __init__(self):
        pass

    def run_single_curve(
        self, lightcurve: Lightcurve, priors: MultibandPriors, sampler="svi", **kwargs
    ) -> PosteriorSamples:
        lightcurve.pad_bands(priors.ordered_bands, PAD_SIZE)
        eq_wt_samples = run_mcmc(lightcurve, sampler=sampler, priors=priors)
        if eq_wt_samples is None:
            return None
        return PosteriorSamples(
            eq_wt_samples, name=lightcurve.name, sampling_method=sampler, sn_class=lightcurve.sn_class
        )

    def run_multi_curve(self, lightcurves, priors, **kwargs) -> List[PosteriorSamples]:
        """Not yet implemented."""
        raise NotImplementedError


def prior_helper(priors, max_flux, aux_b=None):
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
        amp = max_flux * 10 ** numpyro.sample("logA", trunc_norm(priors.amp))
        beta = numpyro.sample("beta", trunc_norm(priors.beta))
        gamma = 10 ** numpyro.sample("log_gamma", trunc_norm(priors.gamma))
        t_0 = numpyro.sample("t0", trunc_norm(priors.t_0))
        tau_rise = 10 ** numpyro.sample("log_tau_rise", trunc_norm(priors.tau_rise))
        tau_fall = 10 ** numpyro.sample("log_tau_fall", trunc_norm(priors.tau_fall))
        extra_sigma = 10 ** numpyro.sample("log_extra_sigma", trunc_norm(priors.extra_sigma))

    else:
        suffix = "_" + str(aux_b)
        amp = numpyro.sample(f"A{suffix}", trunc_norm(priors.amp))
        beta = numpyro.sample(f"beta{suffix}", trunc_norm(priors.beta))
        gamma = numpyro.sample(f"gamma{suffix}", trunc_norm(priors.gamma))
        t_0 = numpyro.sample(f"t0{suffix}", trunc_norm(priors.t_0))
        tau_rise = numpyro.sample(f"tau_rise{suffix}", trunc_norm(priors.tau_rise))
        tau_fall = numpyro.sample(f"tau_fall{suffix}", trunc_norm(priors.tau_fall))
        extra_sigma = numpyro.sample(f"extra_sigma{suffix}", trunc_norm(priors.extra_sigma))

    return amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma


def trunc_norm(fields: PriorFields):
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
        loc=fields.mean, scale=fields.std, low=fields.clip_a, high=fields.clip_b, validate_args=True
    )


def create_jax_model(
    priors, t=None, obsflux=None, uncertainties=None, max_flux=None
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

    amp, beta, gamma, t_0, tau_rise, tau_fall, extra_sigma = prior_helper(ref_priors, max_flux)

    phase = t - t_0
    flux_const = amp / (1.0 + jnp.exp(-phase / tau_rise))
    sigmoid = 1 / (1 + jnp.exp(10.0 * (gamma - phase)))

    flux = flux_const * (
        (1 - sigmoid) * (1 - beta * phase)
        + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
    )
    sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)

    # auxiliary bands
    for b_idx, uniq_b in enumerate(priors.aux_bands):
        b_priors = priors.bands[uniq_b]

        (
            amp_ratio,
            beta_ratio,
            gamma_ratio,
            t0_ratio,
            tau_rise_ratio,
            tau_fall_ratio,
            extra_sigma_ratio,
        ) = prior_helper(b_priors, max_flux, uniq_b)

        amp_b = amp * amp_ratio
        beta_b = beta * beta_ratio
        gamma_b = gamma * gamma_ratio
        t0_b = t_0 * t0_ratio
        tau_rise_b = tau_rise * tau_rise_ratio
        tau_fall_b = tau_fall * tau_fall_ratio

        inc_band_ix = np.arange(b_idx * PAD_SIZE, (b_idx + 1) * PAD_SIZE)

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
            jnp.sqrt(uncertainties[inc_band_ix] ** 2 + extra_sigma_ratio**2 * extra_sigma**2)
        )

    _ = numpyro.sample("obs", dist.Normal(flux, sigma_tot), obs=obsflux)


def create_jax_guide(priors):
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


def run_mcmc(lc, sampler="NUTS", priors=Survey.ZTF().priors):
    """Runs MCMC using numpyro on the lightcurve to get set
    of equally weighted posteriors (sets of fit parameters).

    Parameters
    ----------
    lc : Lightcurve object
        The Lightcurve object on which to run MCMC
    sampler : str, optional
        The MCMC sampler to use. Defaults to "NUTS".
    priors : MultibandPriors, optional
        The prior set to use for fitting. Defaults to ZTF's priors.

    Returns
    -------
    np.ndarray or None
        A set of equally weighted posteriors (sets of fit parameters) as
        a numpy array. If the lightcurve does not contain any valid
        points, None is returned.
    """
    # Require data in all bands.
    for unique_band in priors.ordered_bands:
        if lc.obs_count(unique_band) == 0:
            return None

    def jax_model(t=None, obsflux=None, uncertainties=None, max_flux=None):
        create_jax_model(priors, t, obsflux, uncertainties, max_flux)

    def jax_guide(**kwargs):  # pylint: disable=unused-argument
        create_jax_guide(priors)

    max_flux, _ = lc.find_max_flux(band=priors.reference_band)

    if sampler == "NUTS":
        num_samples = 300
        kernel = NUTS(jax_model, init_strategy=init_to_uniform)

        rng_key = random.PRNGKey(4)
        rng_key, _ = random.split(rng_key)

        mcmc = MCMC(
            kernel,
            num_warmup=1000,
            num_samples=num_samples,
            num_chains=1,
            chain_method="parallel",
            jit_model_args=True,
        )

        # with numpyro.validation_enabled():
        mcmc.run(
            rng_key,
            obsflux=lc.fluxes,
            t=lc.times,
            uncertainties=lc.flux_errors,
            max_flux=max_flux,
        )

        posterior_samples = mcmc.get_samples()

    elif sampler == "svi":
        optimizer = numpyro.optim.Adam(step_size=0.001)
        svi = SVI(jax_model, jax_guide, optimizer, loss=Trace_ELBO())
        num_iter = 10000
        with numpyro.validation_enabled():
            svi_result = svi.run(
                random.PRNGKey(1),
                num_iter,
                stable_update=True,
                obsflux=lc.fluxes,
                t=lc.times,
                uncertainties=lc.flux_errors,
                max_flux=max_flux,
            )
        params = svi_result.params
        posterior_samples = {}
        for param in params:
            if param[-2:] == "mu":
                posterior_samples[param[:-3]] = np.random.normal(
                    loc=params[param], scale=params[param[:-2] + "sigma"], size=100
                )

    else:
        raise ValueError("'sampler' must be 'NUTS' or 'svi'")

    posterior_cube, aux_bands = get_numpyro_cube(posterior_samples, max_flux, priors.aux_bands)

    padded_idxs = lc.flux_errors > 1e5
    red_neg_chisq = calculate_neg_chi_squareds(
        posterior_cube,
        lc.times[~padded_idxs],
        lc.fluxes[~padded_idxs],
        lc.flux_errors[~padded_idxs],
        lc.bands[~padded_idxs],
        ordered_bands=priors.ordered_bands,
        ref_band=priors.reference_band,
    )

    posterior_cube = np.hstack((posterior_cube, red_neg_chisq[np.newaxis, :].T))
    return posterior_cube
