"""MCMC sampling using numpyro."""

from os import urandom
from typing import List

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random, lax, jit
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
from superphot_plus.utils import (
    calculate_chi_squareds,
    get_numpyro_cube,
    villar_fit_constraint
)
#from numpyro.distributions.constraints import Constraint


config.update("jax_enable_x64", True)
numpyro.enable_x64()
numpyro.set_host_device_count(4)


class NumpyroSampler(Sampler):
    """MCMC sampling using numpyro."""

    def __init__(self, sampler='svi'):
        self.sampler = sampler

    def run_single_curve(
        self, lightcurve: Lightcurve, priors: MultibandPriors, rng_seed, ref_params=None, **kwargs
    ) -> PosteriorSamples:
        """Run the sampler on a single light curve.

        Parameters
        ----------
        lightcurve : Lightcurve
            The lightcurve to sample.
        priors : MultibandPriors
            The curve priors to use.
        rng_seed : int or None
            The random seed to use (for testing purposes). The user should pass None in
            cases where they want a fully random run.
        sampler : str
            The numpyro sampler to use. Either "NUTS" or "svi"

        Returns
        -------
        eq_wt_samples : PosteriorSamples
            The resulting samples.
        """
        lightcurve = lightcurve.pad_bands(priors.ordered_bands, PAD_SIZE, in_place=False)
        eq_wt_samples = run_mcmc(
            lightcurve,
            rng_seed=rng_seed,
            sampler=self.sampler,
            priors=priors,
            ref_params=ref_params,
        )
        if eq_wt_samples is None:
            return None
        
        return PosteriorSamples(
            eq_wt_samples[0],
            name=lightcurve.name,
            sampling_method=self.sampler,
            sn_class=lightcurve.sn_class,
        )

    def run_multi_curve(
        self, lightcurves, priors: MultibandPriors, rng_seed, sampler="svi", ref_params=None, **kwargs
    ) -> List[PosteriorSamples]:
        """Not yet implemented."""

        if len(lightcurves) == 0:
            return []

        padded_lcs = []
        for lc in lightcurves:
            padded_lcs.append(lc.pad_bands(priors.ordered_bands, PAD_SIZE, in_place=False))

        eq_wt_samples = run_mcmc(
            padded_lcs,
            rng_seed=rng_seed,
            sampler=sampler,
            priors=priors,
            ref_params=ref_params
        )

        post_list = []
        for i, posts in enumerate(eq_wt_samples):
            if posts is None:
                continue
            post_list.append(
                PosteriorSamples(
                    posts, name=lightcurves[i].name, sampling_method=sampler, sn_class=lightcurves[i].sn_class
                )
            )

        return post_list


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


def _svi_helper_no_recompile(
    lc_single,
    max_flux,
    priors,
    svi,
    svi_state,
    lax_jit,
    num_iter,
    seed,
    ref_params=None,
):
    """Helper function to run SVI on a single light curve with an already
    compiled SVI sampler object."""
    if svi_state is None:
        svi_state = svi.init(
            random.PRNGKey(1),
            obsflux=lc_single.fluxes,
            t=lc_single.times,
            uncertainties=lc_single.flux_errors,
            max_flux=max_flux,
            ref_params=ref_params,
        )
    
    svi_state = lax_jit(
        svi,
        svi_state,
        num_iter,
        obsflux=lc_single.fluxes,
        t=lc_single.times,
        uncertainties=lc_single.flux_errors,
        max_flux=max_flux,
        ref_params=ref_params,
    )

    # params = svi_result.params
    params = svi.get_params(svi_state)
    posterior_samples = {}
    for param in params:
        if param[-2:] == "mu":
            rng = np.random.RandomState(seed[0])
            posterior_samples[param[:-3]] = rng.normal(
                loc=params[param], scale=params[param[:-2] + "sigma"], size=100
            )
    
    
    posterior_cube = get_numpyro_cube(
        posterior_samples, max_flux,
        priors.reference_band, priors.ordered_bands
    )[0]
    
    padded_idxs = lc_single.flux_errors == 1e10
        
    red_chisq = calculate_chi_squareds(
        posterior_cube,
        lc_single.times[~padded_idxs],
        lc_single.fluxes[~padded_idxs],
        lc_single.flux_errors[~padded_idxs],
        lc_single.bands[~padded_idxs],
        max_flux,
        ordered_bands=priors.ordered_bands,
        ref_band=priors.reference_band,
    )
    
    return posterior_cube, red_chisq, svi_state
        
        
def run_mcmc(lc, rng_seed, sampler="NUTS", priors=Survey.ZTF().priors, ref_params=None):
    """Runs MCMC using numpyro on the lightcurve to get set
    of equally weighted posteriors (sets of fit parameters).

    Parameters
    ----------
    lc : Lightcurve object
        The Lightcurve object on which to run MCMC
    rng_seed : int or None
        The random seed to use (for testing purposes). The user should pass None in
        cases where they want a fully random run.
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

    batch = type(lc) is list  # check if one LightCurve or multiple

    if rng_seed is None:
        rng_seed = int.from_bytes(urandom(4), "big")
    print(f"Running numpyro with seed={rng_seed}")

    rng_key = random.PRNGKey(rng_seed)
    rng_key, seed2 = random.split(rng_key)
        

    def jax_model(t=None, obsflux=None, uncertainties=None, max_flux=None, ref_params=None):
        create_jax_model(priors, t, obsflux, uncertainties, max_flux, ref_params)

    def jax_guide(**kwargs):  # pylint: disable=unused-argument
        create_jax_guide(priors)

    if sampler == "NUTS":
        if batch:
            raise ValueError("Batch mode not implemented for NUTS.")

        # Require data in all bands.
        for unique_band in priors.ordered_bands:
            if lc.obs_count(unique_band) == 0:
                return None

        max_flux, _ = lc.find_max_flux(band=priors.reference_band)

        num_samples = 300
        kernel = NUTS(jax_model, init_strategy=init_to_uniform)

        rng_key = random.PRNGKey(rng_seed)
        rng_key, _ = random.split(rng_key)

        mcmc = MCMC(
            kernel,
            num_warmup=10000,
            num_samples=num_samples,
            num_chains=4,
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
        
        posterior_cube, aux_bands = get_numpyro_cube(
            posterior_samples,
            max_flux,
            priors.reference_band,
            priors.ordered_bands
        )
        
        padded_idxs = lc.flux_errors == 1e10
        
        red_chisq = calculate_chi_squareds(
            posterior_cube,
            lc.times[~padded_idxs],
            lc.fluxes[~padded_idxs],
            lc.flux_errors[~padded_idxs],
            lc.bands[~padded_idxs],
            max_flux,
            ordered_bands=priors.ordered_bands,
            ref_band=priors.reference_band,
        )

        posterior_cubes = [
            np.hstack((posterior_cube, red_chisq[np.newaxis, :].T)),
        ]

    elif sampler == "svi":
        optimizer = numpyro.optim.Adam(step_size=0.001)
        svi = SVI(jax_model, jax_guide, optimizer, loss=Trace_ELBO())

        num_iter = 10_000
        lax_jit = jit(lax_helper_function, static_argnums=(0, 2))

        if not batch:
            lc = [
                lc,
            ]

        bad_prev_fit = True
        posterior_cubes = []
        for i, lc_single in enumerate(lc):
            if i % 100 == 0:
                print(i)

            """
            # Require data in all bands.
            for unique_band in priors.ordered_bands:
                if lc_single.obs_count(unique_band) == 0:
                    posterior_cubes.append(None)
                    break
            """

            if bad_prev_fit:
                svi_state = None #reinitialize
            
            max_flux, _ = lc_single.find_max_flux(band=priors.reference_band)
            posterior_cube, red_chisq, svi_state = _svi_helper_no_recompile(
                lc_single,
                max_flux,
                priors,
                svi,
                svi_state,
                lax_jit,
                num_iter,
                seed2,
                ref_params,
            )

            #bad_prev_fit = np.mean(red_chisq) 

            posterior_cube = np.hstack((posterior_cube, red_chisq[np.newaxis, :].T))
            posterior_cubes.append(posterior_cube)

    else:
        raise ValueError("'sampler' must be 'NUTS' or 'svi'")

    return posterior_cubes
