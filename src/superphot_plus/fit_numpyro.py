"""This module provides functionality for running a dynesty importance 
nested sampling algorithm on given data files and returning a set of 
equally weighted posteriors (sets of fit parameters).
"""

import os

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.config import config
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO

from superphot_plus.plotting import (
    plot_posterior_hist,
    plot_sampling_lc_fit_numpyro,
    plot_sampling_trace_numpyro,
)
from numpyro.infer.initialization import init_to_sample, init_to_uniform

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.file_utils import get_posterior_filename

from .constants import *  # pylint: disable=wildcard-import
from .file_paths import FIT_PLOTS_FOLDER, FITS_DIR

config.update("jax_enable_x64", True)
numpyro.enable_x64()


def trunc_norm(low, high, loc, scale):
    """Provides keyword parameters to numpyro's TruncatedNormal.

    Parameters
    ----------
    low : float
        The lower bound of the truncated normal distribution.
    high : float
        The upper bound of the truncated normal distribution.
    loc : float
        The mean of the truncated normal distribution.
    scale : float
        The standard deviation of the truncated normal distribution.

    Returns
    -------
    numpyro.distributions.TruncatedDistribution
        A truncated normal distribution.
    """
    return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)


def run_mcmc(lc, sampler="NUTS", t0_lim=None, plot=False):
    """Runs dynesty importance nested sampling on data file to get set
    of equally weighted posteriors (sets of fit parameters).

    Parameters
    ----------
    lc : Lightcurve object
        The Lightcurve object on which to run MCMC
    sampler : str, optional
        The MCMC sampler to use. Defaults to "NUTS".
    t0_lim : float or None, optional
        Upper time limit for the data. If provided, only data points
        with time values less than or equal to t0_lim will be included.
        Defaults to None.
    plot : bool, optional
        If True, associated plots will be generated and saved. Defaults
        to False.

    Returns
    -------
    np.ndarray or None
        A set of equally weighted posteriors (sets of fit parameters) as
        a numpy array. If the data file does not contain any valid
        points, None is returned.

    """
    rng_key = random.PRNGKey(4)
    rng_key, rng_key_ = random.split(rng_key)  # pylint: disable=unused-variable

    ref_band_idx = 1  # red band # pylint: disable=unused-variable
    n_params = 14  # pylint: disable=unused-variable

    tdata = lc.times
    fdata = lc.fluxes
    ferrdata = lc.flux_errors
    bdata = np.where(lc.bands == "r", ref_band_idx, 1 - ref_band_idx)  # change to integers

    max_flux = np.max(fdata[PAD_SIZE:] - np.abs(ferrdata[PAD_SIZE:]))
    inc_band_ix = np.arange(0, PAD_SIZE)

    def jax_model(t=None, obsflux=None, uncertainties=None, max_flux=None, inc_band_ix=None):
        """JAX model for MCMC.

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
        inc_band_ix : array-like, optional
            Index values for the band. Defaults to None.
        """
        A = max_flux * 10 ** numpyro.sample("logA", trunc_norm(*PRIOR_A))
        beta = numpyro.sample("beta", trunc_norm(*PRIOR_BETA))
        gamma = 10 ** numpyro.sample("log_gamma", trunc_norm(*PRIOR_GAMMA))
        t0 = numpyro.sample("t0", trunc_norm(*PRIOR_T0))
        tau_rise = 10 ** numpyro.sample("log_tau_rise", trunc_norm(*PRIOR_TAU_RISE))
        tau_fall = 10 ** numpyro.sample("log_tau_fall", trunc_norm(*PRIOR_TAU_FALL))
        extra_sigma = 10 ** numpyro.sample("log_extra_sigma", trunc_norm(*PRIOR_EXTRA_SIGMA))

        A_g = numpyro.sample("A_g", trunc_norm(*PRIOR_A_g))
        beta_g = numpyro.sample("beta_g", trunc_norm(*PRIOR_BETA_g))
        gamma_g = numpyro.sample("gamma_g", trunc_norm(*PRIOR_GAMMA_g))
        t0_g = numpyro.sample("t0_g", trunc_norm(*PRIOR_T0_g))
        tau_rise_g = numpyro.sample("tau_rise_g", trunc_norm(*PRIOR_TAU_RISE_g))
        tau_fall_g = numpyro.sample("tau_fall_g", trunc_norm(*PRIOR_TAU_FALL_g))
        extra_sigma_g = numpyro.sample("extra_sigma_g", trunc_norm(*PRIOR_EXTRA_SIGMA_g))
        """
        A_g = numpyro.param("A_g", 1.)
        beta_g = numpyro.param("beta_g", 1.)
        gamma_g = numpyro.param("gamma_g", 1.)
        t0_g = numpyro.param("t0_g", 1.)
        tau_rise_g = numpyro.param("tau_rise_g", 1.)
        tau_fall_g = numpyro.param("tau_fall_g", 1.)
        extra_sigma_g = numpyro.param("extra_sigma_g", 1.)
        """
        A_b = A * A_g  # pylint: disable=unused-variable
        beta_b = beta * beta_g
        gamma_b = gamma * gamma_g
        t0_b = t0 * t0_g
        tau_rise_b = tau_rise * tau_rise_g
        tau_fall_b = tau_fall * tau_fall_g

        phase = t - t0
        flux_const = A / (1.0 + jnp.exp(-phase / tau_rise))
        sigmoid = 1 / (1 + jnp.exp(10.0 * (gamma - phase)))

        flux = flux_const * (
            (1 - sigmoid) * (1 - beta * phase)
            + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
        )

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

        obs = numpyro.sample(
            "obs", dist.Normal(flux, sigma_tot), obs=obsflux
        )  # pylint: disable=unused-variable

    def jax_guide(
        t=None,
        obsflux=None,
        uncertainties=None,
        max_flux=None,
        inc_band_ix=None,  # pylint: disable=unused-variable
    ):
        """JAX guide function for MCMC.

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
        inc_band_ix : array-like, optional
            Index values for the band. Defaults to None.
        """
        logA_mu = numpyro.param(
            "logA_mu",
            PRIOR_A[2],
            constraint=constraints.interval(PRIOR_A[0], PRIOR_A[1]),
        )
        logA_sigma = numpyro.param("logA_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("logA", dist.Normal(logA_mu, logA_sigma))

        beta_mu = numpyro.param(
            "beta_mu",
            PRIOR_BETA[2],
            constraint=constraints.interval(PRIOR_BETA[0], PRIOR_BETA[1]),
        )
        beta_sigma = numpyro.param("beta_sigma", 1e-5, constraint=constraints.positive)
        numpyro.sample("beta", dist.Normal(beta_mu, beta_sigma))

        log_gamma_mu = numpyro.param(
            "log_gamma_mu",
            PRIOR_GAMMA[2],
            constraint=constraints.interval(PRIOR_GAMMA[0], PRIOR_GAMMA[1]),
        )
        log_gamma_sigma = numpyro.param("log_gamma_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("log_gamma", dist.Normal(log_gamma_mu, log_gamma_sigma))

        t0_mu = numpyro.param(
            "t0_mu",
            PRIOR_T0[2],
            constraint=constraints.interval(PRIOR_T0[0], PRIOR_T0[1]),
        )
        t0_sigma = numpyro.param("t0_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("t0", dist.Normal(t0_mu, t0_sigma))

        log_tau_rise_mu = numpyro.param(
            "log_tau_rise_mu",
            PRIOR_TAU_RISE[2],
            constraint=constraints.interval(PRIOR_TAU_RISE[0], PRIOR_TAU_RISE[1]),
        )
        log_tau_rise_sigma = numpyro.param("log_tau_rise_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("log_tau_rise", dist.Normal(log_tau_rise_mu, log_tau_rise_sigma))

        log_tau_fall_mu = numpyro.param(
            "log_tau_fall_mu",
            PRIOR_TAU_FALL[2],
            constraint=constraints.interval(PRIOR_TAU_FALL[0], PRIOR_TAU_FALL[1]),
        )
        log_tau_fall_sigma = numpyro.param("log_tau_fall_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("log_tau_fall", dist.Normal(log_tau_fall_mu, log_tau_fall_sigma))

        log_extra_sigma_mu = numpyro.param(
            "log_extra_sigma_mu",
            PRIOR_EXTRA_SIGMA[2],
            constraint=constraints.interval(PRIOR_EXTRA_SIGMA[0], PRIOR_EXTRA_SIGMA[1]),
        )
        log_extra_sigma_sigma = numpyro.param("log_extra_sigma_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("log_extra_sigma", dist.Normal(log_extra_sigma_mu, log_extra_sigma_sigma))

        # aux bands

        Ag_mu = numpyro.param(
            "A_g_mu",
            PRIOR_A_g[2],
            constraint=constraints.interval(PRIOR_A_g[0], PRIOR_A_g[1]),
        )
        Ag_sigma = numpyro.param("A_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("A_g", dist.Normal(Ag_mu, Ag_sigma))

        beta_g_mu = numpyro.param(
            "beta_g_mu",
            PRIOR_BETA_g[2],
            constraint=constraints.interval(PRIOR_BETA_g[0], PRIOR_BETA_g[1]),
        )
        beta_g_sigma = numpyro.param("beta_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("beta_g", dist.Normal(beta_g_mu, beta_g_sigma))

        gamma_g_mu = numpyro.param(
            "gamma_g_mu",
            PRIOR_GAMMA_g[2],
            constraint=constraints.interval(PRIOR_GAMMA_g[0], PRIOR_GAMMA_g[1]),
        )
        gamma_g_sigma = numpyro.param("gamma_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("gamma_g", dist.Normal(gamma_g_mu, gamma_g_sigma))

        t0_g_mu = numpyro.param(
            "t0_g_mu",
            PRIOR_T0_g[2],
            constraint=constraints.interval(PRIOR_T0_g[0], PRIOR_T0_g[1]),
        )
        t0_g_sigma = numpyro.param("t0_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("t0_g", dist.Normal(t0_g_mu, t0_g_sigma))

        tau_rise_g_mu = numpyro.param(
            "tau_rise_g_mu",
            PRIOR_TAU_RISE_g[2],
            constraint=constraints.interval(PRIOR_TAU_RISE_g[0], PRIOR_TAU_RISE_g[1]),
        )
        tau_rise_g_sigma = numpyro.param("tau_rise_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("tau_rise_g", dist.Normal(tau_rise_g_mu, tau_rise_g_sigma))

        tau_fall_g_mu = numpyro.param(
            "tau_fall_g_mu",
            PRIOR_TAU_FALL_g[2],
            constraint=constraints.interval(PRIOR_TAU_FALL_g[0], PRIOR_TAU_FALL_g[1]),
        )
        tau_fall_g_sigma = numpyro.param("tau_fall_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("tau_fall_g", dist.Normal(tau_fall_g_mu, tau_fall_g_sigma))

        extra_sigma_g_mu = numpyro.param(
            "extra_sigma_g_mu",
            PRIOR_EXTRA_SIGMA_g[2],
            constraint=constraints.interval(PRIOR_EXTRA_SIGMA_g[0], PRIOR_EXTRA_SIGMA_g[1]),
        )
        extra_sigma_g_sigma = numpyro.param("extra_sigma_g_sigma", 1e-3, constraint=constraints.positive)
        numpyro.sample("extra_sigma_g", dist.Normal(extra_sigma_g_mu, extra_sigma_g_sigma))

    if sampler == "NUTS":
        num_samples = 300
        kernel = NUTS(jax_model, init_strategy=init_to_uniform)

        mcmc = MCMC(
            kernel,
            num_warmup=1000,
            num_samples=num_samples,
            num_chains=1,
            chain_method="parallel",
            jit_model_args=True,
        )

        # with numpyro.validation_enabled():
        res = mcmc.run(  # pylint: disable=unused-variable
            rng_key,
            obsflux=fdata,
            t=tdata,
            uncertainties=ferrdata,
            max_flux=max_flux,
            inc_band_ix=inc_band_ix,
        )

        # mcmc.print_summary()
        posterior_samples = mcmc.get_samples()

    elif sampler == "nested":
        num_samples = 300
        ns = NestedSampler(jax_model, constructor_kwargs=None)

        ns.run(
            random.PRNGKey(1),
            obsflux=fdata,
            t=tdata,
            uncertainties=ferrdata,
            max_flux=max_flux,
            inc_band_ix=inc_band_ix,
        )
        posterior_samples = ns.get_samples(random.PRNGKey(3), num_samples=num_samples)

    elif sampler == "svi":
        optimizer = numpyro.optim.Adam(step_size=0.001)
        svi = SVI(jax_model, jax_guide, optimizer, loss=Trace_ELBO())
        num_iter = 10000
        with numpyro.validation_enabled():
            svi_result = svi.run(
                random.PRNGKey(1),
                num_iter,
                stable_update=True,
                obsflux=fdata,
                t=tdata,
                uncertainties=ferrdata,
                max_flux=max_flux,
                inc_band_ix=inc_band_ix,
            )
        params = svi_result.params
        posterior_samples = {}
        for p in params:
            if p[-2:] == "mu":
                posterior_samples[p[:-3]] = np.random.normal(
                    loc=params[p], scale=params[p[:-2] + "sigma"], size=100
                )

    else:
        return None

    """
    predictive = Predictive(jax_model, posterior_samples, infer_discrete=False)
    
    discrete_samples = predictive(random.PRNGKey(1), 
                       t=tdata_stacked, 
                       uncertainties=ferrdata_stacked, 
                       max_flux=max_flux, 
                       inc_band_ix=inc_band_ix)
    
    print(discrete_samples.keys())
    """
    if plot:  # pragma: no cover
        plot_posterior_hist(posterior_samples, parameter="log_tau_fall")
        plot_sampling_lc_fit_numpyro(
            posterior_samples,
            tdata=[tdata],
            fdata=[fdata],
            ferrdata=[ferrdata],
            bdata=[bdata],
            max_flux=[max_flux],
            lcs=[lc],
            t0_lim=t0_lim,
        )
        plot_sampling_trace_numpyro(posterior_samples)

    param_list = [
        "logA",
        "beta",
        "log_gamma",
        "t0",
        "log_tau_rise",
        "log_tau_fall",
        "log_extra_sigma",
        "A_g",
        "beta_g",
        "gamma_g",
        "t0_g",
        "tau_rise_g",
        "tau_fall_g",
        "extra_sigma_g",
    ]

    post_reformatted_for_save = []
    for p in param_list:
        if p == "logA":
            post_reformatted_for_save.append(max_flux * 10 ** posterior_samples[p])
        elif p[:3] == "log":
            post_reformatted_for_save.append(10 ** posterior_samples[p])
        else:
            post_reformatted_for_save.append(posterior_samples[p])

    return np.array(post_reformatted_for_save).T


def run_mcmc_batch(lcs, t0_lim=None, plot=False):
    """Runs numpyro's NUTS sampler on data file to get a set of equally
    weighted posteriors (sets of fit parameters).

    Parameters
    ----------
    lcs : list of Lightcurves
        A list of Lightcurve objects
    t0_lim : float or None, optional
        Upper time limit for the data. Defaults to None.
    plot : bool, optional
        Flag for generating and saving assosciated plots. Defaults to
        False.
    """
    rng_key = random.PRNGKey(4)
    rng_key, rng_key_ = random.split(rng_key)  # pylint: disable=unused-variable

    ref_band_idx = 1  # red band # pylint: disable=unused-variable
    n_params = 14  # pylint: disable=unused-variable

    tdata_stacked = []
    fdata_stacked = []
    ferrdata_stacked = []
    bdata_stacked = []

    for lc in lcs:
        tdata_stacked.append(lc.times)
        fdata_stacked.append(lc.fluxes)
        ferrdata_stacked.append(lc.flux_errors)
        bdata_stacked.append(np.where(lc.bands == "r", ref_band_idx, 1 - ref_band_idx))  # change to integers

    tdata_stacked = np.array(tdata_stacked)
    fdata_stacked = np.array(fdata_stacked)
    ferrdata_stacked = np.array(ferrdata_stacked)
    bdata_stacked = np.array(bdata_stacked)

    max_flux = np.max(fdata_stacked[:, PAD_SIZE:] - np.abs(ferrdata_stacked[:, PAD_SIZE:]), axis=1)

    inc_band_ix = np.arange(0, PAD_SIZE)

    N = len(tdata_stacked)
    print(N)

    def jax_model(t=None, obsflux=None, uncertainties=None, max_flux=None, inc_band_ix=None):
        """JAX model for MCMC.

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
        inc_band_ix : array-like, optional
            Index values for the band. Defaults to None.
        """
        with numpyro.plate("components", N) as sn_index:  # pylint: disable=unused-variable
            A = max_flux * 10 ** numpyro.sample("logA", trunc_norm(*PRIOR_A))
            beta = numpyro.sample("beta", trunc_norm(*PRIOR_BETA))
            gamma = 10 ** numpyro.sample("log_gamma", trunc_norm(*PRIOR_GAMMA))
            t0 = numpyro.sample("t0", trunc_norm(-100.0, 200.0, 0.0, 20.0))
            tau_rise = 10 ** numpyro.sample("log_tau_rise", trunc_norm(*PRIOR_TAU_RISE))
            tau_fall = 10 ** numpyro.sample("log_tau_fall", trunc_norm(*PRIOR_TAU_FALL))
            extra_sigma = 10 ** numpyro.sample("log_extra_sigma", trunc_norm(*PRIOR_EXTRA_SIGMA))

            A_g = numpyro.sample("A_g", trunc_norm(*PRIOR_A_g))
            beta_g = numpyro.sample("beta_g", trunc_norm(*PRIOR_BETA_g))
            gamma_g = numpyro.sample("gamma_g", trunc_norm(*PRIOR_GAMMA_g))
            t0_g = numpyro.sample("t0_g", trunc_norm(*PRIOR_T0_g))
            tau_rise_g = numpyro.sample("tau_rise_g", trunc_norm(*PRIOR_TAU_RISE_g))
            tau_fall_g = numpyro.sample("tau_fall_g", trunc_norm(*PRIOR_TAU_FALL_g))
            extra_sigma_g = numpyro.sample("extra_sigma_g", trunc_norm(*PRIOR_EXTRA_SIGMA_g))

        A_b = A * A_g  # pylint: disable=unused-variable
        beta_b = beta * beta_g
        gamma_b = gamma * gamma_g
        t0_b = t0 * t0_g
        tau_rise_b = tau_rise * tau_rise_g
        tau_fall_b = tau_fall * tau_fall_g

        phase = t - t0[:, np.newaxis]
        flux_const = A[:, np.newaxis] / (1.0 + jnp.exp(-phase / tau_rise[:, np.newaxis]))
        sigmoid = 1 / (1 + jnp.exp(10.0 * (gamma[:, np.newaxis] - phase)))

        flux = flux_const * (
            (1 - sigmoid) * (1 - beta[:, np.newaxis] * phase)
            + sigmoid
            * (1 - beta[:, np.newaxis] * gamma[:, np.newaxis])
            * jnp.exp(-(phase - gamma[:, np.newaxis]) / tau_fall[:, np.newaxis])
        )

        # g band
        phase_b = (t - t0_b[:, np.newaxis])[:, inc_band_ix]
        flux_const_b = A[:, np.newaxis] / (1.0 + jnp.exp(-phase_b / tau_rise_b[:, np.newaxis]))
        sigmoid_b = 1 / (1 + jnp.exp(10.0 * (gamma_b[:, np.newaxis] - phase_b)))

        flux = flux.at[:, inc_band_ix].set(
            flux_const_b
            * (
                (1 - sigmoid_b) * (1 - beta_b[:, np.newaxis] * phase_b)
                + sigmoid_b
                * (1 - beta_b[:, np.newaxis] * gamma_b[:, np.newaxis])
                * jnp.exp(-(phase_b - gamma_b[:, np.newaxis]) / tau_fall_b[:, np.newaxis])
            )
        )

        sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma[:, np.newaxis] ** 2)
        sigma_tot = sigma_tot.at[:, inc_band_ix].set(
            jnp.sqrt(
                uncertainties[:, inc_band_ix] ** 2
                + extra_sigma_g[:, np.newaxis] ** 2 * extra_sigma[:, np.newaxis] ** 2
            )
        )

        obs = numpyro.sample(
            "obs", dist.Normal(flux, sigma_tot), obs=obsflux
        )  # pylint: disable=unused-variable

    kernel = NUTS(jax_model, init_strategy=init_to_sample)
    num_samples = 100
    mcmc = MCMC(
        kernel,
        num_warmup=100,
        num_samples=num_samples,
        num_chains=1,
        chain_method="parallel",
    )  # jit_model_args=True)

    # with numpyro.validation_enabled():
    res = mcmc.run(  # pylint: disable=unused-variable
        rng_key,
        obsflux=fdata_stacked,
        t=tdata_stacked,
        uncertainties=ferrdata_stacked,
        max_flux=max_flux,
        inc_band_ix=inc_band_ix,
    )

    # mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    """
    predictive = Predictive(jax_model, posterior_samples, infer_discrete=False)
    
    discrete_samples = predictive(
        random.PRNGKey(1),
        t=tdata_stacked,
        uncertainties=ferrdata_stacked,
        max_flux=max_flux,
        inc_band_ix=inc_band_ix,
    )
    
    print(discrete_samples.keys())
    """

    if plot:  # pragma: no cover
        plot_posterior_hist(posterior_samples, parameter="log_tau_fall")

        plot_sampling_lc_fit_numpyro(
            posterior_samples,
            tdata=tdata_stacked,
            fdata=fdata_stacked,
            ferrdata=ferrdata_stacked,
            bdata=bdata_stacked,
            max_flux=max_flux,
            lcs=lcs,
            t0_lim=t0_lim,
        )

        plot_sampling_trace_numpyro(posterior_samples)

    return posterior_samples


def main_loop_directory(test_filenames, output_dir=FITS_DIR):
    """Runs MCMC on given filenames and saves results.

    Parameters
    ----------
    test_filenames : list of str
        Names of files to use as input.
    output_dir : str
        Directory to save outputs to. Defaults to FITS_DIR.
    """
    os.makedirs(output_dir, exist_ok=True)

    lcs = []
    for filename in test_filenames:
        lc = Lightcurve.from_file(filename)
        lc.pad_bands(["g", "r"], PAD_SIZE)
        lcs.append(lc)

    eq_samples = run_mcmc_batch(lcs, plot=True)
    if eq_samples is None:
        return None

    print(np.mean(eq_samples["log_tau_fall"]))

    return None


def numpyro_single_curve(lc, output_dir=FITS_DIR, sampler="svi"):
    """Perform model fitting using dynesty on a single light curve.

    This function runs the dynesty importance nested sampling algorithm
    on a single light curve. It saves the resulting equally weighted
    posterior samples to a compressed NumPy archive file.

    Parameters
    ----------
    lc : Lightcurve object
        The light curve of interest.
    output_dir : str
        Directory to save outputs to. Defaults to FITS_DIR.
    sampler : str
        The MCMC sampler to use. Defaults to "svi".

    Returns
    -------
    sample_mean: numpy array
        Return the mean of the MCMC samples or None if the fitting is
        skipped or encounters an error.
    """
    if lc.name is None or lc.name == "":  # pragma: no cover
        raise ValueError("Empty light curve name.")

    os.makedirs(output_dir, exist_ok=True)

    eq_samples = run_mcmc(lc, sampler=sampler, plot=False)
    if eq_samples is None:  # pragma: no cover
        return None

    posterior_filename = get_posterior_filename(lc.name, output_dir, sampler)
    np.savez_compressed(posterior_filename, eq_samples)
    sample_mean = np.mean(eq_samples, axis=0)
    return sample_mean


def numpyro_single_file(test_filename, output_dir=FITS_DIR, sampler="svi"):
    """Runs MCMC on a single file.

    Parameters
    ----------
    test_filename : str
        Name of the file to use as input.
    output_dir : str
        Directory to save outputs to. Defaults to FITS_DIR.
    sampler : str
        The MCMC sampler to use. Defaults to "svi".

    Returns
    -------
    sample_mean: numpy array
        Return the mean of the MCMC samples or None if the fitting is
        skipped or encounters an error.
    """
    lc = Lightcurve.from_file(test_filename)
    lc.pad_bands(["g", "r"], PAD_SIZE)

    sample_mean = numpyro_single_curve(lc, output_dir, sampler)
    return sample_mean
