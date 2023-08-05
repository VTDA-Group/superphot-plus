"""This module provides functionality for running a dynesty importance 
nested sampling algorithm on given data files and returning a set of 
equally weighted posteriors (sets of fit parameters).
"""

import os

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.config import config
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_sample, init_to_uniform


from superphot_plus.constants import PAD_SIZE
from superphot_plus.file_paths import FITS_DIR
from superphot_plus.file_utils import get_posterior_filename
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.plotting import (
    plot_posterior_hist,
    plot_sampling_lc_fit_numpyro,
    plot_sampling_trace_numpyro,
)
from superphot_plus.priors.fitting_priors import MultibandPriors, PriorFields
from superphot_plus.surveys import Survey


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
    return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high, validate_args=True)


def trunc_norm_fields(fields: PriorFields):
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
    return dist.TruncatedNormal(loc=fields.mean, scale=fields.std, low=fields.clip_a, high=fields.clip_b)


def run_mcmc(lc, sampler="NUTS", priors=MultibandPriors.load_ztf_priors(), t0_lim=None, plot=False):
    """Runs dynesty importance nested sampling on data file to get set
    of equally weighted posteriors (sets of fit parameters).

    Parameters
    ----------
    lc : Lightcurve object
        The Lightcurve object on which to run MCMC
    sampler : str, optional
        The MCMC sampler to use. Defaults to "NUTS".
    priors : MultibandPriors, optional
        The prior set to use for fitting. Defaults to ZTF's priors.
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

    all_priors = priors.to_numpy().T
    ref_band = priors.reference_band

    n_params = len(all_priors.T)
    unique_bands = priors.ordered_bands
    ref_band_idx = np.argmax(unique_bands == ref_band)

    # Require data in both the g and r bands.
    for ub in unique_bands:
        if lc.obs_count(ub) == 0:
            return None

    tdata = lc.times
    fdata = lc.fluxes
    ferrdata = lc.flux_errors

    max_flux, max_flux_time = lc.find_max_flux(band=ref_band)
    bdata = lc.band_as_int(priors.ordered_bands)  # change to integers

    def jax_model(t=None, obsflux=None, uncertainties=None, max_flux=None):
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
        """
        ref_priors = priors.bands[ref_band]
        A = max_flux * 10 ** numpyro.sample("logA", trunc_norm_fields(ref_priors.amp))
        beta = numpyro.sample("beta", trunc_norm_fields(ref_priors.beta))
        gamma = 10 ** numpyro.sample("log_gamma", trunc_norm_fields(ref_priors.gamma))
        t0 = numpyro.sample("t0", trunc_norm_fields(ref_priors.t_0))
        tau_rise = 10 ** numpyro.sample("log_tau_rise", trunc_norm_fields(ref_priors.tau_rise))
        tau_fall = 10 ** numpyro.sample("log_tau_fall", trunc_norm_fields(ref_priors.tau_fall))
        extra_sigma = 10 ** numpyro.sample("log_extra_sigma", trunc_norm_fields(ref_priors.extra_sigma))

        phase = t - t0
        flux_const = A / (1.0 + jnp.exp(-phase / tau_rise))
        sigmoid = 1 / (1 + jnp.exp(10.0 * (gamma - phase)))

        flux = flux_const * (
            (1 - sigmoid) * (1 - beta * phase)
            + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
        )
        sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)

        # auxiliary bands
        for b_idx, uniq_b in enumerate(unique_bands):
            if uniq_b == ref_band:
                continue

            b_priors = priors.bands[uniq_b]
            A_ratio = numpyro.sample("A_" + uniq_b, trunc_norm_fields(b_priors.amp))
            beta_ratio = numpyro.sample("beta_" + uniq_b, trunc_norm_fields(b_priors.beta))
            gamma_ratio = numpyro.sample("gamma_" + uniq_b, trunc_norm_fields(b_priors.gamma))
            t0_ratio = numpyro.sample("t0_" + uniq_b, trunc_norm_fields(b_priors.t_0))
            tau_rise_ratio = numpyro.sample("tau_rise_" + uniq_b, trunc_norm_fields(b_priors.tau_rise))
            tau_fall_ratio = numpyro.sample("tau_fall_" + uniq_b, trunc_norm_fields(b_priors.tau_fall))
            extra_sigma_ratio = numpyro.sample(
                "extra_sigma_" + uniq_b, trunc_norm_fields(b_priors.extra_sigma)
            )

            A_b = A * A_ratio  # pylint: disable=unused-variable
            beta_b = beta * beta_ratio
            gamma_b = gamma * gamma_ratio
            t0_b = t0 * t0_ratio
            tau_rise_b = tau_rise * tau_rise_ratio
            tau_fall_b = tau_fall * tau_fall_ratio

            inc_band_ix = np.arange(b_idx * PAD_SIZE, (b_idx + 1) * PAD_SIZE)

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

            sigma_tot = sigma_tot.at[inc_band_ix].set(
                jnp.sqrt(uncertainties[inc_band_ix] ** 2 + extra_sigma_ratio**2 * extra_sigma**2)
            )

        obs = numpyro.sample(
            "obs", dist.Normal(flux, sigma_tot), obs=obsflux
        )  # pylint: disable=unused-variable

    def jax_guide(
        t=None,
        obsflux=None,
        uncertainties=None,
        max_flux=None,
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
        """

        def numpyro_sample(prefix: str, fields: PriorFields, param_constraint: float):
            param_mu = numpyro.param(
                f"{prefix}_mu",
                fields.mean,
                constraint=constraints.interval(fields.clip_a, fields.clip_b),
            )
            param_sigma = numpyro.param(f"{prefix}_sigma", param_constraint, constraint=constraints.positive)
            numpyro.sample(prefix, dist.Normal(param_mu, param_sigma))

        ref_priors = priors.bands[ref_band]
        numpyro_sample("logA", ref_priors.amp, 1e-3)
        numpyro_sample("beta", ref_priors.beta, 1e-5)
        numpyro_sample("log_gamma", ref_priors.gamma, 1e-3)
        numpyro_sample("t0", ref_priors.t_0, 1e-3)
        numpyro_sample("log_tau_rise", ref_priors.tau_rise, 1e-3)
        numpyro_sample("log_tau_fall", ref_priors.tau_fall, 1e-3)
        numpyro_sample("log_extra_sigma", ref_priors.extra_sigma, 1e-3)

        # aux bands
        for b_idx, uniq_b in enumerate(unique_bands):
            if uniq_b == ref_band:
                continue
            b_priors = priors.bands[uniq_b]
            numpyro_sample("A_" + uniq_b, b_priors.amp, 1e-3)
            numpyro_sample("beta_" + uniq_b, b_priors.beta, 1e-3)
            numpyro_sample("gamma_" + uniq_b, b_priors.gamma, 1e-3)
            numpyro_sample("t0_" + uniq_b, b_priors.t_0, 1e-3)
            numpyro_sample("tau_rise_" + uniq_b, b_priors.tau_rise, 1e-3)
            numpyro_sample("tau_fall_" + uniq_b, b_priors.tau_fall, 1e-3)
            numpyro_sample("extra_sigma_" + uniq_b, b_priors.extra_sigma, 1e-3)

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
        )

        # mcmc.print_summary()
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
                obsflux=fdata,
                t=tdata,
                uncertainties=ferrdata,
                max_flux=max_flux,
            )
        params = svi_result.params
        posterior_samples = {}
        for p in params:
            if p[-2:] == "mu":
                posterior_samples[p[:-3]] = np.random.normal(
                    loc=params[p], scale=params[p[:-2] + "sigma"], size=100
                )

    else:
        raise ValueError("'sampler' must be 'NUTS' or 'svi'")

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
        bdata_stacked.append(lc.band_as_int(["g", "r"]))  # change to integers

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
            all_priors = MultibandPriors.load_ztf_priors()
            r_priors = all_priors.bands["r"]
            A = max_flux * 10 ** numpyro.sample("logA", trunc_norm_fields(r_priors.amp))
            beta = numpyro.sample("beta", trunc_norm_fields(r_priors.beta))
            gamma = 10 ** numpyro.sample("log_gamma", trunc_norm_fields(r_priors.gamma))
            t0 = numpyro.sample("t0", trunc_norm(-100.0, 200.0, 0.0, 20.0))
            tau_rise = 10 ** numpyro.sample("log_tau_rise", trunc_norm_fields(r_priors.tau_rise))
            tau_fall = 10 ** numpyro.sample("log_tau_fall", trunc_norm_fields(r_priors.tau_fall))
            extra_sigma = 10 ** numpyro.sample("log_extra_sigma", trunc_norm_fields(r_priors.extra_sigma))

            g_priors = all_priors.bands["g"]
            A_g = numpyro.sample("A_g", trunc_norm_fields(g_priors.amp))
            beta_g = numpyro.sample("beta_g", trunc_norm_fields(g_priors.beta))
            gamma_g = numpyro.sample("gamma_g", trunc_norm_fields(g_priors.gamma))
            t0_g = numpyro.sample("t0_g", trunc_norm_fields(g_priors.t_0))
            tau_rise_g = numpyro.sample("tau_rise_g", trunc_norm_fields(g_priors.tau_rise))
            tau_fall_g = numpyro.sample("tau_fall_g", trunc_norm_fields(g_priors.tau_fall))
            extra_sigma_g = numpyro.sample("extra_sigma_g", trunc_norm_fields(g_priors.extra_sigma))

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
        lc.pad_bands(["r", "g"], PAD_SIZE)
        lcs.append(lc)

    eq_samples = run_mcmc_batch(lcs, plot=True)
    if eq_samples is None:
        return None

    print(np.mean(eq_samples["log_tau_fall"]))

    return None


def numpyro_single_curve(lc, output_dir=FITS_DIR, sampler="svi", priors=MultibandPriors.load_ztf_priors()):
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
    priors : MultibandPriors, optional
        The prior set to use for fitting. Defaults to ZTF's priors.

    Returns
    -------
    sample_mean: numpy array
        Return the mean of the MCMC samples or None if the fitting is
        skipped or encounters an error.
    """
    if lc.name is None or lc.name == "":  # pragma: no cover
        raise ValueError("Empty light curve name.")

    os.makedirs(output_dir, exist_ok=True)

    eq_samples = run_mcmc(lc, sampler=sampler, priors=priors, plot=False)
    if eq_samples is None:  # pragma: no cover
        return None

    posterior_filename = get_posterior_filename(lc.name, output_dir, sampler)
    np.savez_compressed(posterior_filename, eq_samples)
    sample_mean = np.mean(eq_samples, axis=0)
    return sample_mean


def numpyro_single_file(test_filename, output_dir=FITS_DIR, sampler="svi", survey=Survey.ZTF()):
    """Runs MCMC on a single file.

    Parameters
    ----------
    test_filename : str
        Name of the file to use as input.
    output_dir : str
        Directory to save outputs to. Defaults to FITS_DIR.
    sampler : str
        The MCMC sampler to use. Defaults to "svi".
    survey : Survey
        Information about survey used to collect LC data.

    Returns
    -------
    sample_mean: numpy array
        Return the mean of the MCMC samples or None if the fitting is
        skipped or encounters an error.
    """
    lc = Lightcurve.from_file(test_filename)
    lc.pad_bands(survey.priors.ordered_bands, PAD_SIZE)

    sample_mean = numpyro_single_curve(lc, output_dir, sampler, survey.priors)
    return sample_mean
