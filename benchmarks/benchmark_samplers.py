"""A script to manually benchmark the different samplers.

This is very time consuming as each sampler is run multiple times.
"""

import copy
import timeit
from os import urandom

import numpy as np
from memory_profiler import memory_usage

from superphot_plus.data_generation.make_fake_spp_data import create_clean_models, create_ztf_model
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.iminuit_sampler import IminuitSampler
from superphot_plus.samplers.licu_sampler import LiCuSampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import calculate_log_likelihood, calculate_mse


def create_data(num_samples: int, clean_only: bool):
    """Create the test data.

    Parameters
    ----------
    num_samples : int
        The number of data points to generate.
    clean_only : bool
        Whether to sample from clean curves or noisy curves.

    Returns
    -------
    parameters, lcs : list of numpy arrays, list of Lightcurves
    """
    parameters = []
    lcs = []

    for i in range(num_samples):
        if clean_only:
            p_i, lc_i = create_clean_models(1)
            parameters.append(p_i[:7])
            lcs.append(lc_i[0])
        else:
            (
                (A, beta, gamma, t0, tau_rise, tau_fall, es),
                tdata,
                filter_data,
                dirty_model,
                sigmas,
            ) = create_ztf_model()
            parameters.append(np.array([A, beta, gamma, t0, tau_rise, tau_fall, es]))
            lcs.append(Lightcurve(tdata, dirty_model, sigmas, filter_data))

    return parameters, lcs


def setup_sampler(sampler_name, seed, **kwargs):
    """Create a sampler and its kwargs from its name.

    Parameter
    ---------
    sampler_name : str
        The name of the sampler to use. One of "dynesty", "svi" or "NUTS".
    seed : int
        Random seed value used for deterministic data generation.
    kwargs : dict
        The existing keyword arguments. New keyword arguments are appended
        to a copy of this dictionary.

    Returns
    -------
    sampler : Sampler
        The sampler object.
    kwargs2 : dict
        The sampler specific arguments.
    """
    kwargs2 = copy.copy(kwargs)
    kwargs2["priors"] = priors = Survey.ZTF().priors

    if sampler_name == "dynesty":
        sampler_obj = DynestySampler()
    elif sampler_name == "svi":
        sampler_obj = NumpyroSampler()
        kwargs2["sampler"] = "svi"
    elif sampler_name == "NUTS":
        sampler_obj = DynestySampler()
        kwargs2["sampler"] = "NUTS"
    elif sampler_name == "iminuit":
        sampler_obj = IminuitSampler()
    elif sampler_name == "licu-ceres":
        sampler_obj = LiCuSampler(algorithm="ceres")
    elif sampler_name == "licu-mcmc-ceres":
        sampler_obj = LiCuSampler(algorithm="mcmc-ceres", mcmc_niter=10_000)
    else:
        raise ValueError(f"Unknown sampler {sampler_name}")

    kwargs2["rng_seed"] = seed

    return sampler_obj, kwargs2


def run_one_benchmark(sampler_name, lightcurve, seed, **kwargs):
    """Benchmark a single sampler, lightcurve pair.

    Parameters
    ----------
    sampler_name : string
        The anme of the sampler to use. One of "dynesty", "svi" or "NUTS".
    lightcurve : Lightcurves
        The test lightcurve to use.
    seed : int
        Random seed value used for deterministic data generation.
    kwargs : dict
        The existing keyword arguments.

    Returns
    -------
    res : list
        A list of the metrics (resulting time, memory usage, loglikelihood, and
        mean square error) for this run.
    """
    sampler_obj, kwargs2 = setup_sampler(sampler_name, seed, **kwargs)

    # Do initial runs to warm up the sampler and get the accuracy results.
    posterior_samples = sampler_obj.run_single_curve(lightcurve, **kwargs2)
    est_params = posterior_samples.sample_mean()[:14]
    res_logl = calculate_log_likelihood(est_params, lightcurve, ["r", "g"], "r")
    res_mse = calculate_mse(est_params, lightcurve, ["r", "g"], "r")

    # Do three timing runs and use the mean of the time taken.
    tmr = timeit.Timer(stmt="sampler_obj.run_single_curve(lightcurve, **kwargs2)", globals=locals())
    res_time = np.mean(tmr.repeat(repeat=1, number=3))

    # Compute the memory usage.
    res_mem = memory_usage(
        (sampler_obj.run_single_curve, [lightcurve], kwargs2),
        max_usage=True,
        interval=0.1,
        max_iterations=1,
    )

    return [res_time, res_mem, res_logl, res_mse]


def run_all_benchmarks(num_samples):
    _, lightcurves = create_data(num_samples, True)

    seed = int.from_bytes(urandom(4), "big")
    samplers = ["dynesty", "svi", "NUTS", "iminuit", "licu-ceres", "licu-mcmc-ceres"]

    results = {}
    for sampler in samplers:
        results[sampler] = []

    for i, lc in enumerate(lightcurves):
        for sampler in samplers:
            print(f"\nTesting curve {i} with {sampler}:")
            itr_res = run_one_benchmark(sampler, lc, seed)
            results[sampler].append(itr_res)

            print(f"  Itr Running Time: {itr_res[0]}")
            print(f"  Itr Memory Usage: {itr_res[1]}")
            print(f"  Itr LL: {itr_res[2]}")
            print(f"  Itr MSE: {itr_res[3]}")

    print("=" * 20)
    for sampler in samplers:
        aves = np.mean(results[sampler], axis=0)
        print(f"\nFinal results for {sampler} sampler:")
        print(f"  Ave Running Time: {aves[0]}")
        print(f"  Ave Memory Usage: {aves[1]}")
        print(f"  Ave LL: {aves[2]}")
        print(f"  Ave MSE: {aves[3]}")


if __name__ == "__main__":
    run_all_benchmarks(20)
