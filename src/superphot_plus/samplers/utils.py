import copy

from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.iminuit_sampler import IminuitSampler
from superphot_plus.samplers.licu_sampler import LiCuSampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler


def setup_sampler(sampler_name, priors, seed, **kwargs):
    """Creates a sampler and its kwargs from its name.

    Parameter
    ---------
    sampler_name : str
        The name of the sampler to use. One of "dynesty", "svi",
        "NUTS", "iminuit", "licu-ceres" or "licu-mcmc-ceres".
    priors : MultibandPriors
        The survey priors to use.
    seed : int
        Random seed value used for deterministic data generation.

    Returns
    -------
    sampler : Sampler
        The sampler object.
    kwargs2 : dict
        The sampler specific arguments.
    """
    kwargs2 = copy.copy(kwargs)

    kwargs2["priors"] = priors

    if sampler_name == "dynesty":
        sampler_obj = DynestySampler()
    elif sampler_name == "svi":
        sampler_obj = NumpyroSampler()
        kwargs2["sampler"] = "svi"
    elif sampler_name == "NUTS":
        sampler_obj = NumpyroSampler()
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
