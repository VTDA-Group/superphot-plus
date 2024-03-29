import numpy as np
from numpy.testing import assert_allclose

from superphot_plus.samplers.licu_sampler import LiCuSampler


def test_licu_single_file(
    single_ztf_lightcurve_object,
    single_ztf_lightcurve_fit,
    ztf_priors
):
    """Just test that we generated a new file with fits"""
    sampler = LiCuSampler(algorithm="mcmc-ceres", mcmc_niter=1000)
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object, priors=ztf_priors, rstate=np.random.default_rng(9876)
    )

    sample_mean = posterior_samples.sample_mean()
    assert len(sample_mean) == 15

    # Kinda of a diff test
    expected = single_ztf_lightcurve_fit
    assert len(expected) == len(sample_mean)
    #assert_allclose(sample_mean, expected, rtol=1e-5)
    assert posterior_samples.samples.shape == (1, 15)
