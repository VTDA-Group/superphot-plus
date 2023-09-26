import numpy as np
from numpy.testing import assert_allclose

from superphot_plus.samplers.licu_sampler import LiCuSampler


def test_licu_single_file(single_ztf_lightcurve_object, ztf_priors):
    """Just test that we generated a new file with fits"""
    sampler = LiCuSampler(algorithm="mcmc-ceres", mcmc_niter=1000)
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object, priors=ztf_priors, rstate=np.random.default_rng(9876)
    )

    sample_mean = posterior_samples.sample_mean()
    assert len(sample_mean) == 15

    # Kinda of a diff test
    expected = [1.024755e+03,  2.129157e-02,  2.598141e+01, -3.616342e+00,
                2.994506e+00,  7.479264e+00,  2.403655e+01,  5.215213e+02,
                2.140000e-01,  1.655615e-01,  1.036981e+00,  1.712223e+00,
                3.981072e+00,  7.254375e+00, -1.258408e+01]
    assert len(expected) == len(sample_mean)
    assert_allclose(sample_mean, expected, rtol=1e-5)
    assert posterior_samples.samples.shape == (1, 15)
