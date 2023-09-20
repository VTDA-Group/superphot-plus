import numpy as np
from numpy.testing import assert_allclose

from superphot_plus.samplers.iminuit_sampler import IminuitSampler


def test_iminuit_single_file(single_ztf_lightcurve_object, ztf_priors):
    """Just test that we generated a new file with fits"""
    sampler = IminuitSampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object, priors=ztf_priors
    )

    sample_mean = posterior_samples.sample_mean()
    assert len(sample_mean) == 15

    # Kinda of a diff test
    expected = [8.701402e+02,  5.199848e-03,  1.381310e+01, -1.001759e+00,
                3.962802e+00,  2.690830e+01,  3.162278e-01,  1.153953e+01,
                1.042400e+00,  1.017419e+01,  9.999890e-01,  9.253365e+00,
                3.538343e+00,  6.818018e+00, -9.056224e+00]
    assert len(expected) == len(sample_mean)
    assert_allclose(sample_mean, expected, rtol=1e-5)
    assert posterior_samples.samples.shape == (100, 15)
