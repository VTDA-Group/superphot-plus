import numpy as np

from superphot_plus.samplers.flowmc_sampler import FlowMCSampler


def test_flow_mc_single_file(single_ztf_lightcurve_object, ztf_priors, tmp_path):
    """Just test that we generated a new file with fits"""
    sampler = FlowMCSampler()
    posterior_samples = sampler.run_single_curve(single_ztf_lightcurve_object, priors=ztf_priors, rng_seed=4)

    sample_mean = posterior_samples.sample_mean()
    assert len(sample_mean) == 14

    # Check that the same means the same order of magnitude (within 50% relative value).
    # Despite setting the the random seed, we still need to account (so far) unexplained
    # additional variations.
    expected = [
        7.32584736e-02,
        5.18737325e-03,
        1.15182256e00,
        -5.82298785e00,
        5.99567212e-01,
        1.43106448e00,
        -1.54809571e00,
        1.06056227e00,
        1.04250956e00,
        1.00708208e00,
        9.99091344e-01,
        9.66291236e-01,
        5.46973286e-01,
        8.61311435e-01,
    ]
    assert len(expected) == len(sample_mean)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))

    assert len(posterior_samples.samples) == 400