import numpy as np

from superphot_plus.samplers.flowmc_sampler import FlowMCSampler


def test_flow_mc_single_file(single_ztf_lightcurve_object, ztf_priors, tmp_path):
    """Just test that we generated a new file with fits"""
    sampler = FlowMCSampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        priors=ztf_priors,
        rng_seed=4,
        n_chains=1,
        num_epochs=1,
    )

    sample_mean = posterior_samples.sample_mean()
    assert len(sample_mean) == 14

    # Check that the means are the same (within 5% relative value).
    expected = [
        0.16598069,
        0.15574268,
        0.26327706,
        -0.13079915,
        0.12729716,
        -0.04977773,
        -0.91312219,
        0.42187455,
        0.23734298,
        0.10059648,
        0.03599622,
        0.40210226,
        0.17104216,
        0.6381275,
    ]
    assert len(expected) == len(sample_mean)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.05))

    assert len(posterior_samples.samples) == 100
