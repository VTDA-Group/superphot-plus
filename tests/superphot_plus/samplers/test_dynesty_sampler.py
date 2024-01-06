import numpy as np

from superphot_plus.samplers.dynesty_sampler import DynestySampler


def test_dynesty_single_file(
    single_ztf_lightcurve_object,
    ztf_priors,
    single_ztf_lightcurve_fit
):
    """Just test that we generated a new file with fits"""
    sampler = DynestySampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object, priors=ztf_priors, rstate=np.random.default_rng(9876)
    )

    sample_mean = posterior_samples.sample_mean()
    assert len(sample_mean) == 15

    # Check that the same means the same order of magnitude (within 50% relative value).
    # Despite setting the the random seed, we still need to account (so far) unexplained
    # additional variations.
    expected = single_ztf_lightcurve_fit
    assert len(expected) == len(sample_mean)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5, atol=0.2))

    ## could be between ~600 and ~800, and can vary based on hardware.
    assert 600 <= len(posterior_samples.samples) <= 1000
