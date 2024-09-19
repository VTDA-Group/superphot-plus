import numpy as np
import pytest

from superphot_plus.samplers.numpyro_sampler import SVISampler, NUTSSampler, trunc_norm
from superphot_plus.surveys.fitting_priors import PriorFields


def test_trunc_norm(jax_key):
    """Test that the trunc_norm() function, which returns numpyro's
    TruncatedNormal distribution, works as expected.
    """
    fields = PriorFields(None, None, 0, 1.0)
    unit_untrunc = trunc_norm(fields)  # should default to a Normal dist
    assert unit_untrunc.mean == 0.0
    assert unit_untrunc.variance == 1.0

    fields = PriorFields(-0.5, 0.5, 0, 1.0)
    unit_trunc = trunc_norm(fields)

    assert np.all(unit_trunc.log_prob(np.array([-0.4, 0.0, 0.4])) > -np.inf)
    assert np.all(unit_trunc.log_prob(np.array([-0.6, 0.6, 10.0])) == -np.inf)

    assert np.all(np.abs(unit_trunc.sample(jax_key, sample_shape=(10, 10))) < 0.5)

def test_svi_sampler(
    test_ztf_photometry,
    ztf_priors,
    test_sampler_result
):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    sampler = SVISampler(
        priors=ztf_priors, random_state=np.random.default_rng(9876)
    )
    sampler.fit_photometry(test_ztf_photometry)
    sample_mean = np.mean(sampler.result.fit_parameters, axis=0)
    assert sampler.result.fit_parameters.shape == (100, 14)

    # Check that the same means the same order of magnitude (within 50% relative value).
    # Despite setting the the random seed, we still need to account (so far) unexplained
    # additional variations.
    expected_mean = np.mean(test_sampler_result.fit_parameters, axis=0)
    assert len(expected_mean) == len(sample_mean)
    assert np.all(np.isclose(sample_mean, expected_mean, rtol=0.5, atol=0.2))

def test_nuts_sampler(
    test_ztf_photometry,
    ztf_priors,
    test_sampler_result
):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    sampler = NUTSSampler(
        priors=ztf_priors, random_state=np.random.default_rng(9876)
    )
    sampler.fit_photometry(test_ztf_photometry)
    sample_mean = np.mean(sampler.result.fit_parameters, axis=0)
    assert sampler.result.fit_parameters.shape == (100, 14)

    # Check that the same means the same order of magnitude (within 50% relative value).
    # Despite setting the the random seed, we still need to account (so far) unexplained
    # additional variations.
    expected_mean = np.mean(test_sampler_result.fit_parameters, axis=0)
    assert len(expected_mean) == len(sample_mean)
    assert np.all(np.isclose(sample_mean, expected_mean, rtol=0.5, atol=0.2))
