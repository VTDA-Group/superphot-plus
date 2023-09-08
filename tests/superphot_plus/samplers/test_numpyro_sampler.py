import numpy as np
import pytest

from superphot_plus.samplers.numpyro_sampler import NumpyroSampler, run_mcmc, trunc_norm
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


def test_mcmc_missing_band(single_ztf_lightcurve_object):
    """Test that run_mcmc exists out with missing band data."""
    single_ztf_lightcurve_object.filter_by_band(
        [
            "r",
        ]
    )
    assert run_mcmc(single_ztf_lightcurve_object, None) is None


def test_nonimplemented_sampler(single_ztf_lightcurve_object, ztf_priors):
    """Tests that run_mcmc exists out when non-implemented sampler
    name is given.
    """
    with pytest.raises(ValueError):
        sampler = NumpyroSampler()
        _ = sampler.run_single_curve(
            single_ztf_lightcurve_object, priors=ztf_priors, rng_seed=None, sampler="sampler"
        )


def test_numpyro_nuts(ztf_priors, single_ztf_lightcurve_object):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    sampler = NumpyroSampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        priors=ztf_priors,
        rng_seed=4,
        sampler="NUTS",
    )
    # Check output length
    assert len(posterior_samples.samples) == 300

    # Check output values
    expected = [
        895.47189930,
        0.00521055,
        18.16068104,
        -6.17901895,
        2.87213922,
        25.74471707,
        0.025,
        1.05761951,
        1.04262573,
        1.01245135,
        0.99999012,
        0.96668294,
        0.62614179,
        0.85886780,
        -5.59074839,
    ]
    sample_mean = posterior_samples.sample_mean()
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))

    # Test that on the same system and in the same environment, the same random
    # seed produces the same results.
    posterior_samples2 = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        rng_seed=4,
        priors=ztf_priors,
        sampler="NUTS",
    )
    sample_mean2 = posterior_samples2.sample_mean()
    assert np.allclose(sample_mean, sample_mean2)

    # And a different random seed provides a different result.
    posterior_samples3 = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        rng_seed=5,
        priors=ztf_priors,
        sampler="NUTS",
    )
    sample_mean3 = posterior_samples3.sample_mean()
    assert not np.allclose(sample_mean, sample_mean3)


def test_numpyro_svi(ztf_priors, single_ztf_lightcurve_object):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    sampler = NumpyroSampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        rng_seed=1,
        priors=ztf_priors,
        sampler="svi",
    )
    # Check output length
    assert len(posterior_samples.samples) == 100

    # Check output values
    expected = [
        9.64e02,
        5.21e-03,
        1.70e01,
        -6.07e00,
        2.95e00,
        2.36e01,
        2.5e-02,
        1.07e00,
        1.04e00,
        1.01e00,
        9.99e-01,
        9.64e-01,
        5.72e-01,
        8.57e-01,
        -5.43,
    ]
    sample_mean = posterior_samples.sample_mean()
    print(sample_mean)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.25))

    # Test that on the same system and in the same environment, the same random
    # seed produces the same results.
    posterior_samples2 = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        rng_seed=1,
        priors=ztf_priors,
        sampler="svi",
    )
    sample_mean2 = posterior_samples2.sample_mean()
    assert np.allclose(sample_mean, sample_mean2)

    # And a different random seed provides a different result.
    posterior_samples3 = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        rng_seed=2,
        priors=ztf_priors,
        sampler="svi",
    )
    sample_mean3 = posterior_samples3.sample_mean()
    assert not np.allclose(sample_mean, sample_mean3)
