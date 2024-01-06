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
        sampler = NumpyroSampler(sampler="sampler")
        _ = sampler.run_single_curve(
            single_ztf_lightcurve_object,
            priors=ztf_priors,
            rng_seed=None
        )


def test_numpyro_nuts(
    ztf_priors,
    single_ztf_lightcurve_object,
    single_ztf_lightcurve_fit,
):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    sampler = NumpyroSampler(sampler="NUTS")
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        priors=ztf_priors,
        rng_seed=4,
    )
    # Check output length
    assert len(posterior_samples.samples) == 1200

    # Check output values
    expected = single_ztf_lightcurve_fit
    sample_mean = posterior_samples.sample_mean()
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5, atol=0.2))

    # Test that on the same system and in the same environment, the same random
    # seed produces the same results.
    posterior_samples2 = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        rng_seed=4,
        priors=ztf_priors,
    )
    sample_mean2 = posterior_samples2.sample_mean()
    assert np.allclose(sample_mean, sample_mean2)

    # And a different random seed provides a different result.
    posterior_samples3 = sampler.run_single_curve(
        single_ztf_lightcurve_object,
        rng_seed=5,
        priors=ztf_priors,
    )
    sample_mean3 = posterior_samples3.sample_mean()
    assert not np.allclose(sample_mean, sample_mean3)


def test_numpyro_svi(ztf_priors, single_ztf_lightcurve_object, single_ztf_lightcurve_fit):
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
    expected = single_ztf_lightcurve_fit
    sample_mean = posterior_samples.sample_mean()
    print(sample_mean)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5, atol=0.2))

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
