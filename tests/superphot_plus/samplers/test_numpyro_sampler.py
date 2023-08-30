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
    assert run_mcmc(single_ztf_lightcurve_object) is None


def test_nonimplemented_sampler(single_ztf_lightcurve_object, ztf_priors):
    """Tests that run_mcmc exists out when non-implemented sampler
    name is given.
    """
    with pytest.raises(ValueError):
        sampler = NumpyroSampler()
        _ = sampler.run_single_curve(single_ztf_lightcurve_object, priors=ztf_priors, sampler="sampler")


def test_numpyro_nuts(ztf_priors, single_ztf_lightcurve_object):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    sampler = NumpyroSampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object, priors=ztf_priors, sampler="NUTS"
    )
    # Check output length
    assert len(posterior_samples.samples) == 300

    # Check output values
    expected = [
        9.85589522e02,
        5.19716954e-03,
        1.61198756e01,
        -5.75673236e00,
        3.26708896e00,
        2.38970410e01,
        3.64242112e-02,
        1.04759061e00,
        1.04258722e00,
        1.00856218e00,
        9.99988091e-01,
        9.66154117e-01,
        5.76787619e-01,
        8.59146651e-01,
        -5.43,
    ]
    sample_mean = posterior_samples.sample_mean()
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))


def test_numpyro_svi(ztf_priors, single_ztf_lightcurve_object):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    sampler = NumpyroSampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object, priors=ztf_priors, sampler="svi"
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
        2.00e-02,
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
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))
