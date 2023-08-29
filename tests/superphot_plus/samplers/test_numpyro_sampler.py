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
        single_ztf_lightcurve_object, priors=ztf_priors, sampler="NUTS", rng_seed=4,
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
        0.03297451,
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

    print("NUTS:")
    for v in sample_mean:
        print("%.8f," % v)
    
    assert np.all(np.isclose(sample_mean, expected, rtol=0.01))


def test_numpyro_svi(ztf_priors, single_ztf_lightcurve_object):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    sampler = NumpyroSampler()
    posterior_samples = sampler.run_single_curve(
        single_ztf_lightcurve_object, priors=ztf_priors, sampler="svi", rng_seed=1,
    )
    # Check output length
    assert len(posterior_samples.samples) == 100

    # Check output values
    expected = [
        884.51462144,
        0.00528629,
        18.80530674,
        -6.33601188,
        2.74016769,
        25.16111683,
        0.03493173,
        1.06160793,
        1.04252620,
        1.01352523,
        0.99999384,
        0.96840261,
        0.62417873,
        0.86067821,
        -5.63161335,
    ]
    sample_mean = posterior_samples.sample_mean()

    print("SVI:")
    for v in sample_mean:
        print("%.8f," % v)

    assert np.all(np.isclose(sample_mean, expected, rtol=0.01))
