import os

import numpy as np
import pytest

from superphot_plus.fit_numpyro import numpyro_single_file, trunc_norm, run_mcmc
from superphot_plus.lightcurve import Lightcurve

def test_trunc_norm(jax_key):
    """Test that the trunc_norm() function, which returns numpyro's
    TruncatedNormal distribution, works as expected.
    """
    unit_untrunc = trunc_norm(loc=0.0, scale=1.0, low=None, high=None) # should default to a Normal dist
    assert unit_untrunc.mean == 0.0
    assert unit_untrunc.variance == 1.0
    
    unit_trunc = trunc_norm(loc=0.0, scale=1.0, low=-0.5, high=0.5)
    
    assert np.all(unit_trunc.log_prob(np.array([-0.4, 0.0, 0.4])) > -np.inf)
    assert np.all(unit_trunc.log_prob(np.array([-0.6, 0.6, 10.])) == -np.inf)
    
    assert np.all(np.abs(unit_trunc.sample(jax_key, sample_shape=(10,10))) < 0.5)
    

def test_mcmc_missing_band(single_ztf_lightcurve_compressed):
    """Test that run_mcmc exists out with missing band data.
    """
    lc = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    lc.filter_by_band(["r",])
    assert run_mcmc(lc) is None


def test_nonimplemented_sampler(tmp_path, single_ztf_lightcurve_compressed):
    """Tests that run_mcmc exists out when non-implemented sampler
    name is given.
    """
    with pytest.raises(ValueError):
        numpyro_single_file(single_ztf_lightcurve_compressed, tmp_path, sampler="sampler")
    
    
def test_numpyro_nuts(tmp_path, single_ztf_lightcurve_compressed):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    numpyro_single_file(single_ztf_lightcurve_compressed, tmp_path, sampler="NUTS")

    # Check ouput existence
    output_file = os.path.join(tmp_path, "ZTF22abvdwik_eqwt_NUTS.npz")
    assert os.path.exists(output_file)

    # Check output length
    params = np.load(output_file)["arr_0"]
    assert len(params) == 300

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
    ]
    sample_mean = np.mean(params, axis=0)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))


def test_numpyro_svi(tmp_path, single_ztf_lightcurve_compressed):
    """Test that we generated a new file, that its samples that can be
    read, and the mean of samples generated is within a certain range of
    expected values."""
    numpyro_single_file(single_ztf_lightcurve_compressed, tmp_path, sampler="svi")

    # Check ouput existence
    output_file = os.path.join(tmp_path, "ZTF22abvdwik_eqwt_svi.npz")
    assert os.path.exists(output_file)

    # Check output length
    params = np.load(output_file)["arr_0"]
    assert len(params) == 100

    # Check output values
    expected = [
        9.64e02,
        5.21e-03,
        1.70e01,
        -6.07e00,
        2.95e00,
        2.36e01,
        5.00e-02,
        1.07e00,
        1.04e00,
        1.01e00,
        9.99e-01,
        9.64e-01,
        5.72e-01,
        8.57e-01,
    ]
    sample_mean = np.mean(params, axis=0)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))