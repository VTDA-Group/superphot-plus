import os

import numpy as np

from superphot_plus.fit_numpyro import numpyro_single_file


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
        9.85589522e+02,
        5.19716954e-03,
        1.61198756e+01,
        -5.75673236e+00,
        3.26708896e+00,
        2.38970410e+01,
        3.64242112e-02,
        1.04759061e+00,
        1.04258722e+00,
        1.00856218e+00,
        9.99988091e-01,
        9.66154117e-01,
        5.76787619e-01,
        8.59146651e-01
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
