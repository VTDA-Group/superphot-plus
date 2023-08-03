import os

import numpy as np
import pytest

from superphot_plus.file_utils import get_posterior_samples, read_single_lightcurve


def test_read_single_lightcurve(single_ztf_lightcurve_compressed):
    """Test that we can load a single light curve from pickled file."""

    time, flux, flux_err, band = read_single_lightcurve(single_ztf_lightcurve_compressed)

    assert len(time) == 19
    assert len(flux) == 19
    assert len(flux_err) == 19
    assert len(band) == 19


def test_read_single_lightcurve_with_time_celing(single_ztf_lightcurve_compressed):
    """Test that we can load a single light curve from pickled file,
    restricting the time window for events."""

    ## nonsense time.
    time, flux, flux_err, band = read_single_lightcurve(single_ztf_lightcurve_compressed, time_ceiling=45.0)

    expected_length = 0
    assert len(time) == expected_length
    assert len(flux) == expected_length
    assert len(flux_err) == expected_length
    assert len(band) == expected_length

    ## time somewhere within the light curve.
    time, flux, flux_err, band = read_single_lightcurve(
        single_ztf_lightcurve_compressed, time_ceiling=59932.0
    )

    expected_length = 15
    assert len(time) == expected_length
    assert len(flux) == expected_length
    assert len(flux_err) == expected_length
    assert len(band) == expected_length


def test_get_posterior_samples(single_ztf_sn_id, single_ztf_eqwt_compressed, test_data_dir):
    """Test loading the posterior samples from an EQWT fits file"""

    # Check existence of EQWT fits file.
    assert os.path.exists(single_ztf_eqwt_compressed)

    # Read posterior samples from file.
    post_arr = get_posterior_samples(single_ztf_sn_id, fits_dir=test_data_dir)

    # Check output length.
    assert 600 <= len(post_arr) <= 800

    # Check output values.
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
    sample_mean = np.mean(post_arr, axis=0)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))
