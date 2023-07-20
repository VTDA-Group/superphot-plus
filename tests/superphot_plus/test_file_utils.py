import numpy as np
import os
import pytest

from superphot_plus.file_utils import read_single_lightcurve, save_single_lightcurve


def test_read_single_lightcurve(single_ztf_lightcurve_compressed):
    """Test that we can load a single light curve from pickled file."""

    time, flux, flux_err, band = read_single_lightcurve(single_ztf_lightcurve_compressed)

    assert len(time) == 19
    assert len(flux) == 19
    assert len(flux_err) == 19
    assert len(band) == 19


def test_write_and_read_single_lightcurve(tmp_path):
    # Create fake data. Note that the first point in the fluxes must be the brightest
    # and the first time stamp must be zero, because of how read_single_lightcurve
    # shifts the times to be zero at the peak.
    times = np.array(range(10))
    fluxes = np.array([100.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
    bands = np.array(["r"] * 10)
    errors = np.array([0.1] * 10)

    filename = os.path.join(tmp_path, "my_test_file.npz")
    save_single_lightcurve(filename, times, fluxes, errors, bands, overwrite=True)

    # Re-read and check data.
    t2, f2, e2, b2 = read_single_lightcurve(filename)
    assert np.allclose(t2, times)
    assert np.allclose(f2, fluxes)
    assert np.allclose(e2, errors)

    # If we try to save again (without overwrite=True) we should get an error.
    with pytest.raises(FileExistsError):
        save_single_lightcurve(filename, times, fluxes, errors, bands, overwrite=False)


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
