import os

import numpy as np
import pytest

from superphot_plus.file_utils import read_single_lightcurve
from superphot_plus.lightcurve import Lightcurve


def test_create():
    times = np.array(range(10))
    fluxes = np.array([100.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
    bands = np.array(["r", "r", "r", "r", "r", "r", "r", "r", "r", "g"])
    errors = np.array([0.1] * 10)
    lc = Lightcurve(times, fluxes, errors, bands)

    assert np.allclose(times, lc.times)
    assert np.allclose(fluxes, lc.fluxes)
    assert np.allclose(errors, lc.flux_errors)
    assert np.all(bands == lc.bands)
    assert lc.name is None

    # Check the per-band counts
    assert lc.obs_count() == 10
    assert lc.obs_count("r") == 9
    assert lc.obs_count("g") == 1
    assert lc.obs_count("b") == 0

    # Fail a creation
    times2 = np.array(range(20))
    with pytest.raises(ValueError):
        lc2 = Lightcurve(times2, fluxes, bands, errors)


def test_max_flux():
    times = np.array(range(11))
    fluxes = np.array([1.1, 0.2, 0.3, 0.1, 100.2, 20.3, 0.1, 0.1, 1.1, 0.1, 1000.0])
    bands = np.array(["r", "r", "r", "r", "g", "r", "r", "r", "g", "g", "b"])
    errors = np.array([0.1] * 11)
    lc = Lightcurve(times, fluxes, errors, bands)

    # Test all bands with a standard adjustment (-1.0 * |error|)
    all_max, all_max_t = lc.find_max_flux()
    assert pytest.approx(999.9) == all_max
    assert 10 == all_max_t

    # Test g band with a standard adjustment (-1.0 * |error|)
    g_max, g_max_t = lc.find_max_flux(band="g")
    assert pytest.approx(100.1) == g_max
    assert 4 == g_max_t

    # Test r band with no adjustment
    r_max, r_max_t = lc.find_max_flux(band="r", error_coeff=0.0)
    assert pytest.approx(20.3) == r_max
    assert 5 == r_max_t

    # Test y band gives an error
    with pytest.raises(ValueError):
        _, _ = lc.find_max_flux(band="y", error_coeff=0.0)


def test_from_file(single_ztf_lightcurve_compressed):
    """Test that we can load a single light curve from pickled file."""
    lc = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    assert len(lc.times) == 19
    assert len(lc.fluxes) == 19
    assert len(lc.flux_errors) == 19
    assert len(lc.bands) == 19
    assert lc.name == "ZTF22abvdwik"


def test_write_and_read_single_lightcurve(tmp_path):
    # Create fake data. Note that the first point in the fluxes must be the brightest
    # and the first time stamp must be zero, because of how read_single_lightcurve
    # shifts the times to be zero at the peak.
    times = np.array(range(10))
    fluxes = np.array([100.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
    bands = np.array(["r"] * 10)
    errors = np.array([0.1] * 10)
    lc = Lightcurve(times, fluxes, errors, bands, name="my_test_file")

    filename = os.path.join(tmp_path, "my_test_file.npz")
    lc.save_to_file(filename, overwrite=True)

    # Re-read and check data.
    t2, f2, e2, b2 = read_single_lightcurve(filename)
    assert np.allclose(t2, times)
    assert np.allclose(f2, fluxes)
    assert np.allclose(e2, errors)


def test_sort():
    times = np.array([2.0, 1.0, 0.0, 4.0, 3.0])
    fluxes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bands = np.array(["r", "r", "g", "r", "g"])
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    lc = Lightcurve(times, fluxes, errors, bands)
    lc.sort_by_time()

    assert np.all(lc.times == np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.all(lc.fluxes == np.array([3.0, 2.0, 1.0, 5.0, 4.0]))
    assert np.all(lc.flux_errors == np.array([0.3, 0.2, 0.1, 0.5, 0.4]))
    assert np.all(lc.bands == np.array(["g", "r", "r", "g", "r"]))


def test_sort_copy():
    times = np.array([2.0, 1.0, 0.0, 4.0, 3.0])
    fluxes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bands = np.array(["r", "r", "g", "r", "g"])
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    lc = Lightcurve(times, fluxes, errors, bands)
    lc2 = lc.sort_by_time(in_place=False)

    assert np.all(lc2.times == np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.all(lc2.fluxes == np.array([3.0, 2.0, 1.0, 5.0, 4.0]))
    assert np.all(lc2.flux_errors == np.array([0.3, 0.2, 0.1, 0.5, 0.4]))
    assert np.all(lc2.bands == np.array(["g", "r", "r", "g", "r"]))

    assert np.all(lc.times == times)
    assert np.all(lc.fluxes == fluxes)
    assert np.all(lc.flux_errors == errors)
    assert np.all(lc.bands == bands)


def test_padding():
    # Add 7 points in r, 3 in g, 1 in b.
    times = np.array([0.0, 1.5, 5.0, 1.0, 6.0, 2.0, 3.0, 4.0, 0.5, 2.5, 0.05])
    fluxes = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5, 2.5, 1.0])
    bands = np.array(["r", "g", "r", "r", "r", "r", "r", "r", "g", "g", "b"])
    errors = np.array([0.1] * 11)
    lc = Lightcurve(times, fluxes, errors, bands)

    # Do the padding
    lc2 = lc.pad_bands(["r", "g"], 6, in_place=False)
    assert lc2.obs_count() == 12
    assert lc2.obs_count("r") == 6
    assert lc2.obs_count("g") == 6
    assert lc2.obs_count("b") == 0

    # Check that lc2 is correctly padded and ordered by band.
    exp_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 5000.0, 5000.0, 5000.0])
    assert np.all(lc2.times == exp_times)

    exp_flux = np.array([1.0, 3.0, 5.0, 6.0, 7.0, 2.0, 0.5, 1.5, 2.5, 0.0, 0.0, 0.0])
    assert np.all(lc2.fluxes == exp_flux)

    exp_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1e10, 1e10, 1e10])
    assert np.all(lc2.flux_errors == exp_err)

    exp_bands = np.array(["r", "r", "r", "r", "r", "r", "g", "g", "g", "g", "g", "g"])
    assert np.all(lc2.bands == exp_bands)

    # Check that the original lc is not modified.
    assert np.all(times == lc.times)
    assert np.all(fluxes == lc.fluxes)
    assert np.all(errors == lc.flux_errors)
    assert np.all(bands == lc.bands)
