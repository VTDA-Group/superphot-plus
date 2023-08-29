import os

import numpy as np
import pytest

from superphot_plus.lightcurve import Lightcurve


def test_create():
    times = np.arange(10)
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
    assert np.all(lc.unique_bands() == np.array(["g", "r"]))
    assert lc.obs_count() == 10
    assert lc.obs_count("r") == 9
    assert lc.obs_count("g") == 1
    assert lc.obs_count("b") == 0

    # Fail a creation
    times2 = np.arange(20)
    with pytest.raises(ValueError):
        lc2 = Lightcurve(times2, fluxes, bands, errors)


def test_max_flux():
    times = np.arange(11)
    fluxes = np.array([1.1, 0.2, 0.3, 0.1, 100.2, 20.3, 0.1, 0.1, 1.1, 0.1, 1000.0])
    bands = np.array(["r", "r", "r", "r", "g", "r", "r", "r", "g", "g", "b"])
    errors = np.array([0.1] * 11)
    lc = Lightcurve(times, fluxes, errors, bands)

    # Test all bands with a standard adjustment (-1.0 * |error|)
    all_max, all_max_t = lc.find_max_flux()
    assert all_max == pytest.approx(999.9)
    assert all_max_t == 10

    # Test g band with a standard adjustment (-1.0 * |error|)
    g_max, g_max_t = lc.find_max_flux(band="g")
    assert g_max == pytest.approx(100.1)
    assert g_max_t == 4

    # Test r band with no adjustment
    r_max, r_max_t = lc.find_max_flux(band="r", error_coeff=0.0)
    assert r_max == pytest.approx(20.3)
    assert r_max_t == 5

    # Test y band gives an error
    with pytest.raises(ValueError, match=r"ERROR: Light curve has no points. band=y"):
        _, _ = lc.find_max_flux(band="y", error_coeff=0.0)


def test_from_file(single_ztf_lightcurve_compressed):
    """Test that we can load a single light curve from pickled file."""
    lc = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    assert len(lc.times) == 19
    assert len(lc.fluxes) == 19
    assert len(lc.flux_errors) == 19
    assert len(lc.bands) == 19
    assert lc.name == "ZTF22abvdwik"

    # Fail when file does not exist.
    with pytest.raises(FileNotFoundError, match="ERROR: File does not exist file_does_not_exist.err"):
        _ = Lightcurve.from_file("file_does_not_exist.err")


def test_write_and_read_single_lightcurve(tmp_path):
    # Create fake data. Note that the first point in the fluxes must be the brightest
    # and the first time stamp must be zero, because of how read_single_lightcurve
    # shifts the times to be zero at the peak.
    times = np.arange(10)
    fluxes = np.array([100.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
    bands = np.array(["r"] * 10)
    errors = np.array([0.1] * 10)
    lc = Lightcurve(times, fluxes, errors, bands, name="my_test_file")

    filename = os.path.join(tmp_path, "my_test_file.npz")
    lc.save_to_file(filename, overwrite=True)

    # Re-read and check data.
    lc2 = Lightcurve.from_file(filename)
    assert lc2.name == "my_test_file"
    assert np.allclose(lc2.times, times)
    assert np.allclose(lc2.fluxes, fluxes)
    assert np.allclose(lc2.flux_errors, errors)
    assert np.all(lc2.bands == bands)

    # Read the data with a t0_lim of 5.0
    lc3 = Lightcurve.from_file(filename, t0_lim=5.0)
    assert np.all(lc3.times <= 5.0)
    assert len(lc3.fluxes) == 6


def test_write_and_read_single_lightcurve_no_shift(tmp_path):
    # Create fake data. Note that the first point in the fluxes must be the brightest
    # and the first time stamp must be zero, because of how read_single_lightcurve
    # shifts the times to be zero at the peak.
    times = np.array(range(10))
    fluxes = np.array([0.1, 0.2, 0.3, 0.1, 100.2, 0.3, 0.1, 0.1, 0.1, 0.1])
    bands = np.array(["r"] * 10)
    errors = np.array([0.1] * 10)
    lc = Lightcurve(times, fluxes, errors, bands, name="my_test_file2")

    filename = os.path.join(tmp_path, "my_test_file2.npz")
    lc.save_to_file(filename, overwrite=True)

    # Re-read and check data.
    lc2 = Lightcurve.from_file(filename, shift_time=False)
    assert lc.name == "my_test_file2"
    assert np.allclose(lc2.times, times)
    assert np.allclose(lc2.fluxes, fluxes)
    assert np.allclose(lc2.flux_errors, errors)
    assert np.all(lc2.bands == bands)

    # If we do shift, about half the times should be <= 0.
    lc3 = Lightcurve.from_file(filename, shift_time=True)
    assert np.allclose(lc3.times, range(-4, 6))


def test_write_and_read_single_lightcurve_nans(tmp_path):
    times = np.arange(8)
    fluxes = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.0])
    bands = np.array(["g"] * 8)
    errors = np.array([0.1, 0.1, 0.1, 0.1, np.NAN, 0.1, np.NAN, 0.2])
    lc = Lightcurve(times, fluxes, errors, bands, name="my_nan_test_file")

    filename = os.path.join(tmp_path, "my_nan_test_file.npz")
    lc.save_to_file(filename, overwrite=True)

    # Re-read and check data.
    lc2 = Lightcurve.from_file(filename)
    assert lc2.name == "my_nan_test_file"
    assert np.allclose(lc2.times, [0.0, 1.0, 2.0, 3.0, 5.0, 7.0])
    assert np.allclose(lc2.fluxes, [0.1, 0.2, 0.3, 0.1, 0.3, 0.0])
    assert np.allclose(lc2.flux_errors, [0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
    assert np.all(lc2.bands == ["g"] * 6)


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


def test_filter_bands():
    times = np.arange(9)
    fluxes = np.array(range(10, 19))
    bands = np.array(["r", "r", "g", "r", "g", "b", "g", "i", "r"])
    errors = np.arange(9) / 10.0
    lc = Lightcurve(times, fluxes, errors, bands)
    assert np.all(lc.unique_bands() == np.array(["b", "g", "i", "r"]))

    lc2 = lc.filter_by_band(["r", "g"], in_place=False)

    # New array is filtered.
    assert np.all(lc2.times == np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]))
    assert np.all(lc2.fluxes == np.array([10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 18.0]))
    assert np.all(lc2.flux_errors == np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]))
    assert np.all(lc2.bands == np.array(["r", "r", "g", "r", "g", "g", "r"]))

    # Original array is unchanged.
    assert np.all(lc.times == times)
    assert np.all(lc.fluxes == fluxes)
    assert np.all(lc.flux_errors == errors)
    assert np.all(lc.bands == bands)

    # Do the filtering in place and check the original.
    lc3 = lc.filter_by_band(["r", "g"], in_place=True)
    assert np.all(lc.times == np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]))
    assert np.all(lc.fluxes == np.array([10.0, 11.0, 12.0, 13.0, 14.0, 16.0, 18.0]))
    assert np.all(lc.flux_errors == np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]))
    assert np.all(lc.bands == np.array(["r", "r", "g", "r", "g", "g", "r"]))


def test_band_as_int():
    times = np.arange(9)
    fluxes = np.array(range(10, 19))
    bands = np.array(["r", "r", "g", "r", "g", "b", "g", "i", "r"])
    errors = np.arange(9) / 10.0
    lc = Lightcurve(times, fluxes, errors, bands)

    res1 = lc.band_as_int(["r", "g", "i", "b"])
    assert np.all(res1 == np.array([0, 0, 1, 0, 1, 3, 1, 2, 0]))

    # Only a subset of the bands included.
    res2 = lc.band_as_int(["r", "g"], fail_on_missing=False)
    assert np.all(res2 == np.array([0, 0, 1, 0, 1, -1, 1, -1, 0]))

    # A subset of bands with strict checking.
    with pytest.raises(ValueError, match="ERROR: Unmapped bands found in band_as_int."):
        _ = lc.band_as_int(["r", "g"])


def test_padding():
    # Add 7 points in r, 3 in g, 1 in b.
    times = np.array([0.0, 1.5, 5.0, 1.0, 6.0, 2.0, 3.0, 4.0, 0.5, 2.5, 0.05])
    fluxes = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5, 2.5, 1.0])
    bands = np.array(["r", "g", "r", "r", "r", "r", "r", "r", "g", "g", "b"])
    errors = np.array([0.1] * 11)
    lc = Lightcurve(times, fluxes, errors, bands)
    assert np.all(lc.unique_bands() == np.array(["b", "g", "r"]))

    # Do the padding
    lc2 = lc.pad_bands(["r", "g"], 6, in_place=False)
    assert lc2.obs_count() == 12
    assert lc2.obs_count("r") == 6
    assert lc2.obs_count("g") == 6
    assert lc2.obs_count("b") == 0
    assert np.all(lc2.unique_bands() == np.array(["g", "r"]))

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


def test_debug_string():
    times = np.array([0.0, 1.5, 5.0])
    fluxes = np.array([1.0, 1.5, 2.0])
    bands = np.array(["r", "g", "r"])
    errors = np.array([0.0, 0.0, 0.1])

    lc = Lightcurve(times, fluxes, errors, bands, name="Test")
    db_str = lc.debug_string()

    expected_value = (
        "Supernova (name=Test, class=None, size=3)\n"
        "  Times: [0.  1.5 5. ]\n  Fluxes: [1.  1.5 2. ]\n"
        "  Flux Errors: [0.  0.  0.1]\n  Bands: ['r' 'g' 'r']\n"
    )
    assert db_str == expected_value
