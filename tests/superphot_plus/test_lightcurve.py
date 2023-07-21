import numpy as np
import os
import pytest

from superphot_plus.file_utils import read_single_lightcurve
from superphot_plus.lightcurve import Lightcurve


def test_create(tmp_path):
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
