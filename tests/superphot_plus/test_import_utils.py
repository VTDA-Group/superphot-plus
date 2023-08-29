import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from superphot_plus.import_utils import add_to_new_csv, clip_lightcurve_end, import_lc


def test_import_lc(single_ztf_lightcurve):
    """Test that we can load a single light curve from CSV"""

    t, f, ferr, b, ra, dec = import_lc(single_ztf_lightcurve)

    assert len(t) == 19
    assert len(f) == 19
    assert len(ferr) == 19
    assert len(b) == 19
    assert ra == pytest.approx(16.93, 0.01)
    assert dec == pytest.approx(32.44, 0.001)


def test_clip_lightcurve_end(single_ztf_lightcurve):
    """Test that we clip the flat part of a light curve."""

    # Start with 10 points in r with 3 to clip. Flat slope.
    times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fluxes = [0.5, 0.8, 10.0, 200.0, 199.0, 189.9, 50.0, 0.1, 0.1, 0.1]
    bands = ["r"] * 10
    errors = [0.1] * 10

    # Add 8 points in g with 4 to clip. Small downward slope.
    times.extend([0, 1, 2, 3, 4, 5, 6, 7])
    fluxes.extend([19.0, 19.5, 19.1, 15.0, 0.2, 0.15, 0.1, 0.1])
    bands.extend(["g"] * 8)
    errors.extend([0.1] * 8)

    # Add 5 points in u with 0 to clip. No flat part.
    times.extend([0, 1, 2, 3, 4])
    fluxes.extend([19.0, 19.5, 19.1, 15.0, 14.0])
    bands.extend(["u"] * 5)
    errors.extend([0.1] * 5)

    # Add 3 points in i with the max as the last point
    times.extend([2, 3, 4])
    fluxes.extend([14.1, 19.0, 19.5])
    bands.extend(["i"] * 3)
    errors.extend([0.1] * 3)

    (t_clip, f_clip, e_clip, b_clip) = clip_lightcurve_end(
        np.array(times),
        np.array(fluxes),
        np.array(errors),
        np.array(bands),
    )

    t_clip = np.array(t_clip)
    f_clip = np.array(f_clip)
    e_clip = np.array(e_clip)
    b_clip = np.array(b_clip)

    # Check r.
    r_inds = b_clip == "r"
    assert len(b_clip[r_inds]) == 7
    assert np.all(f_clip[r_inds] > 0.2)
    assert np.all(t_clip[r_inds] <= 6)
    assert np.all(e_clip[r_inds] == 0.1)

    # Check g.
    g_inds = b_clip == "g"
    assert len(b_clip[g_inds]) == 4
    assert np.all(f_clip[g_inds] > 0.5)
    assert np.all(t_clip[g_inds] <= 3)
    assert np.all(e_clip[g_inds] == 0.1)

    # Check u.
    u_inds = b_clip == "u"
    assert len(b_clip[u_inds]) == 5
    assert np.all(f_clip[u_inds] > 0.5)
    assert np.all(t_clip[u_inds] <= 4)
    assert np.all(e_clip[u_inds] == 0.1)

    # Check i.
    i_inds = b_clip == "i"
    assert len(b_clip[i_inds]) == 3


def test_add_to_new_csv():
    with tempfile.TemporaryDirectory() as dir_name:
        file_name = f"{dir_name}/tmp_data.dat"

        # Create a new file with a single line.
        add_to_new_csv("Name1", "Label1", 10.0, file_name)
        assert Path(file_name).is_file()
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            assert next(reader) == ["Name1", "Label1", "10.0"]
            with pytest.raises(StopIteration):
                next(reader)

        # Append one line.
        add_to_new_csv("Name2", "Label2", 5.0, file_name)
        assert Path(file_name).is_file()
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            assert next(reader) == ["Name1", "Label1", "10.0"]
            assert next(reader) == ["Name2", "Label2", "5.0"]
            with pytest.raises(StopIteration):
                next(reader)
