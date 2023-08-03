import os

import numpy as np

from superphot_plus.ztf_transient_fit import dynesty_single_file, run_curve_fit


def test_dynesty_single_file(tmp_path, single_ztf_lightcurve_compressed):
    """Just test that we generated a new file with fits"""
    sample_mean = dynesty_single_file(
        single_ztf_lightcurve_compressed,
        tmp_path,
        skip_if_exists=False,
        rstate=np.random.default_rng(9876),
    )
    assert len(sample_mean) == 15

    output_file = os.path.join(tmp_path, "ZTF22abvdwik_eqwt_dynesty.npz")
    assert os.path.exists(output_file)

    params = np.load(output_file)["arr_0"]
    ## could be between ~600 and ~800, and can vary based on hardware.
    assert 600 <= len(params) <= 800

    # Check that the same means the same order of magnitude (within 50% relative value).
    # Despite setting the the random seed, we still need to account (so far) unexplained
    # additional variations.
    expected = [1035.0, 0.005, 13.5, -4.8, 4.0, 23.4, 0.03, 1.1, 1.0, 1.0, 1.0, 0.96, 0.56, 0.87, -5.43]
    assert len(expected) == len(sample_mean)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))


def test_run_curve_fit(single_ztf_lightcurve_compressed, tmp_path):
    """Change detection test on returned curve fit."""
    results = run_curve_fit(single_ztf_lightcurve_compressed, tmp_path, plot=True)

    expected_g = [1.006256, 4.13227076e-13, 0.86950, 1.196358, 1.236848, 0.912510]
    expected_r = [1082.44, 0.01575061, 1.297617, -4.5240395, 0.507503, 1.2562]
    assert np.all(np.isclose(results["g"], expected_g, rtol=0.01))
    assert np.all(np.isclose(results["r"], expected_r, rtol=0.01))

    filepath = os.path.join(tmp_path, "ZTF22abvdwik.png")
    assert os.path.exists(filepath)
