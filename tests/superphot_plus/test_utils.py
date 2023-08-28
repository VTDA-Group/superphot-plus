import numpy as np
import pytest

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.utils import (
    calc_accuracy,
    calculate_neg_chi_squareds,
    f1_score,
    get_band_extinctions,
    get_numpyro_cube,
    params_valid,
    flux_model,
    calculate_log_likelihood,
    calculate_mse,
)


def generate_dummy_posterior_sample_dict(batch=False):
    """Create a posterior sample dictionary for r and g bands with random values"""
    param_list = [
        "logA",
        "beta",
        "log_gamma",
        "t0",
        "log_tau_rise",
        "log_tau_fall",
        "log_extra_sigma",
        "A_g",
        "beta_g",
        "gamma_g",
        "t0_g",
        "tau_rise_g",
        "tau_fall_g",
        "extra_sigma_g",
    ]
    return {
        param: np.random.rand(3, 20) if batch else np.random.rand(1, 20).flatten() for param in param_list
    }


def test_calc_accuracy() -> None:
    truth = np.array([1, 1, 0, 0, 2])

    # Everything correct.
    accuracy1 = calc_accuracy(np.array([1, 1, 0, 0, 2]), truth)
    assert pytest.approx(accuracy1) == 1.0

    # 80% correct.
    accuracy2 = calc_accuracy(np.array([1, 0, 0, 0, 2]), truth)
    assert pytest.approx(accuracy2) == 0.8

    # 60% correct.
    accuracy3 = calc_accuracy(np.array([1, 0, 0, 0, 1]), truth)
    assert pytest.approx(accuracy3) == 0.6

    # Check an empty array.
    with pytest.raises(ValueError):
        _ = calc_accuracy(np.array([]), np.array([]))

    # Check a array size mismatch.
    with pytest.raises(ValueError):
        _ = calc_accuracy(np.array([1, 2, 3]), truth)


def test_f1_score() -> None:
    # One true positive of each class and one mislabel.
    s = f1_score(np.array([1, 1, 0]), np.array([0, 1, 0]), True)
    assert pytest.approx(s) == 2.0 / 3.0
    s = f1_score(np.array([1, 1, 0]), np.array([0, 1, 0]), False)
    assert pytest.approx(s) == 2.0 / 3.0

    # F1s for each class will be: [4/5, 8/9, 1, 1, 1]
    s = f1_score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4]),
        np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]),
        True,
    )
    assert pytest.approx(s) == (4.0 / 5.0 + 6.0 / 7.0 + 1.0 + 1.0 + 1.0) / 5.0

    s = f1_score(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4]),
        np.array([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]),
        False,
    )
    assert pytest.approx(s) == (2.0 * (4.0 / 5.0) + 4.0 * (6.0 / 7.0) + 2.0 + 2.0 + 2.0) / 12.0

    # Test a case where there are no predictions for a class.
    s = f1_score(
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 1, 1]),
        True,
    )
    assert pytest.approx(s) == (1.0 / 3.0)


def test_get_band_extinctions() -> None:
    """This is currently a change detection test where we are just confirming
    the function runs correctly returns the same value as it used to.
    """
    ext_list = get_band_extinctions(0.0, 10.0, [4741.64, 6173.23])
    assert np.all(ext_list == pytest.approx([0.3133, 0.2202], 0.01))


def test_params_valid():
    """Test the params_valid function."""

    # Prior ZTF values are valid.
    assert params_valid(0.0052, 1.1391, 0.599, 1.4296)

    # Invalid combinations
    assert not params_valid(1.0, 1.1391, 0.599, 2.0)
    assert not params_valid(1.0, 2.0, 0.599, 1.0)
    assert not params_valid(1.0, 0.0, 1.0, 2.1)


def test_get_numpyro_cube():
    """Test converting numpyro param dict to an array of all
    sampled parameter vectors.
    """
    dummy_param_dict = generate_dummy_posterior_sample_dict(batch=False)
    cube, aux_bands = get_numpyro_cube(dummy_param_dict, 1e3)

    assert cube.shape == (20, 14)
    assert len(aux_bands) == 1
    assert np.mean(cube[:, 1]) == np.mean(dummy_param_dict["beta"])


def test_calculate_log_likelihood_simple():
    """Do a very simple test where we can compute the LL by hand."""
    tdata = np.array([10.60606061, 12.12121212])
    bands = ["r", "g"]
    bdata = np.array(bands)
    edata = np.array([0.01, 0.05])

    # Generate clean fluxes from the model with extra sigma = 0.0
    cube = np.array(
        [
            3.12033307,
            0.00516744388,
            14.1035747,
            -49.0239436,
            6.31504200,
            30.7416132,
            0.0,
            1.33012285,
            1.04407290,
            1.01418818,
            1.00009386,
            0.980085932,
            0.573802443,
            0.0,
        ]
    )
    ftrue = flux_model(cube, tdata, bdata, bands, "r")
    assert np.allclose(ftrue, np.array([0.65786904, 0.26904324]))

    # r+0.01 and b-0.025
    fdata = np.array([0.66786904, 0.24404324])

    # Compute the true probabilities and the log likelihood via the gaussian equation.
    sigma_sq = np.square(edata)
    probs = (
        1.0 / np.sqrt(2 * np.pi * sigma_sq) * np.exp(-0.5 * np.divide(np.power(fdata - ftrue, 2), sigma_sq))
    )
    LogLTrue = np.sum(np.log(probs))

    lc1 = Lightcurve(tdata, fdata, edata, bdata)
    LogL1 = calculate_log_likelihood(cube, lc1, bands, "r")
    assert LogL1 == pytest.approx(LogLTrue)


def test_calculate_log_likelihood():
    num_observations = 100
    tdata = np.linspace(-50.0, 100.0, num_observations)
    bands = ["r", "g"]
    bdata = np.array([bands[i % 2] for i in range(num_observations)])
    edata = np.array([0.01] * num_observations)

    # Generate clean fluxes from the model.
    cube = np.array(
        [
            3.12033307,
            0.00516744388,
            14.1035747,
            -49.0239436,
            6.31504200,
            30.7416132,
            0.0249219755,
            1.33012285,
            1.04407290,
            1.01418818,
            1.00009386,
            0.980085932,
            0.573802443,
            0.948438711,
        ]
    )
    fdata = flux_model(cube, tdata, bdata, bands, "r")

    lc1 = Lightcurve(tdata, fdata, edata, bdata)
    LogL1 = calculate_log_likelihood(cube, lc1, bands, "r")
    assert LogL1 == pytest.approx(183.17363309388816)  # Change detection only

    # Test noisy models
    for diff in [-0.1, 0.1, 0.5, 1.0]:
        lc2 = Lightcurve(tdata, fdata + diff, edata, bdata)
        test_ll = calculate_log_likelihood(cube, lc2, bands, "r")
        assert LogL1 > test_ll

    # Test error conditions.
    with pytest.raises(ValueError) as err:
        _ = calculate_log_likelihood(cube, lc1, bands, "u")
    assert str(err.value) == "Reference band not included in unique_bands."

    with pytest.raises(ValueError) as err:
        _ = calculate_log_likelihood(cube, lc1, ["r"], "r")
    assert str(err.value) == "Size mismatch with curve parameters. Expected 7. Found 14."

    lc_empty = Lightcurve(np.array([]), np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError):
        _ = calculate_log_likelihood(cube, lc_empty, ["r", "g"], "r")
    assert str(err.value) == "Empty light curve provided."


def test_calculate_mse():
    num_observations = 100
    tdata = np.linspace(-50.0, 100.0, num_observations)
    bands = ["r", "g"]
    bdata = np.array([bands[i % 2] for i in range(num_observations)])
    edata = np.array([0.01] * num_observations)

    # Generate clean fluxes from the model.
    cube = np.array(
        [
            3.12033307,
            0.00516744388,
            14.1035747,
            -49.0239436,
            6.31504200,
            30.7416132,
            0.0249219755,
            1.33012285,
            1.04407290,
            1.01418818,
            1.00009386,
            0.980085932,
            0.573802443,
            0.948438711,
        ]
    )
    fdata = flux_model(cube, tdata, bdata, bands, "r")
    lc1 = Lightcurve(tdata, fdata, edata, bdata)
    mse1 = calculate_mse(cube, lc1, bands, "r")
    assert mse1 == pytest.approx(0.0)

    # Test noisy models
    for diff in [-0.1, 0.1, 0.5, 1.0]:
        lc2 = Lightcurve(tdata, fdata + diff, edata, bdata)
        test_mse = calculate_mse(cube, lc2, bands, "r")
        assert test_mse == pytest.approx(diff * diff)

    # Test error conditions.
    with pytest.raises(ValueError) as err:
        _ = calculate_mse(cube, lc1, bands, "u")
    assert str(err.value) == "Reference band not included in unique_bands."

    with pytest.raises(ValueError) as err:
        _ = calculate_mse(cube, lc1, ["r"], "r")
    assert str(err.value) == "Size mismatch with curve parameters. Expected 7. Found 14."

    lc_empty = Lightcurve(np.array([]), np.array([]), np.array([]), np.array([]))
    with pytest.raises(ValueError) as err:
        _ = calculate_mse(cube, lc_empty, ["r", "g"], "r")
    assert str(err.value) == "Empty light curve provided."


def test_neg_chi_squareds(single_ztf_lightcurve_compressed, test_data_dir, single_ztf_sn_id):
    """This is currently a change detection test where we are just confirming
    the function runs correctly returns the same value as it used to.
    """
    posts = get_posterior_samples(single_ztf_sn_id, fits_dir=test_data_dir, sampler="dynesty")
    lc = Lightcurve.from_file(single_ztf_lightcurve_compressed)

    sn_data = [lc.times, lc.fluxes, lc.flux_errors, lc.bands]
    result = calculate_neg_chi_squareds(posts, *sn_data)
    assert np.isclose(np.mean(result), -5.43, rtol=0.1)
