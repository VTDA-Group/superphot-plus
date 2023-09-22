import os

import numpy as np
import pytest

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.model.config import ModelConfig
from superphot_plus.utils import (
    calc_accuracy,
    calculate_log_likelihood,
    calculate_mse,
    calculate_neg_chi_squareds,
    f1_score,
    flux_model,
    get_band_extinctions,
    get_numpyro_cube,
    get_session_metrics,
    log_metrics_to_tensorboard,
    params_valid,
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


def test_get_numpyro_cube(ztf_priors):
    """Test converting numpyro param dict to an array of all
    sampled parameter vectors.
    """
    dummy_param_dict = generate_dummy_posterior_sample_dict(batch=False)
    cube, ordered_bands = get_numpyro_cube(
        dummy_param_dict,
        1e3,
        ref_band=ztf_priors.reference_band,
        ordered_bands=ztf_priors.ordered_bands,
    )

    assert cube.shape == (20, 14)
    assert len(ordered_bands) == 2
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
    with pytest.raises(ValueError) as err:
        _ = calculate_log_likelihood(cube, lc_empty, bands, "r")
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


def test_get_session_metrics():
    """Checks that we compute the correct train session metrics."""

    # Metrics for 2 folds, 5 epochs

    # min_val_loss -> 0.75, val_acc -> 0.76
    metrics_fold_1 = [
        [0.81, 0.88, 0.78, 0.79, 0.79],  # train_acc
        [0.93, 1.11, 0.87, 1.03, 0.56],  # train_loss
        [0.86, 0.81, 0.75, 0.71, 0.76],  # val_acc
        [0.93, 1.11, 0.87, 1.03, 0.75],  # val_loss
    ]

    # min_val_loss -> 0.59, val_acc -> 0.80
    metrics_fold_2 = [
        [0.77, 0.78, 0.73, 0.75, 0.74],  # train_acc
        [1.03, 0.32, 0.84, 1.28, 1.57],  # train_loss
        [0.81, 0.84, 0.80, 0.82, 0.86],  # val_acc
        [0.74, 0.75, 0.59, 0.60, 0.61],  # val_loss
    ]

    avg_val_loss, avg_val_acc = get_session_metrics(metrics=(metrics_fold_1, metrics_fold_2))

    assert np.mean([0.75, 0.59]) == avg_val_loss
    assert np.mean([0.76, 0.80]) == avg_val_acc


def test_log_metrics_to_tensorboard(tmp_path):
    """Checks that SummaryWriter is writing the metrics to disk."""

    trial_id = "test-run"

    # Metrics for 2 folds, 5 epochs
    metrics_fold_1 = [
        [0.81, 0.88, 0.78, 0.79, 0.79],  # train_acc
        [0.93, 1.11, 0.87, 1.03, 0.56],  # train_loss
        [0.86, 0.81, 0.75, 0.71, 0.76],  # val_acc
        [0.93, 1.11, 0.87, 1.03, 0.75],  # val_loss
    ]
    metrics_fold_2 = [
        [0.77, 0.78, 0.73, 0.75, 0.74],  # train_acc
        [1.03, 0.32, 0.84, 1.28, 1.57],  # train_loss
        [0.81, 0.84, 0.80, 0.82, 0.86],  # val_acc
        [0.74, 0.75, 0.59, 0.60, 0.61],  # val_loss
    ]

    avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs = log_metrics_to_tensorboard(
        metrics=(metrics_fold_1, metrics_fold_2),
        config=ModelConfig(num_epochs=5),
        trial_id=trial_id,
        base_dir=tmp_path,
    )

    assert np.array_equal(
        avg_train_losses,
        np.array(
            [
                np.mean([0.93, 1.03]),
                np.mean([1.11, 0.32]),
                np.mean([0.87, 0.84]),
                np.mean([1.03, 1.28]),
                np.mean([0.56, 1.57]),
            ]
        ),
    )
    assert np.array_equal(
        avg_train_accs,
        np.array(
            [
                np.mean([0.81, 0.77]),
                np.mean([0.88, 0.78]),
                np.mean([0.78, 0.73]),
                np.mean([0.79, 0.75]),
                np.mean([0.79, 0.74]),
            ]
        ),
    )
    assert np.array_equal(
        avg_val_losses,
        np.array(
            [
                np.mean([0.93, 0.74]),
                np.mean([1.11, 0.75]),
                np.mean([0.87, 0.59]),
                np.mean([1.03, 0.60]),
                np.mean([0.75, 0.61]),
            ]
        ),
    )
    assert np.array_equal(
        avg_val_accs,
        np.array(
            [
                np.mean([0.86, 0.81]),
                np.mean([0.81, 0.84]),
                np.mean([0.75, 0.80]),
                np.mean([0.71, 0.82]),
                np.mean([0.76, 0.86]),
            ]
        ),
    )
    assert os.path.exists(os.path.join(tmp_path, trial_id))
    assert os.path.exists(os.path.join(tmp_path, trial_id, "config.yaml"))
