import os

import numpy as np
import pytest
from snapi import LightCurve

from superphot_plus.config import SuperphotConfig
from superphot_plus.supernova_class import SupernovaClass

from superphot_plus.utils import (
    calc_accuracy,
    f1_score,
    #flux_model,
    get_session_metrics,
    log_metrics_to_tensorboard,
    params_valid,
    clip_lightcurve_end,
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

def test_params_valid(ztf_priors):
    """Test the params_valid function."""

    means = ztf_priors.dataframe['mean'].to_numpy()
    means[8] = means[1]*means[8]
    means[9] = 10**(means[2] + means[9])
    means[11] = 10**(means[4] + means[11])
    means[12] = 10**(means[5] + means[12])

    means[2] = 10**means[2]
    means[4] = 10**means[4]
    means[5] = 10**means[5]

    # Prior ZTF values are valid.
    assert params_valid(means)
    assert params_valid(means[7:])

    means[1] = 1.0
    means[2] = 10**1.1391
    means[4] = 10**0.599
    means[5] = 10**2.0
    assert not params_valid(means)

    means[2] = 10**2.0
    means[4] = 10**0.599
    means[5] = 10**1.0
    assert not params_valid(means)

    means[2] = 10**0.0
    means[4] = 10**1.0
    means[5] = 10**2.1
    assert not params_valid(means)


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
        config=SuperphotConfig(num_epochs=5),
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

def test_clip_lightcurve_end():
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

    out_lc = clip_lightcurve_end(
        LightCurve.from_arrays(
            np.array(times),
            np.array(fluxes),
            np.array(errors),
            np.array(bands),
            phased=True
        )
    )

    t_clip = out_lc.phase

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
