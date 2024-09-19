import os

import numpy as np
import pytest

from superphot_plus.config import SuperphotConfig
from superphot_plus.supernova_class import SupernovaClass

from superphot_plus.utils import (
    calc_accuracy,
    f1_score,
    #flux_model,
    get_numpyro_cube,
    get_session_metrics,
    log_metrics_to_tensorboard,
    params_valid,
    clip_lightcurve_end,
    import_labels_only,
    normalize_features
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

    means = ztf_priors.to_numpy().T[2]
    # Prior ZTF values are valid.
    assert params_valid(
        means[1], 10**means[2],
        10**means[4], 10**means[5]
    )
    assert params_valid(
        means[1]*10**means[8],
        10**(means[2] + means[9]),
        10**(means[4] + means[11]),
        10**(means[5] + means[12])
    )

    # Invalid combinations
    assert not params_valid(1.0, 10**1.1391, 10**0.599, 10**2.0)
    assert not params_valid(1.0, 10**2.0, 10**0.599, 10**1.0)
    assert not params_valid(1.0, 10**0.0, 10**1.0, 10**2.1)


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

def test_import_labels_only(tmp_path):
    """Test loading a file of labels and applying filters"""
    csv_file = os.path.join(tmp_path, "labels.csv")
    with open(csv_file, "w+", encoding="utf-8") as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=",")
        csv_writer.writerow(["NAME", "CLASS", "Z"])
        csv_writer.writerow(["ZTF_SN_1234", "SN Ic", 4.5])
        csv_writer.writerow(["ZTF_SN_1234", "SN IIn", 4.5])
        csv_writer.writerow(["ZTF_SN_4567", "SN Ib-Ca-rich", 5.6])

    ## With no weighted fits, we skip all of the inputs
    names, labels, redshifts = import_labels_only(
        [csv_file],
        SupernovaClass.all_classes(),
    )

    assert len(names) == 0
    assert len(labels) == 0
    assert len(redshifts) == 0

    ## Add one weighted fit file and we should pick up that label.
    fits_dir = os.path.join(tmp_path, "fits")
    os.makedirs(fits_dir, exist_ok=True)
    Path(os.path.join(fits_dir, "ZTF_SN_1234_eqwt.npz")).touch()

    names, labels, redshifts = import_labels_only(
        [csv_file],
        SupernovaClass.all_classes(),
        fits_dir=fits_dir
    )

    ## Should not include duplicate label.
    assert names == ["ZTF_SN_1234"]
    assert labels == ["SN Ibc"]
    assert redshifts == [4.5]

    ## Remove that class from the allowed types and we're back to nothing.
    names, labels, redshifts = import_labels_only(
        [csv_file], [SupernovaClass.SUPERLUMINOUS_SUPERNOVA_I], fits_dir=fits_dir
    )

    assert len(names) == 0
    assert len(labels) == 0
    assert len(redshifts) == 0

def test_normalize_features():
    # Feature #1: mean = 1.0, std ~= 0.81649658
    # Feature #2: mean = 0.75, std ~= 0.54006172
    # Feature #3: mean = 1.0, std == 0.0
    features = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [2.0, 1.25, 1.0]])
    expected = np.array(
        [
            [0.0, 0.25 / 0.54006172, 0.0],
            [-1.0 / 0.81649658, -0.75 / 0.54006172, 0.0],
            [1.0 / 0.81649658, 0.5 / 0.54006172, 0.0],
        ]
    )
    computed, mean, std = normalize_features(features)

    assert np.allclose(mean, [1.0, 0.75, 1.0])
    assert np.allclose(std, [0.81649658, 0.54006172, 0.0])
    assert np.allclose(computed, expected)
