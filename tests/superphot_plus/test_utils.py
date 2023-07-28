import numpy as np
import pytest

from superphot_plus.utils import (
    calc_accuracy,
    f1_score,
    flux_model,
    get_band_extinctions,
    calculate_neg_chi_squareds,
)


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


def test_get_band_extinctions() -> None:
    """This is currently a change detection test where we are just confirming
    the function runs correctly returns the same value as it used to.
    """
    ext_list = get_band_extinctions(0.0, 10.0)
    assert np.all(ext_list == pytest.approx([0.3133, 0.2202], 0.01))


def test_neg_chi_squareds(test_data_dir, single_ztf_sn_id):
    """This is currently a change detection test where we are just confirming
    the function runs correctly returns the same value as it used to.
    """
    result = calculate_neg_chi_squareds([single_ztf_sn_id], fit_dir=test_data_dir, data_dirs=[test_data_dir])
    assert np.all(np.isclose(result, [-38.4], rtol=0.1))


def test_flux_model():
    num_times = 100
    cube = np.array(
        [
            1035.0,  # A_r
            0.005,
            13.5,  # gamma_r
            -4.8,  # t0_r
            4.0,
            23.4,
            0.03,
            1.1,  # A_g = A_r * 1.1
            1.0,
            1.0,  # gamma_g = 1.0 * gamma_r
            1.0,  # t0_g = t0_r * 1.0
            0.96,
            0.56,
            0.87,
        ]
    )
    times = np.linspace(-20.0, 30.0, num_times)
    band_options = ["r", "g"]
    bands = np.array([band_options[i % 2] for i in range(num_times)])  # Alternating r and g

    fluxes = flux_model(cube, times, bands)
    assert len(fluxes) == num_times

    # Check that the peak flux seen is <= A
    fluxes_r = fluxes[bands == "r"]
    fluxes_g = fluxes[bands == "g"]
    assert np.all(fluxes_r <= cube[0])
    assert np.all(fluxes_g <= cube[7] * cube[0])

    # Check that the peak happens near t0 + gamma for both bands. It might happen
    # +/- one index depending on the slops of the curves and how close the peak time
    # is to a sampled timestep.
    times_r = times[bands == "r"]
    times_g = times[bands == "g"]
    brightest_ind_r = np.argmin(np.abs(times_r - (cube[3] + cube[2])))
    brightest_ind_g = np.argmin(np.abs(times_g - (cube[10] * cube[3] + cube[9] * cube[2])))
    assert np.abs(brightest_ind_r - np.argmax(fluxes_r)) <= 1
    assert np.abs(brightest_ind_g - np.argmax(fluxes_g)) <= 1

    # Check that the curve is monotonlically increasing before the peak
    for i in range(1, brightest_ind_r):
        assert fluxes_r[i] > fluxes_r[i - 1]
    for i in range(1, brightest_ind_g):
        assert fluxes_g[i] > fluxes_g[i - 1]

    # and monotonically decreasing after the peak
    for i in range(brightest_ind_r + 1, len(fluxes_r)):
        assert fluxes_r[i] < fluxes_r[i - 1]
    for i in range(brightest_ind_g + 1, len(fluxes_g)):
        assert fluxes_g[i] < fluxes_g[i - 1]
