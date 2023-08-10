import numpy as np
import pytest

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.utils import calc_accuracy, f1_score, get_band_extinctions, get_numpyro_cube, calc_neg_chi_squareds


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


def test_get_band_extinctions() -> None:
    """This is currently a change detection test where we are just confirming
    the function runs correctly returns the same value as it used to.
    """
    ext_list = get_band_extinctions(0.0, 10.0, [4741.64, 6173.23])
    assert np.all(ext_list == pytest.approx([0.3133, 0.2202], 0.01))


def test_get_numpyro_cube():
    """Test converting numpyro param dict to an array of all
    sampled parameter vectors.
    """
    dummy_param_dict = generate_dummy_posterior_sample_dict(batch=False)
    cube, aux_bands = get_numpyro_cube(dummy_param_dict, 1e3)

    assert cube.shape == (20, 14)
    assert len(aux_bands) == 1
    assert np.mean(cube[:, 1]) == np.mean(dummy_param_dict["beta"])


def test_neg_chi_squareds(single_ztf_lightcurve_compressed, test_data_dir, single_ztf_sn_id):
    """This is currently a change detection test where we are just confirming
    the function runs correctly returns the same value as it used to.
    """
    posts = get_posterior_samples(single_ztf_sn_id, fits_dir=test_data_dir, sampler="dynesty")
    lc = Lightcurve.from_file(single_ztf_lightcurve_compressed)

    sn_data = [lc.times, lc.fluxes, lc.flux_errors, lc.bands]
    result = calculate_neg_chi_squareds(posts, *sn_data)
    assert np.isclose(np.mean(result), -5.43, rtol=0.1)
