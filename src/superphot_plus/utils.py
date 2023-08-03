import os

import extinction
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.config import config
from dustmaps.sfd import SFDQuery

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.sfd import dust_filepath


def get_band_extinctions(ra, dec):
    """Get g- and r-band extinctions in magnitudes for a single
    supernova lightcurve based on right ascension (RA) and declination
    (DEC).

    Parameters
    ----------
    ra : float
        The right ascension of the object of interest, in degrees.
    dec : float
        The declination of the object of interest, in degrees.

    Returns
    -------
    ext_list : np.ndarray
        A list of extinction magnitudes for the given coordinates, in
        the g- and r-bands.
    """
    config["data_dir"] = dust_filepath
    sfd = SFDQuery()

    # First look up the amount of mw dust at this location
    coords = SkyCoord(ra, dec, frame="icrs", unit="deg")
    Av_sfd = 2.742 * sfd(coords)  # from https://dustmaps.readthedocs.io/en/latest/examples.html

    # for gr, the was are:
    band_wvs = 1.0 / (0.0001 * np.asarray([4741.64, 6173.23]))  # in inverse microns

    # Now figure out how much the magnitude is affected by this dust
    ext_list = extinction.fm07(band_wvs, Av_sfd, unit="invum")  # in magnitudes

    return ext_list


def calc_accuracy(pred_classes, test_labels):
    """Calculates the accuracy of the random forest after predicting all
    classes.

    Parameters
    ----------
    pred_classes : np.ndarray of int
        Classes predicted by MLP.
    test_labels : np.ndarray of int
        True spectroscopic classes.

    Returns
    -------
    Accuracy: float
        The percentage of the predictions that are correct.

    Raises
    ------
    ValueError
        If the pred_classes or test_labels arrays are empty or if they
        are of mismatched sizes.
    """
    num_total = len(pred_classes)
    if num_total == 0:
        raise ValueError("Empty array provided to calc_accuracy.")
    if num_total != len(test_labels):
        raise ValueError(f"Array size mismatch for calc_accuracy {num_total} vs {len(test_labels)}.")

    num_correct = np.sum(np.where(pred_classes == test_labels, 1, 0))
    return num_correct / num_total


def f1_score(pred_classes, true_classes, class_average=False):
    """Calculates the F1 score for the classifier. If
    class_average=True, then the macro-F1 is used. Else, uses the
    weighted-F1 score.

    Parameters
    ----------
    pred_classes : np.ndarray of int
        Classes predicted by MLP.
    true_classes : np.ndarray of int
        True spectroscopic classes.
    class_average : bool, optional
        Determines whether F1 score is weighted equally for each class,
        or by number of samples per class. Defaults to False.

    Returns
    -------
    float
        The calculated F1 score.
    """
    samples_per_class = {}
    for c in true_classes:
        if c in samples_per_class:
            samples_per_class[c] += 1
        else:
            samples_per_class[c] = 1

    f1_sum = 0.0
    for c in samples_per_class:
        tp = len(pred_classes[(pred_classes == c) & (true_classes == c)])
        purity = tp / len(pred_classes[pred_classes == c])
        completeness = tp / len(true_classes[true_classes == c])
        f1 = 2.0 * purity * completeness / (purity + completeness)
        if class_average:
            f1_sum += f1
        else:
            f1_sum += samples_per_class[c] * f1
    if class_average:
        return f1_sum / len(samples_per_class.keys())

    return f1_sum / len(true_classes)


def convert_mags_to_flux(m, merr, zp):
    """Converts magnitudes to flux.

    Parameters
    ----------
    m : array-like
        The magnitudes.
    merr : array-like
        The error in magnitudes.
    zp : float
        The zero point in the magnitude system.

    Returns
    -------
    fluxes, flux_unc : tuple of numpy array
        The calculated fluxes and their uncertainties.
    """
    fluxes = 10.0 ** (-1.0 * (m - zp) / 2.5)
    flux_unc = np.log(10.0) / 2.5 * fluxes * merr
    return fluxes, flux_unc


def flux_model(cube, t_data, b_data, ordered_bands, ref_band):
    """Given "cube" of fit parameters, returns the flux measurements for
    a given set of time and band data.

    Parameters
    ----------
    cube : array-like
        The cube of fit parameters.
    t_data : array-like
        The time data.
    b_data : array-like
        The band data.
    ordered_bands : array-like
        The band names in the cube's parameter order.
    ref_band : str
        The base/reference band which all other values are multiples of.

    Returns
    -------
    f_model : numpy array
        The flux model for the given set of time and band data.
    """
    b_data = np.array(b_data)
    ref_band_idx = np.argmax(ref_band == np.array(ordered_bands))
    start_idx = ref_band_idx * 7

    A, beta, gamma, t0, tau_rise, tau_fall, es = cube[
        start_idx : start_idx + 7
    ]  # pylint: disable=unused-variable
    phase = t_data - t0
    f_model = (
        A / (1.0 + np.exp(-phase / tau_rise)) * (1.0 - beta * gamma) * np.exp((gamma - phase) / tau_fall)
    )
    f_model[phase < gamma] = (
        A / (1.0 + np.exp(-phase[phase < gamma] / tau_rise)) * (1.0 - beta * phase[phase < gamma])
    )

    for band_idx, ordered_band in enumerate(ordered_bands):
        if ordered_band == ref_band:
            continue
        start_idx = 7 * band_idx
        A_b = A * cube[start_idx]
        beta_b = beta * cube[start_idx + 1]
        gamma_b = gamma * cube[start_idx + 2]
        t0_b = t0 * cube[start_idx + 3]
        tau_rise_b = tau_rise * cube[start_idx + 4]
        tau_fall_b = tau_fall * cube[start_idx + 5]

        inc_band_ix = b_data == ordered_band
        phase_b = (t_data - t0_b)[inc_band_ix]
        phase_b2 = (t_data - t0_b)[inc_band_ix & (t_data - t0_b < gamma_b)]

        f_model[inc_band_ix] = (
            A_b
            / (1.0 + np.exp(-phase_b / tau_rise_b))
            * (1.0 - beta_b * gamma_b)
            * np.exp((gamma_b - phase_b) / tau_fall_b)
        )
        f_model[inc_band_ix & (t_data - t0_b < gamma_b)] = (
            A_b / (1.0 + np.exp(-phase_b2 / tau_rise_b)) * (1.0 - phase_b2 * beta_b)
        )

    return f_model


def calculate_neg_chi_squareds(cubes, t, f, ferr, b, ordered_bands=["r", "g"], ref_band="r"):
    """Gets the negative chi-squared of posterior fits from the model
    parameters and original data files.

    Parameters
    ----------
    names : list of str
        The names of the objects.
    fit_dir : str
        The directory where the fit files are located.
    data_dirs : list of str
        The directories where the data files are located.
    ordered_bands : list of str
        Bands in order they appear in cubes. Defaults to ZTF band order.
    ref_band : str
        Base/reference band. Defaults to 'r'.

    Returns
    -------
    log_likelihoods : np.ndarray
        The log likelihoods for each object.
    """
    model_f = np.array(
        [flux_model(cube, t, b, ordered_bands, ref_band) for cube in cubes]
    )  # in future, maybe vectorize flux_model
    extra_sigma_arr = np.ones((len(cubes), len(t))) * np.max(f[b == "r"]) * cubes[:, 6][:, np.newaxis]
    extra_sigma_arr[:, b == "g"] *= cubes[:, -2][:, np.newaxis]
    sigma_sq = extra_sigma_arr**2 + ferr**2

    log_likelihoods = np.sum(
        np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq)) - 0.5 * (f - model_f) ** 2 / sigma_sq, axis=1
    ) / len(t)

    return log_likelihoods
