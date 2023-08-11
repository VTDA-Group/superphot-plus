import extinction
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.config import config
from dustmaps.sfd import SFDQuery

from superphot_plus.sfd import dust_filepath


def get_band_extinctions(ra, dec, wvs):
    """Get g- and r-band extinctions in magnitudes for a single
    supernova lightcurve based on right ascension (RA) and declination
    (DEC).

    Parameters
    ----------
    ra : float
        The right ascension of the object of interest, in degrees.
    dec : float
        The declination of the object of interest, in degrees.
    wvs : list or np.ndarray
        Array of wavelengths, in angstroms.


    Returns
    -------
    ext_dict : Dict
        A dictionary mapping bands to extinction magnitudes for the given coordinates.
    """
    config["data_dir"] = dust_filepath
    sfd = SFDQuery()

    # First look up the amount of mw dust at this location
    coords = SkyCoord(ra, dec, frame="icrs", unit="deg")
    Av_sfd = 2.742 * sfd(coords)  # from https://dustmaps.readthedocs.io/en/latest/examples.html

    band_wvs = 1.0 / (0.0001 * np.asarray(wvs))  # in inverse microns

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
    for true_class in true_classes:
        if true_class in samples_per_class:
            samples_per_class[true_class] += 1
        else:
            samples_per_class[true_class] = 1

    f1_sum = 0.0
    for true_class, count in samples_per_class.items():
        true_positive = len(pred_classes[(pred_classes == true_class) & (true_classes == true_class)])
        purity = true_positive / len(pred_classes[pred_classes == true_class])
        completeness = true_positive / len(true_classes[true_classes == true_class])
        f_1 = 2.0 * purity * completeness / (purity + completeness)
        if class_average:
            f1_sum += f_1
        else:
            f1_sum += count * f_1
    if class_average:
        return f1_sum / len(samples_per_class.keys())

    return f1_sum / len(true_classes)


def convert_mags_to_flux(mag, merr, zero_point):
    """Converts magnitudes to flux.

    Parameters
    ----------
    mag : array-like
        The magnitudes.
    merr : array-like
        The error in magnitudes.
    zero_point : float
        The zero point in the magnitude system.

    Returns
    -------
    fluxes, flux_unc : tuple of numpy array
        The calculated fluxes and their uncertainties.
    """
    fluxes = 10.0 ** (-1.0 * (mag - zero_point) / 2.5)
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

    amp, beta, gamma, t_0, tau_rise, tau_fall, _ = cube[start_idx : start_idx + 7]
    phase = t_data - t_0
    f_model = (
        amp / (1.0 + np.exp(-phase / tau_rise)) * (1.0 - beta * gamma) * np.exp((gamma - phase) / tau_fall)
    )
    f_model[phase < gamma] = (
        amp / (1.0 + np.exp(-phase[phase < gamma] / tau_rise)) * (1.0 - beta * phase[phase < gamma])
    )

    for band_idx, ordered_band in enumerate(ordered_bands):
        if ordered_band == ref_band:
            continue
        start_idx = 7 * band_idx
        amp_b = amp * cube[start_idx]
        beta_b = beta * cube[start_idx + 1]
        gamma_b = gamma * cube[start_idx + 2]
        t0_b = t_0 * cube[start_idx + 3]
        tau_rise_b = tau_rise * cube[start_idx + 4]
        tau_fall_b = tau_fall * cube[start_idx + 5]

        inc_band_ix = b_data == ordered_band
        phase_b = (t_data - t0_b)[inc_band_ix]
        phase_b2 = (t_data - t0_b)[inc_band_ix & (t_data - t0_b < gamma_b)]

        f_model[inc_band_ix] = (
            amp_b
            / (1.0 + np.exp(-phase_b / tau_rise_b))
            * (1.0 - beta_b * gamma_b)
            * np.exp((gamma_b - phase_b) / tau_fall_b)
        )
        f_model[inc_band_ix & (t_data - t0_b < gamma_b)] = (
            amp_b / (1.0 + np.exp(-phase_b2 / tau_rise_b)) * (1.0 - phase_b2 * beta_b)
        )

    return f_model


def params_valid(beta, gamma, tau_rise, tau_fall):
    """Check if parameters are valid given certain model constraints.

    Parameters
    ----------
    beta : float
        Parameter beta.
    gamma : float
        Parameter gamma.
    tau_rise : float
        Parameter tau_rise.
    tau_fall : float
        Parameter tau_fall.

    Returns
    -------
    bool
        True if parameters are valid, False otherwise.
    """
    if tau_fall > 1.0 / beta:
        return False

    if gamma > (1.0 - beta * tau_fall) / beta:
        return False

    if tau_rise * (1.0 + np.exp(gamma / tau_rise)) < tau_fall:
        return False

    return True


def get_numpyro_cube(params, max_flux, aux_bands=None):
    """
    Convert output param dict from numpyro sampler to match that
    of dynesty.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    max_flux : float
        Max flux of light curve
    aux_bands : array-like, optional
        The names of auxiliary bands, in order. If None or excluded,
        attempts to infer them from the dictionary.

    Returns
    ----------
    cube : np.ndarray
        Array of all equal-weight parameter draws
    aux_bands : np.ndarray
        Auxiliary bands, including those inferred if input arg was None.
    """
    if aux_bands is None:
        aux_bands = []
        for k in params:
            if k[:4] == "beta" and k != "beta":
                aux_bands.append(k[5:])

    logA, beta, log_gamma = params["logA"], params["beta"], params["log_gamma"]
    t0, log_tau_rise, log_tau_fall, log_extra_sigma = (
        params["t0"],
        params["log_tau_rise"],
        params["log_tau_fall"],
        params["log_extra_sigma"],
    )

    A = max_flux * 10**logA
    gamma = 10**log_gamma
    tau_rise = 10**log_tau_rise
    tau_fall = 10**log_tau_fall
    extra_sigma = 10**log_extra_sigma  # pylint: disable=unused-variable

    cube = [A, beta, gamma, t0, tau_rise, tau_fall, extra_sigma]

    for b in aux_bands:
        cube.extend(
            [
                params[f"A_{b}"],
                params[f"beta_{b}"],
                params[f"gamma_{b}"],
                params[f"t0_{b}"],
                params[f"tau_rise_{b}"],
                params[f"tau_fall_{b}"],
                params[f"extra_sigma_{b}"],
            ]
        )
    return np.array(cube).T, np.array(aux_bands)


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
