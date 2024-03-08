import os
import shutil

import extinction
import pandas as pd
import numpy as np
import torch
from astropy.coordinates import SkyCoord
from dustmaps.config import config as dustmaps_config
from dustmaps.sfd import SFDQuery
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from scipy.special import lambertw
import jax.numpy as jnp
from numpyro.distributions import constraints

from superphot_plus.sfd import dust_filepath

def get_band_extinctions_from_mwebv(mwebv, wvs):
    """Get extinction list from MWEBV value and wavelengths.
    """
    Av_sfd = 2.742 * mwebv
    band_wvs = 1.0 / (0.0001 * np.asarray(wvs))  # in inverse microns

    # Now figure out how much the magnitude is affected by this dust
    ext_list = extinction.fm07(band_wvs, Av_sfd, unit="invum")  # in magnitudes

    return ext_list


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
    dustmaps_config["data_dir"] = dust_filepath
    sfd = SFDQuery()

    # First look up the amount of mw dust at this location
    coords = SkyCoord(ra, dec, frame="icrs", unit="deg")
      # from https://dustmaps.readthedocs.io/en/latest/examples.html
    mwebv = sfd(coords)
    return get_band_extinctions_from_mwebv(mwebv, wvs)


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
        if len(pred_classes[pred_classes == true_class]) == 0:
            f_1 = 0.0
        else:
            purity = true_positive / len(pred_classes[pred_classes == true_class])
            completeness = true_positive / len(true_classes[true_classes == true_class])

            if purity + completeness == 0:  # pragma: no cover
                f_1 = 0.0
            else:
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


def flux_model(cube, t_data, b_data, max_flux, ordered_bands, ref_band):
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

    amp = max_flux * 10**cube[start_idx]
    beta = cube[start_idx + 1]
    gamma = 10**cube[start_idx + 2]
    t_0 = cube[start_idx + 3]
    tau_rise, tau_fall, extra_sigma = 10**cube[start_idx + 4: start_idx + 7]
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
        amp_b = amp * 10**cube[start_idx]
        beta_b = beta * 10**cube[start_idx + 1]
        gamma_b = gamma * 10**cube[start_idx + 2]
        t0_b = t_0 + cube[start_idx + 3]
        tau_rise_b = tau_rise * 10**cube[start_idx + 4]
        tau_fall_b = tau_fall * 10**cube[start_idx + 5]

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

    if np.any(
        np.isnan([beta, gamma, tau_rise, tau_fall])
    ):
        return False

    # ensure dF2/dtheta < dF1/dtheta at gamma
    if gamma > (1.0 - beta * tau_fall) / beta:
        return False

    # ensure dF_rise/dtheta < 0 at gamma
    if np.exp(-gamma / tau_rise) * (1/tau_rise - beta - beta*gamma/tau_rise) > beta:
        return False

    return True

def villar_fit_constraint(x):
    beta, gamma, tau_rise, tau_fall = x
    return (
        jnp.maximum(gamma - (1.0 - beta * tau_fall) / beta, 0.) +
        jnp.maximum(jnp.exp(-gamma / tau_rise) * (1.0/beta - tau_rise - gamma) - tau_rise, 0.)
    )

    
def get_numpyro_cube(params, max_flux, ref_band, ordered_bands):
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
    logA, beta, log_gamma = params["logA"], params["beta"], params["log_gamma"]
    t0, log_tau_rise, log_tau_fall, log_extra_sigma = (
        params["t0"],
        params["log_tau_rise"],
        params["log_tau_fall"],
        params["log_extra_sigma"],
    )
    
    cube = []

    for b in ordered_bands:
        if b == ref_band:
            cube.extend(
            [
                logA, beta, log_gamma, t0,
                log_tau_rise, log_tau_fall, log_extra_sigma
            ]
        )
        else:
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
    return np.array(cube).T, np.array(ordered_bands)


def calculate_log_likelihood(cube, lightcurve, unique_bands, ref_band):
    """Calculate the log-likelihood of a single lightcurve.
    Copied from dynesty_sampler.py

    Parameters
    ----------
    cube : np.ndarray
        Array of parameters. Must be length 7 * B where B is
        the number of unique bands.
    lightcurve : Lightcurve
        The lightcurve object to evaluate.
    unique_bands : list
        A list of bands to use.
    ref_band : str
        The reference band.

    Returns
    -------
    logL : float
        Log-likelihood value.
    """
    if ref_band not in unique_bands:
        raise ValueError("Reference band not included in unique_bands.")
    if 7 * len(unique_bands) != len(cube):
        raise ValueError(
            f"Size mismatch with curve parameters. Expected {7 * len(unique_bands)}. Found {len(cube)}."
        )
    if len(lightcurve.times) == 0:
        raise ValueError("Empty light curve provided.")

    # Generate points from 'cube' for comparison.
    max_flux, _ = lightcurve.find_max_flux(band=ref_band)
    f_model = flux_model(
        cube, lightcurve.times, lightcurve.bands,
        max_flux, unique_bands, ref_band
    )
    extra_sigma_arr = np.ones(len(lightcurve.times)) * 10**cube[6] * max_flux
    for band_idx, ordered_band in enumerate(unique_bands):
        if ordered_band == ref_band:
            continue
        extra_sigma_arr[lightcurve.bands == ordered_band] *= 10**cube[7 * band_idx + 6]

    # Compute the loglikelihood based on the differences in flux between the observed
    # and generated.
    sigma_sq = lightcurve.flux_errors**2 + extra_sigma_arr**2

    logL = np.sum(
        np.log(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
        - 0.5 * (f_model - lightcurve.fluxes) ** 2 / sigma_sq
    )
    return logL


def calculate_mse(cube, lightcurve, unique_bands, ref_band):
    """Calculate the mean-square error for a lightcurve.

    Parameters
    ----------
    cube : np.ndarray
        Array of parameters. Must be length 7 * B where B is
        the number of unique bands.
    lightcurve : Lightcurve
        The lightcurve object to evaluate.
    unique_bands : list
        A list of bands to use.
    ref_band : str
        The reference band.

    Returns
    -------
    mse : float
        The mean square error of the predictions
    """
    if ref_band not in unique_bands:
        raise ValueError("Reference band not included in unique_bands.")
    if 7 * len(unique_bands) != len(cube):
        raise ValueError(
            f"Size mismatch with curve parameters. Expected {7 * len(unique_bands)}. Found {len(cube)}."
        )
    if len(lightcurve.times) == 0:
        raise ValueError("Empty light curve provided.")

    max_flux, _ = lightcurve.find_max_flux(band="r")
    # Generate points from 'cube' for comparison.
    
    f_model = flux_model(
        cube, lightcurve.times, lightcurve.bands,
        max_flux, unique_bands, ref_band
    )
    mse_sum = np.sum(np.square(f_model - lightcurve.fluxes))
    return mse_sum / len(lightcurve.times)


def calculate_chi_squareds(cubes, t, f, ferr, b, max_flux, ordered_bands=None, ref_band="r"):
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
    if ordered_bands is None:
        ordered_bands = ["r", "g"]
        
    ordered_bands = np.asarray(ordered_bands)
    
    model_f = np.array(
        [flux_model(cube, t, b, max_flux, ordered_bands, ref_band) for cube in cubes]
    )  # in future, maybe vectorize flux_model
    ref_band_idx = np.where(ordered_bands == ref_band)[0][0]
    extra_sigma_arr = np.ones((len(cubes), len(t))) * np.max(f[b == ref_band]) * 10**cubes[:, ref_band_idx + 6][:, np.newaxis]

    for i, ordered_band in enumerate(ordered_bands):
        if ordered_band == ref_band:
            continue
        extra_sigma_arr[:, b == ordered_band] *= 10**cubes[:, 7 * i + 6][:, np.newaxis]
    sigma_sq = extra_sigma_arr**2 + ferr**2
    red_chisq = np.sum((f[np.newaxis, :] - model_f) ** 2 / sigma_sq, axis=1) / len(t)

    return red_chisq


def create_dataset(features, labels):
    """Creates a PyTorch dataset object from numpy arrays.

    Parameters
    ----------
    features : np.ndarray
        The features array.
    labels : np.ndarray
        The labels array.
    idxs : np.ndarray, optional
        The indices array. Defaults to None.

    Returns
    -------
    torch.utils.data.TensorDataset
        The created dataset.
    """
    tensor_x = torch.tensor(features, dtype=torch.float, device='cpu')  # transform to torch tensor
    tensor_y = torch.tensor(labels, dtype=torch.int64, device='cpu')
    return TensorDataset(tensor_x, tensor_y)

def calculate_accuracy(y_pred, y):
    """Calculate the prediction accuracy.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted tensor.
    y : torch.Tensor
        The true tensor.

    Returns
    -------
    torch.Tensor
        The calculated accuracy.
    """
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def epoch_time(start_time, end_time):
    """Sets the time it takes for each epoch to train.

    Parameters
    ----------
    start_time : float
        The start time.
    end_time : float
        The end time.

    Returns
    -------
    tuple
        A tuple containing the elapsed minutes and elapsed seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_test_probabilities(
    obj_names,
    pred_probabilities,
    output_path,
    true_labels=None,
    target_label=None
):
    """Saves probabilities to a separate file for ROC curve generation.

    Parameters
    ----------
    output_filename : str
        The file name to save to.
    pred_probabilities : array-like
        The prediction probabilities.
    true_label : str or int
        The true label.
    output_dir: str
        Where to store the generated file.
    """
    if true_labels is None:
        true_labels = np.ones(len(obj_names)) * -1
        
    if target_label is None:
        #TODO: de-hard-code the headers
        df_dict = {
            'Name': obj_names,
            'Label': true_labels,
            'pSNIa': pred_probabilities[:,0],
            'pSNII': pred_probabilities[:,1],
            'pSNIIn': pred_probabilities[:,2],
            'pSLSNI': pred_probabilities[:,3],
            'pSNIbc': pred_probabilities[:,4]
        }
    else:
        df_dict = {
            'Name': obj_names,
            'Label': true_labels,
            f'p{target_label.replace(" ", "")}': pred_probabilities[:,1],
            'pOther': pred_probabilities[:,0],
        }
    
    df = pd.DataFrame(df_dict)
    df.to_csv(output_path, index=False)
    

def write_metrics_to_file(
    config,
    true_classes,
    pred_classes,
    prob_above_07,
):
    """Calculates the accuracy and f1 score metrics for the
    test set and outputs them to a log file.

    Parameters
    ----------
    config : ModelConfig
        The configuration of the model used for evaluation.
    true_classes : np.ndarray
        The ground truth for the test ZTF objects.
    pred_classes : np.ndarray
        The predicted classes for the test ZTF objects.
    prob_above_07 : np.ndarray
        Indicates which predictions had a 70% confidence.
    log_file : str
        The file where the metrics information will be written.
    """
    test_acc = calc_accuracy(pred_classes, true_classes)
    test_f1_score = f1_score(pred_classes, true_classes, class_average=True)

    with open(config.log_fn, "a+", encoding="utf-8") as the_file:
        the_file.write(str(config.goal_per_class) + " samples per class\n")
        the_file.write(
            str(config.neurons_per_layer)
            + " neurons per each of "
            + str(config.num_hidden_layers)
            + " layers\n"
        )
        the_file.write(str(config.num_epochs) + " epochs\n")
        the_file.write(
            "HOW MANY CERTAIN " + str(len(true_classes)) + " " + str(len(true_classes[prob_above_07])) + "\n"
        )
        the_file.write(f"MLP class-averaged F1-score: {test_f1_score:.04f}\n")
        the_file.write(f"Accuracy: {test_acc:.04f}\n")
        the_file.write(f"Best Validation Loss: {config.best_val_loss:.04f}\n\n")


def extract_wrong_classifications(true_classes, pred_classes, ztf_test_names, fit_folder, wc_folder):
    """Extracts the wrong model classifications and copies them to a separate folder.

    Parameters
    ----------
    true_classes : np.ndarray
        The ground truth for the classified samples.
    predicted_classes : np.ndarray
        The predictions for the classified samples.
    ztf_test_names : np.ndarray
        The ZTF object names for the classified samples.
    """
    wrongly_classified = np.where(true_classes != pred_classes)[0]

    for wc_idx in wrongly_classified:
        wc = ztf_test_names[wc_idx]
        wc_type = true_classes[wc_idx]
        wrong_type = pred_classes[wc_idx]
        fn = wc + ".pdf"
        fn_new = wc + "_" + wc_type + "_" + wrong_type + ".pdf"
        shutil.copy(
            os.path.join(fit_folder, fn),
            os.path.join(wc_folder, wc_type + "/" + fn_new),
        )


def get_session_metrics(metrics):
    """Calculates the validation loss and accuracy for the hyperparameter set.
    The best model is considered the one with the lowest validation loss.

    Parameters
    ----------
    metrics : tuple
        Tuple containing the training accuracies and losses, and the validation
        accuracies and losses, for each epoch and fold.

    Returns
    -------
    tuple
        The mean validation loss and accuracy for the hyperparameter set.
    """
    _, _, val_accs, val_losses = list(zip(*metrics))

    # Find min loss value in all folds
    min_val_losses = list(map(min, val_losses))
    # Find indices for corresponding min values
    min_indices = [val_losses[i].index(min_val_loss) for i, min_val_loss in enumerate(min_val_losses)]
    # Get the accuracies for the best validation losses
    val_accs = [val_accs[min_indices[i]] for i, val_accs in enumerate(val_accs)]

    return np.mean(min_val_losses), np.mean(val_accs)


def log_metrics_to_tensorboard(metrics, config, trial_id, base_dir="runs"):
    """Calculates the training and validation accuracies and losses
    for each epoch (by averaging each fold) and logs these metrics to
    Tensorboard using a SummaryWriter. It also stores the run
    configuration for further reference.

    Parameters
    ----------
    metrics : tuple
        Tuple containing the training accuracies and losses,
        and the validation accuracies and losses, for each
        epoch and fold.
    config : ModelConfig
        The model's training configuration.
    trial_id : str
        The experiment identifier.
    base_dir : str
        The directory where all tensorboard metrics should be stored.
        Defaults to "runs".

    Returns
    -------
    tuple
        The training losses and accuracies, followed by the validation
        losses and accuracies, for each epoch.
    """
    train_accs, train_losses, val_accs, val_losses = list(zip(*metrics))

    avg_train_losses = np.array(train_losses).mean(axis=0)
    avg_val_losses = np.array(val_losses).mean(axis=0)
    avg_train_accs = np.array(train_accs).mean(axis=0)
    avg_val_accs = np.array(val_accs).mean(axis=0)

    run_dir = os.path.join(base_dir, trial_id)

    writer = SummaryWriter(run_dir)

    for i in range(config.num_epochs):
        writer.add_scalar("Loss/train", avg_train_losses[i], i)
        writer.add_scalar("Loss/val", avg_val_losses[i], i)
        writer.add_scalar("Accuracy/train", avg_train_accs[i], i)
        writer.add_scalar("Accuracy/val", avg_val_accs[i], i)

    # Store current config to file
    config.write_to_file(f"{run_dir}/config.yaml")

    return avg_train_losses, avg_train_accs, avg_val_losses, avg_val_accs
