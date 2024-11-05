import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import jax.numpy as jnp
from snapi.analysis import SamplerResult
from snapi import LightCurve

from superphot_plus.supernova_class import SupernovaClass as SnClass

LOW_SNR_FILE="low_snr_classes.dat"
LOW_VAR_FILE="low_var_classes.dat"


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
    if cube.ndim == 1:
        cube = np.atleast_2d(cube)

    cube = np.repeat(cube.T[:,:,np.newaxis], len(t_data), axis=2)
    t_data = np.repeat(t_data[np.newaxis,:], cube.shape[1], axis=0)
    b_data = np.repeat(b_data[np.newaxis,:], cube.shape[1], axis=0)

    ref_band_idx = np.argmax(ref_band == np.array(ordered_bands))
    si = ref_band_idx * 7

    gamma = 10**cube[si + 2]
    phase = t_data - cube[si + 3]
    f_model = 10**cube[si] / (1.0 + np.exp(-phase / 10**cube[si + 4]))
    f_model = np.where(phase >= gamma, f_model * (1.0 - cube[si+1] * gamma) * np.exp((gamma - phase) / 10**cube[si + 5]), f_model)
    f_model = np.where(phase < gamma, f_model * (1.0 - cube[si+1] * phase), f_model)

    for band_idx, ordered_band in enumerate(ordered_bands):
        if ordered_band == ref_band:
            continue
        si2 = 7 * band_idx
        amp_b = 10**(cube[si] + cube[si2])
        beta_b = cube[si2 + 1] + cube[si + 1]
        gamma_b = gamma * 10**cube[si2 + 2]
        tau_rise_b = 10**(cube[si + 4] + cube[si2 + 4])
        tau_fall_b = 10**(cube[si + 5] + cube[si2 + 5])

        inc_band_ix = b_data[0] == ordered_band
        phase_b = phase - cube[si2 + 3]

        f_model = np.where(inc_band_ix, amp_b / (1.0 + np.exp(-phase_b / tau_rise_b)), f_model)
        f_model = np.where(inc_band_ix & (phase - cube[si2 + 3] >= gamma_b), f_model * (1.0 - beta_b * gamma_b) * np.exp((gamma_b - phase_b) / tau_fall_b), f_model)
        f_model = np.where(inc_band_ix & (phase - cube[si2 + 3] < gamma_b), f_model * (1.0 - phase_b * beta_b), f_model)

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
    if (beta > 0.0) and (gamma > (1.0 - beta * tau_fall) / beta): # no constraint if beta <= 0
        return False

    # ensure dF_rise/dtheta < 0 at gamma
    #if np.exp(-gamma / tau_rise) * (1/tau_rise - beta - beta*gamma/tau_rise) > beta:
    #    return False

    return True

def villar_fit_constraint(x):
    beta, gamma, tau_rise, tau_fall = x
    return (
        jnp.maximum(gamma - (1.0 - beta * tau_fall) / beta, 0.) +
        jnp.maximum(jnp.exp(-gamma / tau_rise) * (1.0/beta - tau_rise - gamma) - tau_rise, 0.)
    )

    
def get_numpyro_cube(params, ref_band, ordered_bands):
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
    logA, beta, log_gamma = params['logA'], params['beta'], params['log_gamma']
    t0, log_tau_rise, log_tau_fall, log_extra_sigma = (
        params['t0'],
        params['log_tau_rise'],
        params['log_tau_fall'],
        params['log_extra_sigma'],
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


def clip_lightcurve_end(light_curve: LightCurve):
    """Clips end of lightcurve with approximately 0 slope. Checks from
    back to max of lightcurve.

    Parameters
    ----------
    times : np.ndarray
        Time values of the light curve.
    fluxes : np.ndarray
        Flux values of the light curve.
    fluxerrs : np.ndarray
        Flux error values of the light curve.
    bands : np.ndarray
        Band information of the light curve.

    Returns
    -------
    tuple
        Tuple containing the clipped light curve data.
    """
    t_b = light_curve.detections['time']
    f_b = light_curve.detections['flux']

    if np.argmax(f_b) == len(f_b) - 1:
        return light_curve.copy()

    end_i = len(t_b) - np.argmax(f_b)
    num_to_cut = 0

    m_cutoff = 0.2 * np.abs((f_b[-1] - np.amax(f_b)) / (t_b[-1] - t_b[np.argmax(f_b)]))

    for i in range(2, end_i):
        cut_idx = -1 * i
        m = (f_b[cut_idx] - f_b[-1]) / (t_b[cut_idx] - t_b[-1])

        if np.abs(m) < m_cutoff:
            num_to_cut = i

    if num_to_cut > 0:
        return LightCurve(
            light_curve.detections[:-num_to_cut],
            filt=light_curve.filter
        )
    return light_curve.copy()


def import_labels_only(input_csvs, allowed_types, fits_dir=None, needs_posteriors=True, sampler=None):
    """Filters CSVs for rows where label is in allowed_types and returns
    names, labels.

    Parameters
    ----------
    input_csvs : list of str
        List of input CSV file paths.
    allowed_types : list
        List of allowed types for labels.
    fits_dir : str, optional
        Directory path for FITS files. Defaults to None.
    needs_posteriors: boolean, optional
        Indicates whether to load posterior samples.
    sampler : str, optional
        The sampler to get posteriors from.

    Returns
    -------
    tuple of np.ndarray
        Tuple of names, labels and redshifts.

    Notes
    -----
    Maps groups of similar labels to a single representative label name
    (eg, "SN Ic", "SNIc-BL", and "21" all become "SN Ibc").
    """
    
    labels = []
    labels_orig = []
    repeat_ct = 0
    names = []
    redshifts = []
    
    for input_csv in input_csvs:
        df = pd.read_csv(input_csv)
        names_all = df.NAME.to_numpy()
        labels_all = df.CLASS.to_numpy()
        redshifts_all = df.Z.to_numpy()
        
        for i, name in enumerate(names_all):
            if needs_posteriors and (
                    fits_dir is None or not has_posterior_samples(
                    lc_name=name, fits_dir=fits_dir, sampler=sampler
                )
            ):
                continue
                
            label_orig = labels_all[i]
            row_label = SnClass.canonicalize(label_orig)

            if row_label not in allowed_types:
                continue

            if name not in names:
                names.append(name)
                labels.append(row_label)
                labels_orig.append(label_orig)
                redshifts.append(float(redshifts_all[i]))
            else:
                repeat_ct += 1

    tally_each_class(labels_orig)
    print(repeat_ct)

    return np.array(names), np.array(labels), np.array(redshifts)


def tally_each_class(labels):
    """Prints the number of samples with each class label.

    Parameters
    ----------
    labels: list
        Input labels.
    """
    un_labels, cts = np.unique(labels, return_counts=True)
    for u, c in zip(un_labels, cts):
        print(f"{u}: {c}")
    print()


def retrieve_posterior_set(
    lc_names, fits_dir, sampler=None,
    redshifts=None, labels=None,
    chisq_cutoff=np.inf,
):
    """Retrieve all sets of posterior samples, excluding
    poor median fits and invalid redshift values.
    
    Parameters
    ----------
    lc_names : str
        Lightcurve names.
    fits_dir : str
        Where fit parameters are stored.
    sampler : str, optional
        The name of the sampler to use.
    redshifts : list, optional
        List of redshift values.
    chisq_cutoff : float, optional
        Ignore all fit sets with median chisq above this value.
    """
    samples = []
    if redshifts is None:
        redshifts = np.ones(len(lc_names))

    for i, name in enumerate(lc_names):
        if np.isnan(redshifts[i]) or redshifts[i] <= 0:
            continue
        try:
            post_obj = SamplerResult.load(
                name=name,
                input_dir=fits_dir,
                sampling_method=sampler
            )
        except:
            continue
        # bandaid: add redshifts to PosteriorSamples object here
        post_obj.redshift = redshifts[i]
        if labels is not None:
            post_obj.sn_class = labels[i]        
        if post_obj.score > chisq_cutoff:
            continue
        
        samples.append(post_obj)

    return np.array(samples)


def normalize_features(features, mean=None, std=None):
    """Normalizes the features for feeding into the neural network.

    Parameters
    ----------
    features : numpy array
        Input features. Must be a 2-d array where each row corresponds
        to a data point and each entry to a feature.
    mean : ndarray, optional
        Mean values for normalization. Defaults to None.
    std : ndarray, optional
        Standard deviation values for normalization. Defaults to None.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing normalized features, mean values, and standard
        deviation values.
    """
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)

    safe_std = np.copy(std)
    safe_std[std == 0.0] = 1.0
    return (features - mean) / safe_std, mean, std
