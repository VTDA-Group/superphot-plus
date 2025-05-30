import os

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import jax.numpy as jnp
from snapi import LightCurve


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


def flux_model(cube, t_data, b_data):
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
    if cube.ndim == 2:
        cube = np.atleast_3d(cube) # (num_params, num_times, num_fits)
    # flip last two dimensions
    amp, beta, gamma, t0, tau_rise, tau_fall, _ = cube.transpose(0,2,1)
    
    t_data = np.repeat(t_data[np.newaxis,:], cube.shape[2], axis=0)
    b_data = np.repeat(b_data[np.newaxis,:], cube.shape[2], axis=0)
    phase = np.clip(t_data - t0, a_min = -50. * tau_rise, a_max = None)
    phase = np.clip(phase, a_min = gamma - 50. * tau_fall, a_max = None)
    f_model = amp / (1.0 + np.exp(-phase / tau_rise))
    f_model = np.where(
        phase >= gamma,
        f_model * (1.0 - beta * gamma) * np.exp((gamma - phase) / tau_fall),
        f_model * (1.0 - beta * phase)
    )
    return f_model


def params_valid(cube):
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

    if np.any(np.isnan(cube)):
        return False
    
    if np.any(cube[1] > 1. / cube[2]):
        return False
    
    if np.any(
        np.exp(-cube[2]/cube[4]) * (cube[5]/cube[4] - 1.) > 1.0
    ):
        return False
    
    if np.any(
        cube[1] * cube[5] > 1. - cube[1] * cube[2]
    ):
        return False
    """
    if np.any(
        np.exp(-cube[2]/cube[4]) * (1. - cube[1]*cube[2] - 2.*cube[1]*cube[4]) > 1.0 - cube[1]*cube[2] + 2.*cube[1]*cube[4]
    ):
        return False
    """

    return True

def villar_fit_constraint(x):

    return (
        jnp.maximum(x[2] * x[1] - 1., 0.) +
        jnp.maximum(jnp.exp(-x[2]/x[4])*(x[5]/x[4]-1) - 1., 0.) +
        jnp.maximum(x[1]*x[5] - 1. + x[1]*x[2], 0.)
        #jnp.maximum(jnp.exp(-x[2] / x[4]) * (1.0 / x[1] - x[4] - x[2]) - x[4], 0.)
    )

def create_dataset(features, labels, device='cpu'):
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
    tensor_x = torch.tensor(features, dtype=torch.float, device=device)  # transform to torch tensor
    tensor_y = torch.tensor(labels, dtype=torch.int64, device=device)
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
    
def write_metrics_to_file(
    config,
    model,
    probs_df,
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
    true_classes = probs_df['true_class'].to_numpy()
    pred_classes = probs_df['pred_class'].to_numpy()
    
    test_acc = calc_accuracy(pred_classes, true_classes)
    test_f1_score = f1_score(pred_classes, true_classes, class_average=True)

    with open(config.log_fn, "a+", encoding="utf-8") as the_file:
        the_file.write(str(config.fits_per_majority) + " samples per majority class event\n")
        the_file.write(
            str(config.neurons_per_layer)
            + " neurons per each of "
            + str(config.num_hidden_layers)
            + " layers\n"
        )
        the_file.write(str(config.num_epochs) + " epochs\n")
        the_file.write(f"MLP class-averaged F1-score: {test_f1_score:.04f}\n")
        the_file.write(f"Accuracy: {test_acc:.04f}\n")
        the_file.write(f"Best Validation Loss: {model.best_val_loss:.04f}\n\n")


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