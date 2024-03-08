"""This module provides various functions for analyzing and visualizing
light curve data."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from superphot_plus.plotting.utils import read_probs_csv, retrieve_four_class_info
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import calc_accuracy, f1_score
from superphot_plus.constants import BIGGER_SIZE, MEDIUM_SIZE, SMALL_SIZE

from superphot_plus.plotting.format_params import set_global_plot_formatting

set_global_plot_formatting()


def plot_high_confidence_confusion_matrix(probs_csv, filename, cutoff=0.7, num_include=None):
    """Plot confusion matrices for high-confidence predictions.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    filename : str
        Base filename for saving the confusion matrix plots.
    cutoff : float, optional
        Probability cutoff value for high-confidence predictions.
        Default is 0.7.
    """
    _, classes_to_labels = SnClass.get_type_maps()

    _, true_classes, probs, pred_classes, folds, _ = read_probs_csv(probs_csv)
    
    if num_include is not None:
        confidences = np.max(probs, axis=1)
        conf_ordered = np.sort(confidences)
        print(conf_ordered)
        cutoff = conf_ordered[len(conf_ordered) - num_include - 1]
        
    high_conf_mask = np.max(probs, axis=1) > cutoff

    true_labels = [classes_to_labels[x] for x in true_classes[high_conf_mask]]
    pred_labels = [classes_to_labels[x] for x in pred_classes[high_conf_mask]]
    
    try:
        folds = folds[high_conf_mask]
    except:
        folds = None

    plot_confusion_matrix(true_labels, pred_labels, filename + "_c.pdf", folds, purity=False)
    plot_confusion_matrix(true_labels, pred_labels, filename + "_p.pdf", folds, purity=True)


def plot_binary_confusion_matrix(probs_csv, filename, cutoff=0.5):
    """Merge all non-Ia into one core collapse class and plot resulting
    binary confusion matrix.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    filename : str
        Base filename for saving the confusion matrix plots.
    """
    df = pd.read_csv(probs_csv)
    true_classes = df.Label.to_numpy()
    prob_Ia = df.pSNIa.to_numpy()
    
    pred_binary = np.where(prob_Ia > cutoff, "SN Ia", "SN CC")
    true_binary = np.where(true_classes == 0, "SN Ia", "SN CC")
    try:
        folds = df.Fold.to_numpy()
    except:
        folds = None

    plot_confusion_matrix(true_binary, pred_binary, filename + "_c.pdf", folds, purity=False)
    plot_confusion_matrix(true_binary, pred_binary, filename + "_p.pdf", folds, purity=True)


def compare_four_class_confusion_matrices(probs_csv, probs_alerce_csv, save_dir, p07=False):
    """Plots ALeRCE's classifications as confusion matrix, and compare
    to condensed four-class CM of Superphot+.

    Only four classes as SNe IIn is not a label in their transient
    classifier.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing Superphot+ probability predictions.
    probs_alerce_csv : str
        Path to the CSV file containing ALeRCE predicted classes.
    save_dir : str
        Directory for saving the confusion matrix plots.
    p07 : bool, optional
        If True, only include predictions with a probability >= 0.7.
        Default is False.
    """
    (
        _, true_labels, _,
        pred_labels, pred_alerce, folds
    ) = retrieve_four_class_info(probs_csv, probs_alerce_csv, p07)

    plot_confusion_matrix(
        true_labels, pred_labels,
        os.path.join(save_dir, "superphot4_c.pdf"),
        folds=folds, purity=False, cmap='custom_cmap1',
    )
    plot_confusion_matrix(
        true_labels, pred_labels,
        os.path.join(save_dir, "superphot4_p.pdf"),
        folds=folds, purity=True, cmap='custom_cmap1',
    )

    plot_confusion_matrix(
        true_labels, pred_alerce,
        os.path.join(save_dir, "alerce_c.pdf"),
        folds=folds, purity=False, cmap='custom_cmap2',
    )
    plot_confusion_matrix(
        true_labels, pred_alerce,
        os.path.join(save_dir, "alerce_p.pdf"),
        folds=folds, purity=True, cmap='custom_cmap2',
    )


def plot_true_agreement_matrix(probs_csv, probs_alerce_csv, save_dir, spec=True):
    """Plot agreement matrix between ALeRCE and Superphot+
    classifications.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    probs_alerce_csv : str
        Path to the CSV containing ALeRCE predictions.
    save_dir : str
        Directory path for saving the agreement matrix plot.
    """
    pred_labels, pred_alerce, folds = retrieve_four_class_info(
        probs_csv,
        probs_alerce_csv,
        False,
    )[3:6]

    plot_agreement_matrix_from_arrs(pred_labels, pred_alerce, folds, save_dir, spec=spec)


def plot_expected_agreement_matrix(probs_csv, probs_alerce_csv, save_dir, cmap="custom_cmap2"):
    """Plot expected agreement matrix based on independent ALeRCE and
    Superphot+ confusion matrices.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    save_dir : str
        Directory for saving the expected agreement matrix plot.
    cmap : matplotlib.colors.Colormap, optional
        Color map for the plot. Default is plt.cm.Purples.
    """
    (_, true_labels, _, pred_labels, alerce_preds, folds) = retrieve_four_class_info(
        probs_csv, probs_alerce_csv
    )

    accs = []
    cm_vals_all = []
    alerce_preds = np.array(alerce_preds)
    classes = unique_labels(alerce_preds, pred_labels)

    for f in np.unique(folds):
        ap_fold = alerce_preds[folds == f]
        cm_purity = confusion_matrix(
            true_labels[folds == f],
            pred_labels[folds == f],
            normalize="pred"
        )
        cm_complete = confusion_matrix(
            true_labels[folds == f],
            ap_fold,
            normalize="true"
        )
        cm_expected = cm_purity.T @ cm_complete

        exp_acc = 0
        # calculate agreement score
        for i, single_class in enumerate(classes):
            num_in_class = len(ap_fold[ap_fold == single_class])
            exp_acc += num_in_class * cm_expected[i, i]

        accs.append(exp_acc / len(ap_fold))
        cm_vals_all.append(cm_expected)
        
    cm_vals_all = np.asarray(cm_vals_all)
    cm_expected = np.median(cm_vals_all, axis=0)
    cm_low = np.abs(cm_expected - np.percentile(cm_vals_all, 10, axis=0))
    cm_high = np.abs(np.percentile(cm_vals_all, 90, axis=0) - cm_expected)

    acc = np.median(accs)
    acc_low = acc - np.percentile(accs, 10)
    acc_high = np.percentile(accs, 90) - acc

    title = f"Expected Agreement Matrix,\nSpec. ($A' = {acc:.2f}^{{+{acc_high:.2f}}}_{{-{acc_low:.2f}}}$)"
    fig, axis = plt.subplots(figsize=(6,6))
    _ = axis.imshow(cm_expected, interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap)

    axis.set(
        xticks=np.arange(cm_expected.shape[1]),
        yticks=np.arange(cm_expected.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="ALeRCE Classification",
        xlabel="Superphot+ Classification",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"
    thresh = cm_expected.max() / 1.5
    for i in range(cm_expected.shape[0]):
        for j in range(cm_expected.shape[1]):
            axis.text(
                j,
                i,
                f"${cm_expected[i, j]:.2f}^{{+{cm_high[i, j]:.2f}}}_{{-{cm_low[i, j]:.2f}}}$",
                ha="center",
                va="center",
                color="white" if cm_expected[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)
    plt.savefig(
        os.path.join(save_dir, "expected_agreement.pdf"),
        bbox_inches='tight',
    )
    plt.close()


def plot_agreement_matrix_from_arrs(our_labels, alerce_labels, folds, save_dir, spec=True, cmap="custom_cmap2"):
    """Helper function to plot agreement matrices.

    Plot agreement matrix based on input arrays of ALeRCE and Superphot+
    classifications.

    Parameters
    ----------
    our_labels : list
        List of our predicted labels.
    alerce_labels : list
        List of ALeRCE predicted labels.
    filename : str
        Base filename for saving the agreement matrix plot.
    cmap : matplotlib.colors.Colormap, optional
        Color map for the plot. Default is plt.cm.Purples.
    """
    if spec:
        suffix_title = "Spec."
        suffix = "spec"
    else:
        suffix_title = "Phot."
        suffix = "phot"

    accs = []
    cm_vals_all = []
    classes = unique_labels(alerce_labels, our_labels)
    our_labels = np.array(our_labels)
    alerce_labels = np.array(alerce_labels)
    
    for f in np.unique(folds):
        cm_vals_all.append(
            confusion_matrix(
                alerce_labels[folds == f],
                our_labels[folds == f], normalize="true"
            )
        )
        
        accs.append(
            calc_accuracy(
                alerce_labels[folds == f],
                our_labels[folds == f]
            )
        )

    cm_vals_all = np.asarray(cm_vals_all)
    cm = np.median(cm_vals_all, axis=0)
    cm_low = np.abs(cm - np.percentile(cm_vals_all, 10, axis=0))
    cm_high = np.abs(np.percentile(cm_vals_all, 90, axis=0) - cm)

    acc = np.median(accs)
    acc_low = acc - np.percentile(accs, 10)
    acc_high = np.percentile(accs, 90) - acc
        
    if spec:
        title = "True Agreement Matrix,\n" + fr"{suffix_title} ($A' = {acc:.2f}^{{+{acc_high:.2f}}}_{{-{acc_low:.2f}}}$)"
    else:
        title = "True Agreement Matrix,\n" + fr"{suffix_title} ($A' = {acc:.2f}$)"
        
    fig, ax = plt.subplots(figsize=(6,6))
    _ = ax.imshow(cm, interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="ALeRCE Classification",
        xlabel="Superphot+ Classification",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 1.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            class_i = classes[i]
            class_j = classes[j]
            num_in_cell = len(our_labels[(our_labels == class_j) & (alerce_labels == class_i)])
            if spec:
                ax.text(
                    j,
                    i,
                    f"${cm[i, j]:.2f}^{{+{cm_high[i, j]:.2f}}}_{{-{cm_low[i, j]:.2f}}}$" + f"\n({num_in_cell})",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            else:
                ax.text(
                j,
                i,
                f"${cm[i, j]:.2f}$" + f"\n({num_in_cell})",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
                
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)

    plt.savefig(
        os.path.join(save_dir, f"true_agreement_{suffix}.pdf"),
        bbox_inches='tight'
    )
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename, folds=None, purity=False, cmap="custom_cmap1"):
    """Plot the confusion matrix between given true and predicted
    labels.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    filename : str
        Base filename for saving the confusion matrix plot.
    purity : bool, optional
        If True, plot the purity confusion matrix. Default is False.
    cmap : matplotlib.colors.Colormap, optional
        Color map for the plot. Default is plt.cm.Purples.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if folds is None:
        acc = calc_accuracy(y_pred, y_true)
        f1_avg = f1_score(y_pred, y_true, class_average=True)
        
        if purity:
            title = f"Purity\n$N = {len(y_pred)}, A = {acc:.2f}, F_1 = {f1_avg:.2f}$"
            cm_vals = confusion_matrix(y_true, y_pred, normalize="pred")
        else:
            title = f"Completeness\n$N = {len(y_pred)}, A = {acc:.2f}, F_1 = {f1_avg:.2f}$"
            cm_vals = confusion_matrix(y_true, y_pred, normalize="true")
            
    else:
        folds = np.array(folds)

        print(len(y_pred), len(folds))
        accs = []
        f1s = []
        cm_vals_all = []
        for f in np.unique(folds):
            y_pred_sub = y_pred[folds == f]
            y_true_sub = y_true[folds == f]
            
            accs.append(
                calc_accuracy(y_pred_sub, y_true_sub)
            )
            
            f1s.append(
                f1_score(y_pred_sub, y_true_sub, class_average=True)
            )
            if purity:
                cm_vals_all.append(
                    confusion_matrix(y_true_sub, y_pred_sub, normalize="pred")
                )
            else:
                cm_vals_all.append(
                    confusion_matrix(y_true_sub, y_pred_sub, normalize="true")
                )
                
        cm_vals_all = np.asarray(cm_vals_all)
        cm_vals = np.median(cm_vals_all, axis=0)
        cm_low = np.abs(cm_vals - np.percentile(cm_vals_all, 10, axis=0))
        cm_high = np.abs(np.percentile(cm_vals_all, 90, axis=0) - cm_vals)
        
        acc = np.median(accs)
        acc_low = acc - np.percentile(accs, 10)
        acc_high = np.percentile(accs, 90) - acc
        f1_avg = np.median(f1s)
        f1_low = f1_avg - np.percentile(f1s, 10)
        f1_high = np.percentile(f1s, 90) - f1_avg
        
        # plt.rcParams["figure.figsize"] = (16, 16)
        if purity:
            title = f"Purity\n$N = {len(y_pred)}, "
        else:
            title = f"Completeness\n$N = {len(y_pred)}, "
            
        title += f"A = {acc:.2f}^{{+{acc_high:.2f}}}_{{-{acc_low:.2f}}}, "
        title += f"F_1 = {f1_avg:.2f}^{{+{f1_high:.2f}}}_{{-{f1_low:.2f}}}$"

    classes = unique_labels(y_true, y_pred)

    N_class = len(np.unique(y_true))
    fig, ax = plt.subplots(figsize=(7, 7))
    _ = ax.imshow(cm_vals, interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap)

    ax.set(
        xticks=np.arange(cm_vals.shape[1]),
        yticks=np.arange(cm_vals.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="Spectroscopic Classification",
        xlabel="Photometric Classification",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm_vals.max() / 1.5

    for i in range(cm_vals.shape[0]):
        for j in range(cm_vals.shape[1]):
            class_i = classes[i]
            class_j = classes[j]
            num_in_cell = len(y_pred[(y_pred == class_j) & (y_true == class_i)])
            if folds is None:
                ax.text(
                    j,
                    i,
                    f"${cm_vals[i, j]:.2f}\n({num_in_cell})$",
                    ha="center",
                    va="center",
                    color="white" if cm_vals[i, j] > thresh else "black",
                )
            else:
                ax.text(
                    j,
                    i,
                    f"${cm_vals[i, j]:.2f}^{{+{cm_high[i, j]:.2f}}}_{{-{cm_low[i, j]:.2f}}}$" + f"\n({num_in_cell})",
                    ha="center",
                    va="center",
                    color="white" if cm_vals[i, j] > thresh else "black",
                )
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_matrices(
    config,
    true_classes,
    pred_classes,
    prob_above_07,
):
    """Plots confusion matrices for test set metrics.

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
    cm_folder : str
        The folder where the plot figures will be stored.
    """
    cm_folder = config.cm_dir
    fn_prefix = f"cm_{config.goal_per_class}_{config.num_epochs}_{config.neurons_per_layer}_{config.num_hidden_layers}"

    fn_purity = os.path.join(cm_folder, fn_prefix + "_p.pdf")
    fn_completeness = os.path.join(cm_folder, fn_prefix + "_c.pdf")
    fn_purity_07 = os.path.join(cm_folder, fn_prefix + "_p_p07.pdf")
    fn_completeness_07 = os.path.join(cm_folder, fn_prefix + "_c_p07.pdf")

    # Plot full confusion matrices
    plot_confusion_matrix(true_classes, pred_classes, fn_purity, purity=True)
    plot_confusion_matrix(true_classes, pred_classes, fn_completeness, purity=False)

    # Plot confusion matrices for p > 0.7
    if np.any(prob_above_07):
        plot_confusion_matrix(
            true_classes[prob_above_07],
            pred_classes[prob_above_07],
            fn_purity_07,
            purity=True,
        )
        plot_confusion_matrix(
            true_classes[prob_above_07],
            pred_classes[prob_above_07],
            fn_completeness_07,
            purity=False,
        )
