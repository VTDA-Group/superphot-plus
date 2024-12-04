"""This module provides various functions for analyzing and visualizing
light curve data."""

import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from superphot_plus.utils import calc_accuracy, f1_score


def plot_confusion_matrix(ax, probs_df, purity=False, cmap="Purples"):
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
    y_true = probs_df['true_class'].to_numpy()
    y_pred = probs_df['pred_class'].to_numpy()
    folds = probs_df.get('fold', default=None)
    
    if (folds is None) or (len(np.unique(folds)) == 1): # all the same K-fold
        acc = calc_accuracy(y_pred, y_true)
        f1_avg = f1_score(y_pred, y_true, class_average=True)
        
        if purity:
            title = "Purity\n"+rf"$N = {len(y_pred)}, A = {acc:.2f}, F_1 = {f1_avg:.2f}$"
            cm_vals = confusion_matrix(y_true, y_pred, normalize="pred")
        else:
            title = "Completeness\n"+rf"$N = {len(y_pred)}, A = {acc:.2f}, F_1 = {f1_avg:.2f}$"
            cm_vals = confusion_matrix(y_true, y_pred, normalize="true")
            
    else:
        folds = folds.to_numpy()
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
            title = "Purity\n" +rf"$N = {len(y_pred)}, "
        else:
            title = "Completeness\n" + rf"$N = {len(y_pred)}, "
            
        title += rf"A = {acc:.2f}^{{+{acc_high:.2f}}}_{{-{acc_low:.2f}}}, "
        title += rf"F_1 = {f1_avg:.2f}^{{+{f1_high:.2f}}}_{{-{f1_low:.2f}}}$"

    classes = unique_labels(y_true, y_pred)

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
            if (folds is None) or (len(np.unique(folds)) == 1):
                ax.text(
                    j,
                    i,
                    rf"${cm_vals[i, j]:.2f}$"+ "\n" + rf"$({num_in_cell})$",
                    ha="center",
                    va="center",
                    color="white" if cm_vals[i, j] > thresh else "black",
                )
            else:
                ax.text(
                    j,
                    i,
                    rf"${cm_vals[i, j]:.2f}^{{+{cm_high[i, j]:.2f}}}_{{-{cm_low[i, j]:.2f}}}$" + "\n" + rf"$({num_in_cell})$",
                    ha="center",
                    va="center",
                    color="white" if cm_vals[i, j] > thresh else "black",
                )
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(len(classes) - 0.5, -0.5)
    
    return ax


def plot_matrices(
    config,
    probs_df,
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
    cm_prefix = config.cm_prefix
    
    folds = probs_df.get('fold', None)
    if (folds is not None) and (len(np.unique(folds)) == 1):
        cm_prefix += f"_{folds.iloc[0]}"
        
    prefix = os.path.join(cm_folder, cm_prefix)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax = plot_confusion_matrix(ax, probs_df, purity=False)
    fig.tight_layout()
    fig.savefig(prefix + "_completeness.pdf", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax = plot_confusion_matrix(ax, probs_df, purity=True)
    fig.tight_layout()
    fig.savefig(prefix+"_purity.pdf", bbox_inches='tight')
    plt.close()
