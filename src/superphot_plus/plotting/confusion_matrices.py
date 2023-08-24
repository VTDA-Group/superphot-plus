"""This module provides various functions for analyzing and visualizing
light curve data."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from superphot_plus.file_paths import CM_FOLDER
from superphot_plus.plotting.utils import read_probs_csv
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import calc_accuracy, f1_score


def plot_high_confidence_confusion_matrix(probs_csv, filename, cutoff=0.7):
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

    (
        names,
        true_classes,
        probs,
        pred_classes,
    ) = read_probs_csv(probs_csv)
    high_conf_mask = np.max(probs, axis=1) > cutoff

    true_labels = [classes_to_labels[x] for x in true_classes[high_conf_mask]]
    pred_labels = [classes_to_labels[x] for x in pred_classes[high_conf_mask]]

    plot_confusion_matrix(true_labels, pred_labels, filename + "_c.pdf", purity=False)
    plot_confusion_matrix(true_labels, pred_labels, filename + "_p.pdf", purity=True)


def plot_snIa_confusion_matrix(probs_csv, filename, p07=False):
    """Merge all non-Ia into one core collapse class and plot resulting
    binary confusion matrix.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    filename : str
        Base filename for saving the confusion matrix plots.
    p07 : bool, optional
        If True, only include predictions with a probability >= 0.7.
        Default is False.
    """
    (
        names,
        true_classes,
        probs,
        pred_classes,
    ) = read_probs_csv(probs_csv)
    pred_binary = np.where(probs[:, 0] > 0.5, "SN Ia", "SN CC")
    true_binary = np.where(true_classes == 0, "SN Ia", "SN CC")

    plot_confusion_matrix(true_binary, pred_binary, filename + "_c.pdf", purity=False)
    plot_confusion_matrix(true_binary, pred_binary, filename + "_p.pdf", purity=True)


def plot_alerce_confusion_matrix(probs_csv, filename, p07=False):
    """Plots ALeRCE's classifications as confusion matrix.

    Only four classes as SNe IIn is not a label in their transient
    classifier.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    filename : str
        Base filename for saving the confusion matrix plots.
    p07 : bool, optional
        If True, only include predictions with a probability >= 0.7.
        Default is False.
    """
    _, classes_to_labels = SnClass.get_type_maps()
    true_classes = []
    pred_classes = []
    with open(probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for e, row in enumerate(csvreader):
            name = row[0]
            try:
                pass
                # pred_class = get_alerce_pred_class(name, reflect_style=True)
            except:
                print(name, " skipped")
                continue
            if p07 and np.max(np.array(row[2:]).astype(float)) < 0.7:
                continue
            if int(row[1][-2]) == 2:
                true_classes.append(1)
            else:
                true_classes.append(int(row[1][-2]))

            pred_index = np.argmax(
                np.array(
                    [
                        float(row[2]),
                        float(row[3]) + float(row[4]),
                        0.0,
                        float(row[5]),
                        float(row[6]),
                    ]
                )
            )
            pred_classes.append(pred_index)
            # pred_classes.append(pred_class)
            print(e)
    true_labels = [classes_to_labels[x] for x in true_classes]
    # pred_labels = pred_classes
    pred_labels = [classes_to_labels[x] for x in pred_classes]
    plot_confusion_matrix(true_labels, pred_labels, filename + "_c.pdf", purity=False, cmap=plt.cm.Reds)
    plot_confusion_matrix(true_labels, pred_labels, filename + "_p.pdf", purity=True, cmap=plt.cm.Reds)


def plot_agreement_matrix(probs_csv, filename):
    """Plot agreement matrix between ALeRCE and Superphot+
    classifications.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    filename : str
        Base filename for saving the agreement matrix plot.
    """
    _, classes_to_labels = SnClass.get_type_maps()
    pred_classes = []
    alerce_preds = []
    with open(probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for e, row in enumerate(csvreader):
            name = row[0]
            try:
                alerce_pred = get_alerce_pred_class(name, reflect_style=True)
                print(alerce_pred, e)
            except:
                print(name, " skipped")
                continue
            pred_index = np.argmax(
                np.array(
                    [
                        float(row[-5]),
                        float(row[-4]) + float(row[-3]),
                        0.0,
                        float(row[-2]),
                        float(row[-1]),
                    ]
                )
            )
            alerce_preds.append(alerce_pred)
            pred_classes.append(pred_index)
    pred_labels = [classes_to_labels[x] for x in pred_classes]

    plot_agreement_matrix_from_arrs(pred_labels, alerce_preds, filename)


def plot_expected_agreement_matrix(probs_csv, filename, cmap=plt.cm.Purples):
    """Plot expected agreement matrix based on independent ALeRCE and
    Superphot+ confusion matrices.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    filename : str
        Base filename for saving the expected agreement matrix plot.
    cmap : matplotlib.colors.Colormap, optional
        Color map for the plot. Default is plt.cm.Purples.
    """
    _, classes_to_labels = SnClass.get_type_maps()
    pred_classes = []
    alerce_preds = []

    true_classes = []
    with open(probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for e, row in enumerate(csvreader):
            name = row[0]
            try:
                alerce_pred = get_alerce_pred_class(name, reflect_style=True)
                print(alerce_pred, e)
            except:
                print(name, " skipped")
                continue
            if int(row[1][-2]) == 2:
                true_classes.append(1)
            else:
                true_classes.append(int(row[1][-2]))

            pred_index = np.argmax(
                np.array(
                    [
                        float(row[2]),
                        float(row[3]) + float(row[4]),
                        0.0,
                        float(row[5]),
                        float(row[6]),
                    ]
                )
            )
            alerce_preds.append(alerce_pred)
            pred_classes.append(pred_index)
    pred_labels = [classes_to_labels[x] for x in pred_classes]
    true_labels = [classes_to_labels[x] for x in true_classes]

    cm_purity = confusion_matrix(true_labels, pred_labels, normalize="pred")

    cm_complete = confusion_matrix(true_labels, alerce_preds, normalize="true")

    cm = cm_purity.T @ cm_complete
    classes = unique_labels(alerce_preds, pred_labels)

    alerce_preds = np.array(alerce_preds)

    exp_acc = 0
    # calculate agreement score
    for i, c in enumerate(classes):
        num_in_class = len(alerce_preds[alerce_preds == c])
        exp_acc += num_in_class * cm[i, i]

    exp_acc /= len(alerce_preds)

    title = r"Expected Agreement Matrix, Spec. ($A' = %.2f$)" % exp_acc
    fig, ax = plt.subplots()
    im = ax.imshow(
        cm, interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap
    )  # pylint: disable=unused-variable
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
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
    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)
    plt.savefig(os.path.join(CM_FOLDER, filename))
    plt.close()


def plot_agreement_matrix_from_arrs(our_labels, alerce_labels, filename, cmap=plt.cm.Purples):
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
    cm = confusion_matrix(alerce_labels, our_labels, normalize="true")
    classes = unique_labels(alerce_labels, our_labels)

    our_labels = np.array(our_labels)
    alerce_labels = np.array(alerce_labels)

    exp_acc = calc_accuracy(alerce_labels, our_labels)
    title = r"True Agreement Matrix, Spec. ($A' = %.2f$)" % exp_acc
    fig, ax = plt.subplots()
    im = ax.imshow(
        cm, interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap
    )  # pylint: disable=unused-variable

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
    fmt = ".2f"  # pylint: disable=unused-variable
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            class_i = classes[i]
            class_j = classes[j]
            num_in_cell = len(our_labels[(our_labels == class_j) & (alerce_labels == class_i)])
            ax.text(
                j,
                i,
                "%.2f\n(%d)" % (cm[i, j], num_in_cell),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)

    plt.savefig(os.path.join(CM_FOLDER, filename))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename, purity=False, cmap=plt.cm.Purples):
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

    acc = calc_accuracy(y_pred, y_true)
    f1 = f1_score(y_pred, y_true, class_average=True)

    # plt.rcParams["figure.figsize"] = (16, 16)
    if purity:
        title = r"Purity ($N = %d, A = %.2f, F_1 = %.2f$)" % (len(y_pred), acc, f1)
        cm = confusion_matrix(y_true, y_pred, normalize="pred")
    else:
        title = r"Completeness ($N = %d, A = %.2f, F_1 = %.2f$)" % (len(y_pred), acc, f1)
        cm = confusion_matrix(y_true, y_pred, normalize="true")

    classes = unique_labels(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(
        cm, interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap
    )  # pylint: disable=unused-variable

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"  # pylint: disable=unused-variable
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            class_i = classes[i]
            class_j = classes[j]
            num_in_cell = len(y_pred[(y_pred == class_j) & (y_true == class_i)])
            ax.text(
                j,
                i,
                "%.2f\n(%d)" % (cm[i, j], num_in_cell),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
