"""This module provides various functions for analyzing and visualizing
light curve data."""

import csv
import os

import arviz as az
import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from alerce.core import Alerce
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.lightcurve import Lightcurve

from .constants import BIGGER_SIZE, MEDIUM_SIZE, SMALL_SIZE
from .file_paths import CM_FOLDER, FIT_PLOTS_FOLDER
from .format_data_ztf import import_labels_only, oversample_using_posteriors
from .import_utils import clip_lightcurve_end, import_lc
from .supernova_class import SupernovaClass as SnClass
from .utils import calc_accuracy, f1_score, flux_model

alerce = Alerce()

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


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
    true_classes = []
    pred_classes = []

    with open(probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row_np = np.array(row)
            if np.max(row_np[2:].astype(float)) < cutoff:
                continue
            true_classes.append(int(row_np[1][-2]))
            pred_classes.append(np.argmax(row_np[2:].astype(float)))
    true_labels = [classes_to_labels[x] for x in true_classes]
    pred_labels = [classes_to_labels[x] for x in pred_classes]
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
    classes_to_labels = {0: "SN Ia", 1: "SN CC"}
    true_classes = []
    pred_classes = []
    with open(probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if p07 and np.max(np.array(row[2:]).astype(float)) < 0.7:
                continue
            if int(row[1][-2]) == 0:
                true_classes.append(0)
            else:
                true_classes.append(1)
            snIa_prob = float(row[2])
            sncc_prob = (
                float(row[3]) + float(row[4]) + float(row[5]) + float(row[6])
            )  # pylint: disable=unused-variable
            if snIa_prob > 0.5:
                pred_classes.append(0)
            else:
                pred_classes.append(1)

    true_labels = [classes_to_labels[x] for x in true_classes]
    pred_labels = [classes_to_labels[x] for x in pred_classes]
    plot_confusion_matrix(true_labels, pred_labels, filename + "_c.pdf", purity=False)
    plot_confusion_matrix(true_labels, pred_labels, filename + "_p.pdf", purity=True)


def get_pred_class(ztf_name, reflect_style=False):
    """Get alerce probabilities corresponding to the four (no SN IIn)
    classes in our ZTF classifier.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the object.
    reflect_style : bool, optional
        If True, change format of output labels. Default is False.

    Returns
    -------
    str
        Predicted class label.
    """
    global alerce
    o = alerce.query_probabilities(oid=ztf_name, format="pandas")
    o_transient = o[o["classifier_name"] == "lc_classifier_transient"]
    label = o_transient[o_transient["ranking"] == 1]["class_name"].iat[0]
    return SnClass.get_reflect_style(label) if reflect_style else label


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
                # pred_class = get_pred_class(name, reflect_style=True)
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
                alerce_pred = get_pred_class(name, reflect_style=True)
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
                alerce_pred = get_pred_class(name, reflect_style=True)
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


def save_class_fractions(spec_probs_csv, phot_probs_csv, save_fn):
    """Save class fractions from spectroscopic, photometric, and
    corrected photometric.

    Parameters
    ----------
    spec_probs_csv : str
        Path to the CSV file containing spectroscopic probability
        predictions.
    phot_probs_csv : str
        Path to the CSV file containing photometric probability
        predictions.
    save_fn : str
        Filename for saving the class fractions.
    """
    labels_to_class, _ = SnClass.get_type_maps()
    true_classes = []
    pred_classes = []
    pred_classes_spec = []
    alerce_preds = []
    alerce_preds_spec = []
    true_classes_alerce = []

    ct = 0
    with open(spec_probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            ct += 1
            print(ct)
            try:
                alerce_pred = labels_to_class[get_pred_class(row[0], reflect_style=True)]
            except:
                continue
            alerce_preds_spec.append(alerce_pred)
            l = int(row[1][-2])
            true_classes.append(l)
            if l == 2:
                true_classes_alerce.append(1)
            else:
                true_classes_alerce.append(l)
            pred_classes_spec.append(np.argmax(np.array(row[2:])))

    with open(phot_probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for e, row in enumerate(csvreader):
            print(e)
            name = row[0]
            if row[1] == "SKIP":
                continue
            try:
                alerce_pred = labels_to_class[get_pred_class(name, reflect_style=True)]
                # print(alerce_pred, e)
            except:
                print(name, " skipped")
                continue
            alerce_preds.append(alerce_pred)
            pred_classes.append(np.argmax(np.array(row[2:])))

    true_classes = np.array(true_classes)
    pred_classes = np.array(pred_classes)
    alerce_preds = np.array(alerce_preds)

    cm_p = confusion_matrix(true_classes, pred_classes_spec, normalize="pred")
    cm_p_alerce = confusion_matrix(true_classes_alerce, alerce_preds_spec, normalize="pred")

    true_fracs = np.array([len(true_classes[true_classes == i]) / len(true_classes) for i in range(5)])
    pred_fracs = np.array([len(pred_classes[pred_classes == i]) / len(pred_classes) for i in range(5)])
    alerce_fracs = np.array([len(alerce_preds[alerce_preds == i]) / len(alerce_preds) for i in range(5)])

    pred_fracs_corr = []
    alerce_fracs_corr = []
    for i in range(5):
        pred_fracs_corr.append(np.sum(pred_fracs * cm_p[i]))
        if i == 2:
            alerce_fracs_corr.append(0.0)
        elif i > 2:
            alerce_fracs_corr.append(np.sum(np.delete(alerce_fracs, 2) * cm_p_alerce[i - 1]))
        else:
            alerce_fracs_corr.append(np.sum(np.delete(alerce_fracs, 2) * cm_p_alerce[i]))

    pred_fracs_corr = np.array(pred_fracs_corr)
    alerce_fracs_corr = np.array(alerce_fracs_corr)

    with open(save_fn, "a+") as sf:
        csvwriter = csv.writer(sf)
        csvwriter.writerow(true_fracs)
        csvwriter.writerow(pred_fracs)
        csvwriter.writerow(pred_fracs_corr)
        csvwriter.writerow(alerce_fracs)
        csvwriter.writerow(alerce_fracs_corr)


def plot_class_fractions(saved_cf_file, fig_dir, filename):
    """Plot class fractions saved from 'save_class_fractions'.

    Parameters
    ----------
    saved_cf_file : str
        Path to the saved class fractions file.
    fig_dir : str
        Directory for saving the class fractions plot.
    filename: str
        Filename for the class fractions plot figure.
    """
    _, classes_to_labels = SnClass.get_type_maps()
    labels = [
        "Spec (ZTF)",
        "Spec (YSE)",
        "Spec (PS1-MDS)",
        "Phot",
        "Phot (corr.)",
        "ALeRCE",
        "ALeRCE (corr.)",
    ]
    width = 0.6

    fracs = []
    with open(saved_cf_file, "r") as sf:
        csvreader = csv.reader(sf)
        for row in csvreader:
            fracs.append([float(x) for x in row])

    true_fracs = fracs[0]
    pred_fracs = fracs[1]
    pred_fracs_corr = fracs[2]
    alerce_fracs = fracs[3]
    alerce_fracs_corr = fracs[4]

    # Plot YSE class fractions too
    yse_counts = np.array([314, 107, 15, 2, 32])
    yse_fracs = yse_counts / np.sum(yse_counts)

    # Plot PS-MDS
    psmds_counts = np.array([404, 94, 24, 17, 19])
    psmds_fracs = psmds_counts / np.sum(psmds_counts)

    combined_fracs = np.array(
        [
            true_fracs,
            yse_fracs,
            psmds_fracs,
            pred_fracs,
            pred_fracs_corr,
            alerce_fracs,
            alerce_fracs_corr,
        ]
    ).T
    fig, ax = plt.subplots(figsize=(11, 16))  # pylint: disable=unused-variable
    bar = ax.bar(labels, combined_fracs[0], width, label=classes_to_labels[0])
    for i in range(len(combined_fracs[0])):
        bari = bar.patches[i]
        ax.annotate(
            round(combined_fracs[0][i], 3),
            (bari.get_x() + bari.get_width() / 2, bari.get_y() + bari.get_height() / 2),
            ha="center",
            va="center",
            color="white",
        )

    for i in range(1, 5):
        bar = ax.bar(
            labels,
            combined_fracs[i],
            width,
            bottom=np.sum(combined_fracs[0:i], axis=0),
            label=classes_to_labels[i],
        )
        for j in range(len(combined_fracs[0])):
            barj = bar.patches[j]
            # Create annotation
            ax.annotate(
                round(combined_fracs[i][j], 3),
                (barj.get_x() + barj.get_width() / 2, barj.get_y() + barj.get_height() / 2),
                ha="center",
                va="center",
                color="white",
            )

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=5, fontsize=15
    )

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    # plt.legend(loc=3)
    plt.ylabel("Fraction", fontsize=20)
    plt.savefig(os.path.join(fig_dir, filename))
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
    plt.savefig(os.path.join(CM_FOLDER, filename))
    plt.close()


def corner_plot_all(input_csvs, save_file):
    """Plot combined corner plot of all training set samples, excluding
    the overall scaling A.

    Parameters
    ----------
    input_csvs : list
        List of input CSV file paths containing probability predictions.
    save_file : str
        Path to save the combined corner plot.
    """
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    names, labels = import_labels_only(input_csvs, allowed_types)

    chis = np.ones(len(names))
    features, labels, chis = oversample_using_posteriors(names, labels, chis, 4000)

    figure = corner.corner(
        np.delete(features, [0, 3], axis=1),
        bins=20,
        labels=[
            r"$\beta$",
            r"$\gamma$",
            r"$\tau_r$",
            r"$\tau_f$",
            r"$\sigma_{ex}$",
            r"$A_g$",
            r"$\beta_g$",
            r"$\gamma_g$",
            r"$t_{0,g}$",
            r"$\tau_{r,g}$",
            r"$\tau_{r,g}$",
            r"$\sigma_{ex,g}$",
        ],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 20},
        color="purple",
    )
    # Extract the axes
    axes = np.array(figure.axes)
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)

    figure.savefig(save_file)


def plot_lightcurve_clipping(ztf_name, data_folder, save_dir):
    """Plot the lightcurve WITH clipped points and lines demonstrating
    how those points are clipped.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the plotted object.
    data_folder: str
        The path to the folder holding the CSV data.
    save_dir: str
        Directory path where to store the plot figure.
    """
    data_fn = f"{data_folder}/{ztf_name}.csv"
    t, f, ferr, b, ra, dec = import_lc(data_fn)  # pylint: disable=unused-variable
    t_clip, f_clip, ferr_clip, b_clip = clip_lightcurve_end(t, f, ferr, b)

    idx_clip = ~np.isin(t, t_clip)
    t_clip = t[idx_clip]
    f_clip = f[idx_clip]
    ferr_clip = ferr[idx_clip]
    b_clip = b[idx_clip]

    plt.errorbar(t[b == "r"], f[b == "r"], yerr=ferr[b == "r"], fmt="o", c="r")
    plt.errorbar(t[b == "g"], f[b == "g"], yerr=ferr[b == "g"], fmt="o", c="g")

    # overlay clipped points
    plt.errorbar(
        t_clip[b_clip == "r"],
        f_clip[b_clip == "r"],
        yerr=ferr_clip[b_clip == "r"],
        fmt="o",
        c="orange",
    )
    plt.errorbar(
        t_clip[b_clip == "g"],
        f_clip[b_clip == "g"],
        yerr=ferr_clip[b_clip == "g"],
        fmt="o",
        c="blue",
    )

    # plot lines from last to max flux point
    t_r = t[b == "r"]
    f_r = f[b == "r"]
    t_g = t[b == "g"]
    f_g = f[b == "g"]

    t_range_r = np.linspace(t_r[np.argmax(f_r)], np.max(t_r), num=10)
    m_r = (f_r[np.argmax(t_r)] - np.max(f_r)) / (np.max(t_r) - t_r[np.argmax(f_r)])
    y_r = f_r[np.argmax(t_r)] + m_r * (t_range_r - np.max(t_r))

    t_range_g = np.linspace(t_g[np.argmax(f_g)], np.max(t_g), num=10)
    m_g = (f_g[np.argmax(t_g)] - np.max(f_g)) / (np.max(t_g) - t_g[np.argmax(f_g)])
    y_g = f_g[np.argmax(t_g)] + m_g * (t_range_g - np.max(t_g))

    plt.plot(t_range_r, y_r, c="r", label="Max r-band slope", linewidth=1)
    plt.plot(t_range_g, y_g, c="g", label="Max g-band slope", linewidth=1)

    # plot slope of clipped portion
    t_clip_r = t_clip[b_clip == "r"]
    f_clip_r = f_clip[b_clip == "r"]
    t_clip_g = t_clip[b_clip == "g"]
    f_clip_g = f_clip[b_clip == "g"]

    t_range_r = np.linspace(t_clip_r[np.argmax(f_clip_r)], np.max(t_clip_r), num=10)
    m_r = (f_clip_r[np.argmax(t_clip_r)] - np.max(f_clip_r)) / (
        np.max(t_clip_r) - t_clip_r[np.argmax(f_clip_r)]
    )
    y_r = f_clip_r[np.argmax(t_clip_r)] + m_r * (t_range_r - np.max(t_clip_r))

    t_range_g = np.linspace(t_clip_g[np.argmax(f_clip_g)], np.max(t_clip_g), num=10)
    m_g = (f_clip_g[np.argmax(t_clip_g)] - np.max(f_clip_g)) / (
        np.max(t_clip_g) - t_clip_g[np.argmax(f_clip_g)]
    )
    y_g = f_clip_g[np.argmax(t_clip_g)] + m_g * (t_range_g - np.max(t_clip_g))

    plt.plot(t_range_r, y_r, c="orange", label="Clipped r-band slope", linewidth=1)
    plt.plot(t_range_g, y_g, c="blue", label="Clipped g-band slope", linewidth=1)

    plt.title(ztf_name, fontsize=20)
    plt.xlabel("MJD", fontsize=15)
    plt.ylabel("Flux (arbitrary scaling)", fontsize=15)
    plt.legend()

    plt.savefig(os.path.join(save_dir, "lc_clip_demo.pdf"))
    plt.close()


def plot_lc_fit(ztf_name, data_dir, fit_dir, out_dir, sampling_method="dynesty"):
    """Plot an existing light curve fit.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the object.
    data_dir : str
        Directory containing the data files.
    fit_dir : str
        Directory containing the fit files.
    out_dir : str
        Directory for saving the plot.
    sampling_method : str, optional
        Sampling method used for the fit. Default is "dynesty".
    """
    data_fn = os.path.join(data_dir, ztf_name + ".npz")

    lightcurve = Lightcurve.from_file(data_fn)

    eq_wt_samples = get_posterior_samples(ztf_name, fit_dir, sampling_method)

    plot_sampling_lc_fit(
        ztf_name,
        out_dir,
        lightcurve.times,
        lightcurve.fluxes,
        lightcurve.flux_errors,
        lightcurve.bands,
        eq_wt_samples,
    )


def plot_sampling_lc_fit(
    ztf_name,
    out_dir,
    tdata,
    fdata,
    ferrdata,
    bdata,
    eq_wt_samples,
    sampling_method="dynesty",
):
    """
    Plot lightcurve sampling fit using in-memory samples.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the object.
    out_dir : str
        Directory for saving the plot.
    tdata : array-like
        Time of data.
    fdata : array-like
        Flux of data.
    ferrdata : array-like
        Error in flux of data.
    bdata : array-like
        Band of data.
    eq_wt_samples: array-like
        Equally weighted samples from data.
    sampling_method : str, optional
        Sampling method used for the fit. Default is "dynesty".
    """

    plt.errorbar(
        tdata[bdata == "g"],
        fdata[bdata == "g"],
        yerr=ferrdata[bdata == "g"],
        c="g",
        label="g",
        fmt="o",
    )
    plt.errorbar(
        tdata[bdata == "r"],
        fdata[bdata == "r"],
        yerr=ferrdata[bdata == "r"],
        c="r",
        label="r",
        fmt="o",
    )

    trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

    for sample in eq_wt_samples[:30]:
        plt.plot(
            trange_fine,
            flux_model(sample, trange_fine, ["g"] * len(trange_fine)),
            c="g",
            lw=1,
            alpha=0.1,
        )
        plt.plot(
            trange_fine,
            flux_model(sample, trange_fine, ["r"] * len(trange_fine)),
            c="r",
            lw=1,
            alpha=0.1,
        )

    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title(ztf_name + ": " + sampling_method)

    plt.savefig(os.path.join(out_dir, ztf_name + "_" + sampling_method + ".png"))

    plt.close()


def flux_from_posteriors(t, params, max_flux):
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

    A_g, beta_g, gamma_g = params["A_g"], params["beta_g"], params["gamma_g"]
    t0_g, tau_rise_g, tau_fall_g, extra_sigma_g = (  # pylint: disable=unused-variable
        params["t0_g"],
        params["tau_rise_g"],
        params["tau_fall_g"],
        params["extra_sigma_g"],
    )

    A_b = A * A_g  # pylint: disable=unused-variable
    beta_b = beta * beta_g
    gamma_b = gamma * gamma_g
    t0_b = t0 * t0_g
    tau_rise_b = tau_rise * tau_rise_g
    tau_fall_b = tau_fall * tau_fall_g

    phase = t - t0
    flux_const = A / (1.0 + jnp.exp(-phase / tau_rise))
    sigmoid = 1 / (1 + jnp.exp(10.0 * (gamma - phase)))

    flux_r = flux_const * (
        (1 - sigmoid) * (1 - beta * phase)
        + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
    )

    # g band
    phase_b = t - t0_b
    flux_const_b = A / (1.0 + jnp.exp(-phase_b / tau_rise_b))
    sigmoid_b = 1 / (1 + jnp.exp(10.0 * (gamma_b - phase_b)))

    flux_g = flux_const_b * (
        (1 - sigmoid_b) * (1 - beta_b * phase_b)
        + sigmoid_b * (1 - beta_b * gamma_b) * jnp.exp(-(phase_b - gamma_b) / tau_fall_b)
    )

    return flux_g, flux_r


def plot_posterior_hist(posterior_samples, parameter, output_dir=None):
    """
    Plot histogram for a posterior parameter.

    Parameters
    ----------
    posterior_samples :
        Dictionary of posterior samples.
    parameter :
        Posterior parameter for which to plot histogram.
    output_dir :
        The directory where to store the plot.
    """
    if parameter is None or parameter not in posterior_samples:
        raise ValueError("Invalid posterior parameter.")

    output_file = f"test_hist_{parameter}.png"
    if output_dir is not None:
        output_file = os.path.join(output_dir, output_file)

    samples = posterior_samples[parameter]

    if len(samples.shape) > 1:
        samples = samples[:, 0]
    else:
        samples = samples.flatten()

    plt.hist(samples, bins=10)
    plt.savefig(output_file)
    plt.close()


def plot_sampling_trace_numpyro(posterior_samples, output_dir=None):
    """
    Plot trace of all posterior samples.

    Parameters
    ----------
    posterior_samples :
        The lightcurve samples given by MCMC.
    output_dir :
        The directory where to store the plot.
    """

    output_file = "test_trace.png"
    if output_dir is not None:
        output_file = os.path.join(output_dir, output_file)

    post_reformatted = {}
    for p in posterior_samples:
        post_reformatted[p] = np.array(
            [
                posterior_samples[p],
            ]
        )

    az.plot_trace(post_reformatted, compact=True)
    plt.savefig(output_file)
    plt.close()


def plot_sampling_lc_fit_numpyro(
    posterior_samples,
    tdata,
    fdata,
    ferrdata,
    bdata,
    max_flux,
    lcs,
    t0_lim=None,
    output_folder=FIT_PLOTS_FOLDER,
):
    """
    Plot lightcurve sampling fit using in-memory samples.

    Parameters
    ----------
    posterior_samples :
        Posterior samples from the MCMC run.
    tdata : array-like
        Time of data.
    fdata : array-like
        Flux of data.
    ferrdata : array-like
        Error in flux of data.
    bdata : array-like
        Band of data.
    max_flux : array-like
        Max flux of data.
    lcs: array-like
        Light curve objects on which sampling was run.
    t0_lim:  float or None, optional
        Upper time limit for the data.
    output_folder : str or FITS_PLOTS_FOLDER, optional
        Directory where to store the light curve sampling fit.
    """

    for i in range(len(tdata)):
        ignore_idx = ferrdata[i] == 1e10  # pylint: ignore-superfluous parens
        tdata = tdata[i][~ignore_idx]
        fdata = fdata[i][~ignore_idx]
        ferrdata = ferrdata[i][~ignore_idx]
        bdata = bdata[i][~ignore_idx]

        model_i = np.array(
            [
                {
                    k: posterior_samples[k][j, i]
                    if len(posterior_samples[k].shape) > 1
                    else posterior_samples[k][j]
                    for k in posterior_samples.keys()
                }
                for j in range(len(posterior_samples["log_tau_fall"]))
            ]
        )

        plt.errorbar(
            tdata[bdata == 0],
            fdata[bdata == 0],
            yerr=ferrdata[bdata == 0],
            c="g",
            label="g",
            fmt="o",
        )
        plt.errorbar(
            tdata[bdata == 1],
            fdata[bdata == 1],
            yerr=ferrdata[bdata == 1],
            c="r",
            label="r",
            fmt="o",
        )

        trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

        for sample in model_i[:30]:
            plt.plot(
                trange_fine,
                flux_from_posteriors(trange_fine, sample, max_flux[i])[0],
                c="g",
                lw=1,
                alpha=0.1,
            )
            plt.plot(
                trange_fine,
                flux_from_posteriors(trange_fine, sample, max_flux[i])[1],
                c="r",
                lw=1,
                alpha=0.1,
            )

        plt.xlabel("MJD")
        plt.ylabel("Flux")
        plt.title(lcs[i].name)

        if t0_lim is None:
            plt.savefig(os.path.join(output_folder, "%s.pdf" % lcs[i].name))
        else:
            plt.savefig(os.path.join(output_folder, "%s_%.02f.pdf" % (lcs[i].name, t0_lim)))

        plt.close()
