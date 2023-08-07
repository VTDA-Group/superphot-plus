import matplotlib.pyplot as plt
import numpy as np
import csv
import os

from superphot_plus.supernova_class import SupernovaClass as SnClass

from superphot_plus.plotting.format_params import *

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