import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix
from matplotlib.ticker import AutoMinorLocator
from astropy.cosmology import Planck13 as cosmo
from scipy.stats import binned_statistic

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.format_data_ztf import import_labels_only
from superphot_plus.utils import calculate_neg_chi_squareds
from superphot_plus.file_utils import get_multiple_posterior_samples

from superphot_plus.plotting.format_params import *
from superphot_plus.plotting.utils import read_probs_csv, histedges_equalN, get_survey_fracs


def save_class_fractions(spec_probs_csv, probs_alerce_csv, phot_probs_csv, save_path):
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
        Filename + dir for saving the class fractions.
    """
    labels_to_class, _ = SnClass.get_type_maps()

    # import spec dataframe
    names_spec, true_class_spec, probs_spec, pred_class_spec = read_probs_csv(spec_probs_csv)

    num_classes = probs_spec.shape[1]

    true_class_alerce = true_class_spec.copy()
    true_class_alerce[true_class_alerce == 2] = 1
    
    # read in ALeRCE classes
    df_alerce = pd.read_csv(probs_alerce_csv)
    pred_alerce = df_alerce.alerce_label.to_numpy().astype(str)
    
    ignore_mask = ( pred_alerce == "None" ) | ( pred_alerce == "nan" ) | ( pred_alerce == "SKIP" )
    # ignore true SNe IIn
    ignore_mask = ignore_mask | (true_class_alerce == 2)
    
    true_class_alerce = true_class_alerce[~ignore_mask]
    pred_alerce = pred_alerce[~ignore_mask]
    
    pred_class_spec_alerce = np.array(
        [labels_to_class[x] for x in pred_alerce]
    )

    # import phot dataframe
    names_phot, pred_label_alerce, _, pred_class_phot = read_probs_csv(phot_probs_csv)
    skip_idx = (pred_label_alerce == "SKIP")
    pred_label_alerce, pred_class_phot = pred_label_alerce[~skip_idx], pred_class_phot[~skip_idx]
    
    pred_class_phot_alerce = np.array(
        [labels_to_class[SnClass.from_alerce_to_superphot_format(x)] for x in pred_label_alerce]
    )

    cm_p = confusion_matrix(true_class_spec, pred_class_spec, normalize="pred")
    cm_p_alerce = confusion_matrix(true_class_alerce, pred_class_spec_alerce, normalize="pred")

    true_fracs = np.array(
        [len(true_class_spec[true_class_spec == i]) / len(true_class_spec) for i in range(num_classes)]
    )
    pred_fracs = np.array(
        [len(pred_class_phot[pred_class_phot == i]) / len(pred_class_phot) for i in range(num_classes)]
    )
    alerce_fracs = np.array(
        [
            len(pred_class_phot_alerce[pred_class_phot_alerce == i]) / len(pred_class_phot_alerce)
            for i in range(num_classes)
        ]
    )

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

    save_df = pd.DataFrame(
        {
            "spec_fracs": true_fracs,
            "phot_fracs": pred_fracs,
            "phot_fracs_corr": pred_fracs_corr,
            "alerce_fracs": alerce_fracs,
            "alerce_fracs_corr": alerce_fracs_corr,
        }
    )
    save_df.to_csv(save_path, index=False)


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

    frac_df = pd.read_csv(saved_cf_file)

    true_fracs, pred_fracs, pred_fracs_corr, alerce_fracs, alerce_fracs_corr = frac_df.to_numpy().T

    survey_sn_fracs = get_survey_fracs()
    yse_fracs, psmds_fracs = survey_sn_fracs["YSE"], survey_sn_fracs["PS-MDS"]

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
    _, ax = plt.subplots(figsize=(11, 16))
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


def generate_roc_curve(probs_csv, save_dir):
    """Generate a combined ROC curve of all SN classes.

    Parameters
    ----------
    probs_csv : str
        CSV file where class probabilities are stored.
    save_dir : str
        Where to save the figure.
    """
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    colors = [plt.cm.Set1(i) for i in range(10)]
    fig, ax = plt.subplots(1, 2, figsize=(8, 7))
    ax1, ax2 = ax
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 0.1])
    ax2.set_ylim([0.0, 1.05])
    ax1.set_ylabel("True Positive Rate")
    ratio = 1.2
    plt.locator_params(axis="x", nbins=3)

    legend_lines = []
    fpr = []
    tpr = []

    for ref_label in range(len(classes_to_labels)):
        names, true_labels, probs, preds = read_probs_csv(probs_csv)
        y_true = np.where(true_labels == ref_label, 1, 0)
        y_score = probs[:, ref_label]

        f, t, threshholds = roc_curve(y_true, y_score)
        idx_50 = np.argmin((threshholds - 0.5) ** 2)
        (l,) = ax1.plot(f, t, label="%s" % classes_to_labels[ref_label], c=colors[ref_label])
        ax2.plot(f, t, label="%s" % classes_to_labels[ref_label], c=colors[ref_label])
        legend_lines.append(l)
        # ax1.scatter(f[idx_50], t[idx_50], color=colors[ref_label], s=40)
        ax2.scatter(f[idx_50], t[idx_50], color=colors[ref_label], s=100, marker="d")
        fpr.append(f)
        tpr.append(t)

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes_to_labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes_to_labels)

    for ax_i in ax:
        (l,) = ax_i.plot(
            all_fpr, mean_tpr, label="Macro-averaged", linewidth=3, linestyle="dashed", c="black"
        )

        x_left, x_right = ax_i.get_xlim()
        y_low, y_high = ax_i.get_ylim()

        ax_i.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
        ax_i.yaxis.set_minor_locator(AutoMinorLocator())
        ax_i.xaxis.set_minor_locator(AutoMinorLocator())
        ax_i.set_xlabel("False Positive Rate")

    legend_lines.append(l)

    fig.legend(legend_lines, [*list(labels_to_classes.keys()), "Combined"], loc="lower center", ncol=3)
    plt.savefig(
        os.path.join(save_dir, "roc_all.pdf"), bbox_inches="tight"
    )
    plt.close()


def plot_phase_vs_accuracy(phased_probs_csv, save_dir):
    """Plot classification accuracy as a function of phase.

    Parameters
    ----------
    phased_probs_csv : str
        Where classification probabilities and LC truncated phases are saved.
    save_dir : str
        Where to save the output figures.
    """
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={"hspace": 0})
    ax, ax2 = axes
    _, classes_to_labels = SnClass.get_type_maps()
    allowed_types = np.arange(len(classes_to_labels))

    true_type, phase, _, pred_type = read_probs_csv(phased_probs_csv)
    correct_class = (true_type == pred_type).astype(int)

    legend_lines = []
    for at in allowed_types:
        correct_t = correct_class[true_type == at]
        phase_t = phase[true_type == at]

        bins = np.arange(-16, 52, 4)
        # bins = histedges_equalN(phase_t[phase_t > -18.], 20)

        correct_hist, bin_edges, _ = binned_statistic(phase_t, correct_t, statistic="sum", bins=bins)
        all_hist, _, _ = binned_statistic(phase_t, np.ones(len(phase_t)), statistic="sum", bins=bins)
        acc_hist_t = correct_hist / all_hist
        # acc_hist_comb += acc_hist_t
        (l,) = ax.step(bins, np.append(acc_hist_t, acc_hist_t[-1]), where="post", label=at)
        legend_lines.append(l)

    # acc_hist = acc_hist_comb / 3
    # l, = ax.step(bins, np.append(acc_hist, acc_hist[-1]), where='post', color="k", label="Average")
    # leg_lines.append(l)

    ax.axvline(x=0.0, color="grey", linestyle="dotted")
    ax.set_ylabel("Classification Accuracy")
    ax.set_xlim((-18.0, 48.0))

    # also plot the over/under-classification fraction of each type compared to final classification
    legend_lines = []
    # bins_eq=histedges_equalN(phase[phase > -30.], 20) # all points
    bins_eq = np.arange(-16, 52, 4)
    all_hist, _, _ = binned_statistic(phase, np.ones(len(true_type)), statistic="sum", bins=bins_eq)

    idxs_type = []

    for at in allowed_types:
        eff_num = np.zeros(len(bins_eq) - 1)  # effective numerator
        for at2 in allowed_types:
            idx_sub = true_type == at2
            phase_t = phase[idx_sub]

            bins_eq = np.arange(-16, 52, 4)

            true_hist, _, _ = binned_statistic(phase_t, np.ones(len(phase_t)), statistic="sum", bins=bins_eq)
            frac_hist = true_hist / all_hist  # within each bin, fraction that is that true type

            normed_const = 0.2 / frac_hist

            # get fraction of true type at2 classified as at, and add it to total 'at' fraction
            idx_sub2 = (true_type == at2) & (pred_type == at)
            phase_sub = phase[idx_sub2]
            if len(phase_sub) == 0:
                continue
            pred_hist, _, _ = binned_statistic(
                phase_sub, np.ones(len(phase_sub)), statistic="sum", bins=bins_eq
            )

            eff_num += normed_const * pred_hist

            # acc_hist_comb += acc_hist_t
        pred_frac = eff_num / all_hist
        pred_frac_normed = pred_frac / pred_frac[-1]
        (l,) = ax2.step(bins_eq, np.append(pred_frac_normed, pred_frac_normed[-1]), where="post", label=at)
        legend_lines.append(l)

    ax2.axhline(y=1.0, color="k", xmin=-30, xmax=50, linestyle="--")
    ax2.axvline(x=0.0, color="grey", linestyle="dotted")
    ax2.set_xlabel(r"Phase (days)")
    ax2.set_ylabel("Overprediction Fraction")
    ax2.set_xlim((-18.0, 48.0))
    fig.legend(leg_lines, [classes_to_labels[x] for x in allowed_types], loc="lower center", ncol=3)
    plt.savefig(
        os.path.join(save_dir, "phase_vs_accuracy.pdf"),
        bbox_inches="tight"
    )
    plt.close()


def plot_redshifts_abs_mags(probs_snr_csv, save_dir):
    """
    Plot redshift and absolute magnitude distributions used in the
    redshift-inclusive classifier.

    Parameters
    ----------
    probs_snr_csv : str
        Where probabilities + SNRs are stored.
    save_dir : str
        Where to save figures.
    """
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()
    allowed_types = list(labels_to_classes.keys())
    allowed_classes = [str(labels_to_classes[x]) for x in allowed_types]

    names, classes, redshifts = import_labels_only(
        [
            probs_snr_csv,
        ],
        allowed_classes,
        needs_posteriors=False,
        redshift=True,
    )

    labels = np.array([classes_to_labels[int(x)] for x in classes])
    df = pd.read_csv(probs_snr_csv)
    amplitudes = df.iloc[:, -5].to_numpy()
    app_mags = -2.5 * np.log10(amplitudes) + 26.3

    k_correction = 2.5 * np.log10(1.0 + redshifts)
    dist = cosmo.luminosity_distance([redshifts]).value[0]  # returns dist in Mpc
    abs_mags = app_mags - 5.0 * np.log10(dist * 1e6 / 10.0) + k_correction

    fig, axes = plt.subplots(1, 2)
    z_ax = axes[0]
    mag_ax = axes[1]

    mag_hist, bin_edges = np.histogram(-abs_mags, bins=40, density=True, range=(15, 25))
    bin_width = bin_edges[1] - bin_edges[0]
    mag_cumsum = np.cumsum(mag_hist) * bin_width
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    legend_lines = []

    for at in allowed_types:
        a = at
        features_1_t = -abs_mags[labels == a]
        feature_hist, bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        (l,) = mag_ax.step(-bin_centers, cumsum, where="mid", label=a)
        legend_lines.append(l)

    # l, = mag_ax.step(bin_centers, mag_cumsum, where='mid', c="k", label="Combined", linewidth=2)
    # leg_lines.append(l)
    mag_ax.set_xlabel("Absolute Magnitude")
    # mag_ax.set_yticklabels([])
    # mag_ax.set_yticks([])
    mag_ax.invert_xaxis()

    z_hist, bin_edges = np.histogram(redshifts, bins=40, density=True, range=(-0.1, 0.6))
    bin_width = bin_edges[1] - bin_edges[0]
    z_cumsum = np.cumsum(z_hist) * bin_width
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    for at in allowed_types:
        a = at
        features_1_t = redshifts[labels == a]
        feature_hist, bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        (l,) = z_ax.step(bin_centers, cumsum, where="mid", label=a)

    # l, = z_ax.step(bin_centers, z_cumsum, where='mid', c="k", label="Combined", linewidth=2)
    z_ax.set_xlabel("Redshift")
    z_ax.set_ylabel("Cumulative Fraction")
    # z_ax.set_yticklabels([])
    # z_ax.set_yticks([])

    for ax in axes:
        ratio = 1.0
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    fig.legend(legend_lines, [*allowed_types, "Combined"], loc="lower center", ncol=3)
    plt.savefig(
        os.path.join(save_dir, "abs_mag_hist.pdf"),
        bbox_inches="tight"
    )
    plt.close()


def plot_snr_npoints_vs_accuracy(probs_snr_csv, save_dir):
    """
    Generate plots of number of SNR > 5 points versus
    accuracy, and top 10% SNR versus accuracy.

    TODO: add functionality for only one type.

    Parameters
    ----------
    probs_snr_csv : str
        Where probabilities + SNRs are stored.
    save_dir : str
        Where to save figures.
    """

    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    names, true_type, _, pred_classes = read_probs_csv(probs_snr_csv)
    correct_class = np.where(true_type == pred_classes, 1, 0)

    df = pd.read_csv(probs_snr_csv)
    snr, n_high_snr = df.SNR90, df.nSNR3

    for t in np.unique(true_type):
        snr_t = snr[true_type == t]
        correct_t = correct_class[true_type == t]

        nbins = 8
        while nbins >= 1:
            try:
                snr_vs_accuracy, snr_bin_edges, _ = binned_statistic(
                    snr_t, correct_t, "mean", bins=histedges_equalN(snr_t, nbins)
                )
                break
            except:
                nbins /= 2

        if nbins < 1:
            continue

        snr_vs_accuracy[np.isnan(snr_vs_accuracy)] = 1.0

        plt.step(
            snr_bin_edges,
            np.append(snr_vs_accuracy, snr_vs_accuracy[-1]),
            label=classes_to_labels[t],
            where="post",
        )

    plt.xlim((5, 30))

    plt.xlabel("90th Percentile SNR")
    plt.ylabel("Classification Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(save_dir, "snr_vs_accuracy.pdf"),
        bbox_inches='tight'
    )
    plt.close()

    # second plot
    for t in np.unique(true_type):
        correct_t = correct_class[true_type == t]
        n_high_t = n_high_snr[true_type == t]

        nbins = 8
        while nbins >= 1:
            try:
                n_vs_accuracy, n_bin_edges, _ = binned_statistic(
                    n_high_t, correct_t, "mean", bins=histedges_equalN(n_high_t, nbins)
                )
                break
            except:
                nbins /= 2

        if nbins < 1:
            continue

        n_vs_accuracy[np.isnan(n_vs_accuracy)] = 1.0

        plt.step(
            n_bin_edges, np.append(n_vs_accuracy, n_vs_accuracy[-1]), label=classes_to_labels[t], where="post"
        )

    plt.xlim((8, 100))

    plt.xlabel(r"Number of $\geq 3\sigma$ Datapoints")
    plt.ylabel("Classification Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(save_dir, "n_vs_accuracy.pdf"),
        bbox_inches='tight'
    )
    plt.close()


def plot_snr_hist(probs_snr_csv, save_dir):
    """
    Replicates SNR plots needed for publication.

    Parameters
    ----------
    probs_snr_csv : str
        Where probability + SNR info is stored.
    save_dir : str
        Where to save figure.
    """
    df = pd.read_csv(probs_snr_csv)
    snr, n_snr_3, n_snr_5, n_snr_10 = df.iloc[:, -4:].to_numpy().T
    skip_mask = (df.iloc[:, 1] == "SKIP").to_numpy()

    plt.hist(n_snr_3[~skip_mask], histtype="step", label=r"$3\sigma$", bins=np.arange(0, 603, 3))
    plt.hist(n_snr_5[~skip_mask], histtype="step", label=r"$5\sigma$", bins=np.arange(0, 603, 3))
    plt.hist(n_snr_10[~skip_mask], histtype="step", label=r"$10\sigma$", bins=np.arange(0, 603, 3))
    plt.loglog()
    plt.xlabel("Number of Datapoints at Given SNR")
    plt.ylabel("Number of Lightcurves")
    plt.legend()
    plt.savefig(
        os.path.join(save_dir, "snr_hist.pdf"),
        bbox_inches='tight'
    )
    plt.close()


def compare_mag_distributions(probs_classified, probs_unclassified, save_dir, zeropoint=26.3):
    """
    Generate overlaid magnitude distributions of the classified and unclassified datasets.
    Assumes that unclassified LCs that did not pass the chi-squared cut are marked as "SKIP".

    Parameters
    ----------
    probs_classified : str
        CSV filename where probs of spectroscopic set are stored.
    probs_unclassified : str
        CSV filename where probs of photometric set are stored.
    save_dir : str
        Where to save figure.
    zeropoint : float, optional
        Zeropoint used when converting mags to fluxes. Defaults to 26.3.
    """
    classified_df = pd.read_csv(probs_classified)
    max_flux = classified_df.Fmax.to_numpy()
    max_r_classified = -2.5 * np.log10(max_flux) + zeropoint

    unclassified_df = pd.read_csv(probs_unclassified)
    max_flux = unclassified_df.Fmax.to_numpy()
    max_r_unclassified_all = -2.5 * np.log10(max_flux) + zeropoint

    mask_high_chisquared = (unclassified_df.iloc[:, 1] == "SKIP").to_numpy()
    max_r_unclassified = max_r_unclassified_all[~mask_high_chisquared]
    max_r_skipped = max_r_unclassified_all[mask_high_chisquared]

    plt.hist(
        max_r_classified,
        histtype="stepfilled",
        bins=np.arange(5.0, 21.0, 0.5),
        alpha=0.5,
        label="Classified",
        density=True,
    )
    plt.hist(
        max_r_unclassified,
        histtype="stepfilled",
        bins=np.arange(5.0, 21.0, 0.5),
        alpha=0.5,
        label="Unclassified (included)",
        density=True,
    )

    plt.hist(
        max_r_skipped,
        histtype="stepfilled",
        bins=np.arange(5.0, 21.0, 0.5),
        alpha=0.5,
        label="Unclassified (excluded)",
        density=True,
    )

    plt.yscale("log")
    plt.legend(loc="upper left")
    plt.xlabel("Apparent Magnitude (m)")
    plt.ylabel("Fraction of Lightcurves")
    plt.savefig(
        os.path.join(save_dir, "appm_hist_compare.pdf"),
        bbox_inches='tight',
    )
    plt.close()


def plot_chisquared_vs_accuracy(
    pred_spec_fn,
    pred_phot_fn,
    fits_dir,
    save_dir,
    sampler=None,
):
    """
    Plot chi-squared value histograms for both the spectroscopic and photometric
    datasets, and plot spec chi-squared as a function of classification accuracy.

    TODO: IN PROGRESS

    Parameters
    ----------
    pred_spec_fn : str
        CSV filename where probs of spectroscopic set are stored.
    pred_phot_fn : str
        CSV filename where probs of photometric set are stored.
    save_dir : str
        Where to save figure.
    """
    sn_names, true_classes, _, pred_classes = read_probs_csv(pred_spec_fn)

    correctly_classified = np.where(true_classes == pred_classes, 1, 0)
    mult_posteriors = get_multiple_posterior_samples(sn_names, fits_dir, sampler=sampler)
    
    train_chis = np.array([-1*np.mean(mult_posteriors[x][:,-1]) for x in  sn_names])

    sn_names, _, _, _ = read_probs_csv(pred_phot_fn)

    mult_posteriors = get_multiple_posterior_samples(sn_names, fits_dir, sampler=sampler)
    
    train_chis_phot = np.array([-1*np.mean(mult_posteriors[x][:,-1]) for x in  sn_names])
    
    # plot
    fig, ax2 = plt.subplots(figsize=(7, 4.8))
    ax1 = ax2.twinx()
    bins = np.arange(3.5, 14, 0.5)

    correct_hist, bin_edges, _ = binned_statistic(
        train_chis, correctly_classified, statistic="sum", bins=bins
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    bin_width = bin_edges[1] - bin_edges[0]

    all_hist, _, _ = binned_statistic(train_chis, np.ones(len(train_chis)), statistic="sum", bins=bins)
    all_hist_phot, _, _ = binned_statistic(
        train_chis_phot, np.ones(len(train_chis_phot)), statistic="sum", bins=bins
    )

    ax2.hist(bin_centers, bin_edges, weights=all_hist, color="purple", alpha=0.5, label="Spectroscopic")
    ax2.hist(bin_centers, bin_edges, weights=all_hist_phot, color="red", alpha=0.5, label="Photometric")

    all_hist[all_hist == 0] = np.inf

    acc_hist = correct_hist / all_hist

    idx_keep = (bin_centers < 10) & (bin_centers > 5)
    ax1.step(
        bin_centers[idx_keep], acc_hist[idx_keep], where="mid", color="blue", linewidth=3, label="Accuracy"
    )
    ax1.axvline(x=10, color="black", linestyle="--", linewidth=4, label=r"Phot. $\chi^2$ cutoff")

    # put bin counts on top of bars
    """
    for bin_i in range(len(bins)-1):
        try:
            height = acc_hist[bin_i]
            plt.annotate(
                '%d' % all_hist[bin_i],
                xy=(bin_centers[bin_i], height),
                xytext=(1, 1), # 3 points vertical offset
                textcoords="offset points",
                fontsize=10,
                ha='center', va='bottom'
            )
        except:
            plt.annotate(
                '0',
                xy=(bin_centers[bin_i], height),
                xytext=(1, 1), # 3 points vertical offset
                textcoords="offset points",
                fontsize=10,
                ha='center', va='bottom'
            )
            
    """

    ax2.set_xlabel(r"Reduced $\chi^2$")
    ax1.set_ylabel("Accuracy", va="bottom", rotation=270)
    ax2.set_ylabel("Counts")
    ax2.legend()

    ax1.yaxis.label.set_color("blue")
    ax1.spines["right"].set_color("blue")
    ax1.tick_params(axis="y", colors="blue")

    ax1.legend(loc="lower right")
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    plt.savefig(
        os.path.join(save_dir, "chisq_vs_accuracy.pdf"),
        bbox_inches="tight"
    )
    plt.close()


def plot_model_metrics(metrics, num_epochs, plot_name, metrics_dir):
    """Plots training and validation results and exports them to files.

    Parameters
    ----------
    metrics: tuple
        Train and validation accuracies and losses.
    num_epochs: int
        The total number of epochs.
    plot_name: str
        The name for the plot figure files.
    metrics_dir: str
        Where to store the plot figures.
    """
    train_acc, train_loss, val_acc, val_loss = metrics

    # Plot accuracy
    plt.plot(np.arange(0, num_epochs), train_acc, label="Training")
    plt.plot(np.arange(0, num_epochs), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(metrics_dir, f"accuracy_{plot_name}.pdf"),
        bbox_inches='tight',
    )
    plt.close()

    # Plot loss
    plt.plot(np.arange(0, num_epochs), train_loss, label="Training")
    plt.plot(np.arange(0, num_epochs), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(
        os.path.join(metrics_dir, f"loss_{plot_name}.pdf"),
        bbox_inches='tight'
    )
    plt.close()
