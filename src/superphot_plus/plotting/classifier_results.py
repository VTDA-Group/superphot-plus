"""This module provides various functions for analyzing and visualizing
classification results."""

import os, glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from matplotlib.ticker import AutoMinorLocator

from astropy.cosmology import Planck13 as cosmo
from scipy.stats import binned_statistic

from superphot_plus.plotting.format_params import set_global_plot_formatting, CUSTOM_COLORSET
from superphot_plus.file_utils import get_multiple_posterior_samples
from superphot_plus.format_data_ztf import import_labels_only, retrieve_posterior_set
from superphot_plus.posterior_samples import PosteriorSamples

from superphot_plus.plotting.utils import (
    histedges_equalN, read_probs_csv,
    get_survey_fracs, retrieve_four_class_info,
    calc_precision_recall,
    rebin_prec_recall,
    roc_curve_w_uncertainties,
    calc_calibration_curve
)
from superphot_plus.supernova_class import SupernovaClass as SnClass

set_global_plot_formatting()


def save_class_fractions(spec_probs_csv, probs_alerce_csv, phot_probs_csv, probs_alerce_phot_csv, save_path):
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
    labels_to_class, classes_to_labels = SnClass.get_type_maps()

    # import spec dataframe
    _, true_class_spec, probs_spec, pred_class_spec, _, _ = read_probs_csv(spec_probs_csv)

    num_classes = probs_spec.shape[1]

    true_class_alerce = true_class_spec.copy()
    true_class_alerce[true_class_alerce == 2] = 1

    # read in ALeRCE classes
    df_alerce = pd.read_csv(probs_alerce_csv)
    pred_alerce = df_alerce.alerce_label.to_numpy().astype(str)

    ignore_mask = (pred_alerce == "None") | (pred_alerce == "nan") | (pred_alerce == "SKIP")
    # ignore true SNe IIn
    ignore_mask = ignore_mask | (true_class_alerce == 2)

    true_class_alerce = true_class_alerce[~ignore_mask]
    pred_alerce = pred_alerce[~ignore_mask]

    pred_class_spec_alerce = np.array([labels_to_class[x] for x in pred_alerce])

    # import phot dataframe
    pred_class_phot = read_probs_csv(phot_probs_csv)[3]
    pred_class_phot_alerce = retrieve_four_class_info(phot_probs_csv, probs_alerce_phot_csv)[4]

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
            len(pred_class_phot_alerce[pred_class_phot_alerce == classes_to_labels[i]]) / len(pred_class_phot_alerce) for i in range(num_classes)
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

    for i in range(5):
        if i == 0:
            bottom = 0
        else:
            bottom = np.sum(combined_fracs[0:i], axis=0)
        stacked_bar = ax.bar(
            labels,
            combined_fracs[i],
            width,
            bottom=bottom,
            label=classes_to_labels[i],
        )
        for j, fracs_j in enumerate(combined_fracs[i]):
            if fracs_j == 0.0:
                continue
            barj = stacked_bar.patches[j]
            # Create annotation
            ax.annotate(
                round(fracs_j, 3),
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

    colors = CUSTOM_COLORSET
    fig, double_axes = plt.subplots(1, 2, figsize=(8, 7))
    ax1, ax2 = double_axes
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 0.1])
    ax2.set_ylim([0.0, 1.05])
    ax1.set_ylabel("True Positive Rate")
    ratio = 1.2
    plt.locator_params(axis="x", nbins=3)

    legend_lines = []
    fprs = []
    tprs = []

    for ref_class, ref_label in enumerate(classes_to_labels):
        _, true_classes, probs, _, folds, _ = read_probs_csv(probs_csv)
        y_true = np.where(true_classes == ref_class, 1, 0)
        y_score = probs[:, ref_class]

        t, fpr, tpr, tpr_err = roc_curve_w_uncertainties(y_true, y_score, folds)
        idx_50 = np.argmin((t - 0.5) ** 2)
        (legend_line,) = ax1.step(fpr, tpr, label=ref_label, c=colors[ref_class], where='post')
        ax1.fill_between(
            fpr, tpr-tpr_err, tpr+tpr_err,
            color=colors[ref_class], step='post', alpha=0.2
        )
        ax2.step(fpr, tpr, label=ref_label, c=colors[ref_class], where='post')
        legend_lines.append(legend_line)
        ax2.fill_between(
            fpr, tpr-tpr_err, tpr+tpr_err,
            color=colors[ref_class], step='post', alpha=0.2
        )
        ax2.scatter(
            (fpr[idx_50] + fpr[idx_50 + 1]) / 2, tpr[idx_50],
            color=colors[ref_class], s=100, marker="d", zorder=1000
        )
        #fprs.append(fpr)
        #tprs.append(tpr)

    ax1.plot(
        [0, 1], [0,1],
        c="#BBBBBB", linestyle='dotted'
    )
    
    """
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes_to_labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes_to_labels)
    """
    for ax_i in double_axes:
        """
        (legend_line,) = ax_i.plot(
            all_fpr, mean_tpr, label="Macro-averaged", linewidth=3, linestyle="dashed", c="black"
        )
        """
        x_left, x_right = ax_i.get_xlim()
        y_low, y_high = ax_i.get_ylim()

        ax_i.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
        ax_i.yaxis.set_minor_locator(AutoMinorLocator())
        ax_i.xaxis.set_minor_locator(AutoMinorLocator())
        ax_i.set_xlabel("False Positive Rate")

    #legend_lines.append(legend_line)
    #legend_keys = [*list(labels_to_classes.keys()), "Combined"]
    legend_keys = list(labels_to_classes.keys())
    fig.legend(legend_lines, legend_keys, loc="lower center", ncol=3)
    plt.savefig(os.path.join(save_dir, "roc_all.pdf"), bbox_inches="tight")
    plt.close()


def plot_precision_recall(probs_csv, save_dir, plot_fleet=True):
    """Show how adjusting binary threshholds impact
    purity and completeness values."""
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    colors = CUSTOM_COLORSET
    #fig, double_axes = plt.subplots(1, 2, figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(6,8))
    ax1 = ax
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    #ax2.set_xlim([0.0, 1.05])
    #ax2.set_ylim([0.0, 1.05])
    ratio = 1.0
    ax1.set_ylabel("Purity")
    #ax2.set_ylabel("Purity (Rescaled)")
    plt.locator_params(axis="x", nbins=3)

    legend_lines = []

    for ref_class, ref_label in enumerate(classes_to_labels):
        _, true_classes, probs, _, folds, _ = read_probs_csv(probs_csv)
        y_true = np.where(true_classes == ref_class, 1, 0)
        y_score = probs[:, ref_class]

        prevalence = sum(y_true) / len(y_true)
        t, r, p, perr = calc_precision_recall(y_true, y_score, folds)

        idx_50 = np.argmin((t - 0.5) ** 2)
        (legend_line,) = ax1.step(
            r, p, label=ref_label, c=colors[ref_class], where='post'
        )
        ax1.fill_between(r, p-perr, p+perr, alpha=0.2, color=colors[ref_class], step='post')
        legend_lines.append(legend_line)
        ax1.scatter(
            (r[idx_50]+r[idx_50+1])/2, p[idx_50],
            color=colors[ref_class], s=100, marker="d", zorder=1000
        )
        #ax1.axhline(y=prevalence, linestyle='dashed', linewidth=1, color=colors[ref_class])
        """
        # rescaled to baseline
        p_scaled = (p - prevalence) / (1 - prevalence)
        perr_scaled = perr / (1 - prevalence)

        ax2.step(
            r, p_scaled, label=ref_label, c=colors[ref_class], where='post'
        )
        ax2.fill_between(
            r, p_scaled-perr_scaled, p_scaled+perr_scaled,
            alpha=0.2, color=colors[ref_class], step='post'
        )
        ax2.scatter(
            (r[idx_50]+r[idx_50+1])/2, p_scaled[idx_50],
            color=colors[ref_class], s=100, marker="d", zorder=1000
        )
        """
        # print AUPR value
        aupr = np.sum((r[1:] - r[:-1]) * p[:-1])
        aupr_min = np.sum((r[1:] - r[:-1]) * (p-perr)[:-1])
        aupr_max = np.sum((r[1:] - r[:-1]) * (p+perr)[:-1])
        print(ref_label, aupr, aupr_min, aupr_max)
        
    if plot_fleet:
        fleet_fn = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../..', 'data', 'SLSN_late-time.txt')
        )
        prevalence = 187 / 4780
        fleet_df = pd.read_csv(fleet_fn, sep='\s+')
        p_fleet = fleet_df['Purity'].to_numpy()
        perr_fleet = fleet_df['PurityStd'].to_numpy()
        r_fleet = fleet_df['Completeness'].to_numpy()
        rerr_fleet = fleet_df['CompletenessStd'].to_numpy()
        t_fleet = fleet_df['P(SLSN-I)'].to_numpy()
        
        rbin_fleet, pbin_fleet, pmin_fleet, pmax_fleet = rebin_prec_recall(
            t_fleet, r_fleet, rerr_fleet, p_fleet, perr_fleet
        )
        idx_50 = np.argmin((t_fleet - 0.5) ** 2)
        
        pscaled_fleet = (pbin_fleet - prevalence) / (1 - prevalence)
        pmin_scaled = (pmin_fleet - prevalence) / (1 - prevalence)
        pmax_scaled = (pmax_fleet - prevalence) / (1 - prevalence)
        
        (legend_line,) = ax1.step(
            rbin_fleet, pbin_fleet, label='FLEET (SLSN-I)', c=colors[5], where='post'
        )
        ax1.fill_between(
            rbin_fleet, pmin_fleet, pmax_fleet, alpha=0.2, color=colors[5], step='post'
        )
        legend_lines.append(legend_line)
        ax1.scatter(
            r_fleet[idx_50], p_fleet[idx_50],
            color=colors[5], s=100, marker="d", zorder=1000
        )
        
        aupr = np.sum((rbin_fleet[:-1] - rbin_fleet[1:]) * pbin_fleet[:-1])
        aupr_min = np.sum((rbin_fleet[:-1] - rbin_fleet[1:]) * pmin_fleet[:-1])
        aupr_max = np.sum((rbin_fleet[:-1] - rbin_fleet[1:]) * pmax_fleet[:-1])
        print("FLEET", aupr, aupr_min, aupr_max)
        """
        ax2.step(
            rbin_fleet, pscaled_fleet, label='FLEET (SLSN-I)', c=colors[5], where='post'
        )
        ax2.fill_between(
            rbin_fleet, pmin_scaled, pmax_scaled, alpha=0.2, color=colors[5], step='post'
        )
        ax2.scatter(
            (r_fleet[idx_50]+r_fleet[idx_50+1])/2, pscaled_fleet[idx_50],
            color=colors[5], s=100, marker="d", zorder=1000
        )
        """
        
    #for ax in double_axes:
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()

    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel("Completeness")

    #legend_lines.append(legend_line)
    legend_keys = [*list(labels_to_classes.keys()), "FLEET (SLSN-I)"]
    fig.legend(legend_lines, legend_keys, loc="lower center", ncol=3)
    plt.savefig(os.path.join(save_dir, "prec_recall_all.pdf"), bbox_inches="tight")
    plt.close()

    
def plot_metrics_over_mjd(mjd_bins, p_matrix, c_matrix, save_dir):
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    colors = CUSTOM_COLORSET
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={"hspace": 0.1})
    ax, ax2 = axes
    ratio = 1.0
    ax2.set_ylabel("Purity")
    ax.set_ylabel("Completeness")
    ax.set_ylim((0, 1))
    ax2.set_ylim((0, 1))
    plt.locator_params(axis="x", nbins=3)

    legend_lines = []
    for i in range(5):
        mean_p = np.mean(p_matrix[i], axis=0)
        mean_c = np.mean(c_matrix[i], axis=0)
        p_err = np.std(p_matrix[i], axis=0)
        c_err = np.std(c_matrix[i], axis=0)
        
        (legend_line,) = ax.step(
            mjd_bins, mean_c, where='post',
            c=colors[i], label=classes_to_labels[i]
        )
        ax.fill_between(
            mjd_bins, mean_c - c_err, mean_c + c_err,
            color=colors[i], alpha=0.2, step='post'
        )
        ax2.step(
            mjd_bins, mean_p, where='post', c=colors[i]
        )
        ax2.fill_between(
            mjd_bins, mean_p - p_err, mean_p + p_err,
            color=colors[i], alpha=0.2, step='post'
        )
        legend_lines.append(legend_line)
    
    ax2.set_xlabel("MJD")
    fig.legend(loc="lower center", ncol=3)
    plt.savefig(os.path.join(save_dir, "metrics_over_mjd.pdf"), bbox_inches="tight")
    plt.close()
    
    
def plot_phase_vs_accuracy(phased_probs_dir, all_probs_csv, save_dir):
    """Plot classification accuracy as a function of phase.

    Parameters
    ----------
    phased_probs_dir : str
        Where classification probabilities and LC truncated phases are saved.
    save_dir : str
        Where to save the output figures.
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 16), gridspec_kw = {'hspace':0.05})
    ax, ax2, ax3 = axes
    _, classes_to_labels = SnClass.get_type_maps()
    allowed_types = np.arange(len(classes_to_labels))

    phases = []
    accs_full_means = []
    accs_full_stddevs = []
    accs_early_means = []
    accs_early_stddevs = []
    fracs_early_means = []
    fracs_early_stddevs = []
    fracs_full_means = []
    fracs_full_stddevs = []
    f1_early_means = []
    f1_early_stddevs = []
    f1_full_means = []
    f1_full_stddevs = []
    
    all_probs_files = glob.glob(
        os.path.join(
            phased_probs_dir,
            "full_*_concat.csv"
        )
    )
    
    full_probs_df = pd.read_csv(all_probs_csv)
    all_true_labels = full_probs_df.Label.to_numpy()
    class_counts = [len(all_true_labels[l == all_true_labels]) for l in allowed_types]
    class_fracs = np.asarray(class_counts) / np.sum(class_counts)
    
    n_phases = len(all_probs_files)
    
    for probs_file_full in all_probs_files:
        _, true_type, _, pred_type, folds, _ = read_probs_csv(probs_file_full)
        phase_counts = [len(true_type[l == true_type]) for l in allowed_types]
        phase_fracs = np.asarray(phase_counts) / np.sum(phase_counts)
    
        correct_class = (true_type == pred_type).astype(int)
        acc_mu_single = []
        acc_std_single = []
        fracs_mu_single = []
        fracs_std_single = []
        f1_mu_single = []
        f1_std_single = []
        
        for i, allowed_type in enumerate(allowed_types):
            accs = []
            fracs = []
            f1s = []
            for f in range(10):
                correct_t = correct_class[(folds == f) & (true_type == allowed_type)]
                completeness = np.sum(correct_t) / len(true_type[(true_type == allowed_type) & (folds == f)])
                all_preds = pred_type[(pred_type == allowed_type) & (folds == f)]
                all_trues = true_type[(pred_type == allowed_type) & (folds == f)]
                adj_pred = np.sum([
                    class_fracs[j] * len(all_preds[all_trues == at2]) / phase_fracs[j] for j, at2 in enumerate(allowed_types)
                ])
                purity = class_fracs[i] * np.sum(correct_t) / adj_pred / phase_fracs[i]
                accs.append(completeness)
                fracs.append(purity)
                if purity == 0 and completeness == 0:
                    f1s.append(0)
                else:
                    f1s.append(2 * purity * completeness / (purity + completeness))
            
            fracs_mu_single.append(np.nanmean(fracs))
            fracs_std_single.append(np.nanstd(fracs))
            acc_mu_single.append(np.nanmean(accs))
            acc_std_single.append(np.nanstd(accs))
            f1_mu_single.append(np.nanmean(f1s))
            f1_std_single.append(np.nanstd(f1s))
            
            
        accs_full_means.append(acc_mu_single)
        accs_full_stddevs.append(acc_std_single)
        fracs_full_means.append(fracs_mu_single)
        fracs_full_stddevs.append(fracs_std_single)
        f1_full_means.append(f1_mu_single)
        f1_full_stddevs.append(f1_std_single)
        
        phase = probs_file_full.split("/")[-1].split("_")[1]

        if round(float(phase), 2) == 0.61:
            print("PHASE ZERO FULL")
            print(acc_mu_single, acc_std_single)
            print(fracs_mu_single, fracs_std_single)
            
        if round(float(phase), 2) == 70.00:
            print("PHASE LATE FULL")
            print(acc_mu_single, acc_std_single)
            print(fracs_mu_single, fracs_std_single)
                
        phases.append(float(phase))
        probs_file_early = os.path.join(
            phased_probs_dir,
            f"early_{phase}_concat.csv"
        )
        _, true_type, _, pred_type, folds, _ = read_probs_csv(probs_file_early)
        correct_class = (true_type == pred_type).astype(int)
        acc_mu_single = []
        acc_std_single = []
        fracs_mu_single = []
        fracs_std_single = []
        f1_mu_single = []
        f1_std_single = []
        
        for i, allowed_type in enumerate(allowed_types):
            accs = []
            fracs = []
            f1s = []
            for f in range(10):
                correct_t = correct_class[(folds == f) & (true_type == allowed_type)]
                completeness = np.sum(correct_t) / len(true_type[(true_type == allowed_type) & (folds == f)])
                all_preds = pred_type[(pred_type == allowed_type) & (folds == f)]
                all_trues = true_type[(pred_type == allowed_type) & (folds == f)]
                adj_pred = np.sum([
                    class_fracs[j] * len(all_preds[all_trues == at2]) / phase_fracs[j] for j, at2 in enumerate(allowed_types)
                ])
                purity = class_fracs[i] * np.sum(correct_t) / adj_pred / phase_fracs[i]
                accs.append(completeness)
                fracs.append(purity)
                if purity == 0 and completeness == 0:
                    f1s.append(0)
                else:
                    f1s.append(2 * purity * completeness / (purity + completeness))
            
            fracs_mu_single.append(np.nanmean(fracs))
            fracs_std_single.append(np.nanstd(fracs))
            acc_mu_single.append(np.nanmean(accs))
            acc_std_single.append(np.nanstd(accs))
            f1_mu_single.append(np.nanmean(f1s))
            f1_std_single.append(np.nanstd(f1s))
            
        accs_early_means.append(acc_mu_single)
        accs_early_stddevs.append(acc_std_single)
        fracs_early_means.append(fracs_mu_single)
        fracs_early_stddevs.append(fracs_std_single)
        f1_early_means.append(f1_mu_single)
        f1_early_stddevs.append(f1_std_single)
        
        if round(float(phase), 2) == 0.61:
            print("PHASE ZERO EARLY")
            print(acc_mu_single, acc_std_single)
            print(fracs_mu_single, fracs_std_single)
            
        if round(float(phase), 2) == 70.00:
            print("PHASE LATE EARLY")
            print(acc_mu_single, acc_std_single)
            print(fracs_mu_single, fracs_std_single)
            
        
    sort_idx = np.argsort(phases)
    phases = np.asarray(phases)[sort_idx]
    accs_early_means = np.asarray(accs_early_means)[sort_idx].T
    accs_early_stddevs = np.asarray(accs_early_stddevs)[sort_idx].T
    accs_full_means = np.asarray(accs_full_means)[sort_idx].T
    accs_full_stddevs = np.asarray(accs_full_stddevs)[sort_idx].T
    fracs_early_means = np.asarray(fracs_early_means)[sort_idx].T
    fracs_early_stddevs = np.asarray(fracs_early_stddevs)[sort_idx].T
    fracs_full_means = np.asarray(fracs_full_means)[sort_idx].T
    fracs_full_stddevs = np.asarray(fracs_full_stddevs)[sort_idx].T
    f1_early_means = np.asarray(f1_early_means)[sort_idx].T
    f1_early_stddevs = np.asarray(f1_early_stddevs)[sort_idx].T
    f1_full_means = np.asarray(f1_full_means)[sort_idx].T
    f1_full_stddevs = np.asarray(f1_full_stddevs)[sort_idx].T
    
    legend_lines = []
    for i, allowed_type in enumerate(allowed_types):
        (legend_line,) = ax.plot(
            phases, accs_full_means[i], label=allowed_type, color=CUSTOM_COLORSET[i]
        )
        ax.plot(
            phases, accs_early_means[i], linestyle='dashed', color=CUSTOM_COLORSET[i]
        )
        ax.fill_between(
            phases, accs_full_means[i]-accs_full_stddevs[i],
            accs_full_means[i]+accs_full_stddevs[i], alpha=0.2, color=CUSTOM_COLORSET[i]
        )
        legend_lines.append(legend_line)
        
        ax2.plot(
            phases, fracs_full_means[i], label=allowed_type, color=CUSTOM_COLORSET[i]
        )
        ax2.plot(
            phases, fracs_early_means[i], linestyle='dashed', color=CUSTOM_COLORSET[i]
        )
        ax2.fill_between(
            phases, fracs_full_means[i]-fracs_full_stddevs[i],
            fracs_full_means[i]+fracs_full_stddevs[i],
            alpha=0.2, color=CUSTOM_COLORSET[i]
        )
        
        ax3.plot(
            phases, f1_full_means[i], label=allowed_type, color=CUSTOM_COLORSET[i]
        )
        ax3.plot(
            phases, f1_early_means[i], linestyle='dashed', color=CUSTOM_COLORSET[i]
        )
        ax3.fill_between(
            phases, f1_full_means[i]-f1_full_stddevs[i],
            f1_full_means[i]+f1_full_stddevs[i],
            alpha=0.2, color=CUSTOM_COLORSET[i]
        )
        
    (legend_line,) = ax3.plot(
        phases, np.mean(f1_full_means, axis=0), label="Macro", color='k'
    )
    ax3.plot(
        phases, np.mean(f1_early_means, axis=0), linestyle='dashed', color='k'
    )
    legend_lines.append(legend_line)
    
    ax.plot(
        phases, np.mean(accs_full_means, axis=0), label="Macro", color='k'
    )
    ax.plot(
        phases, np.mean(accs_early_means, axis=0), linestyle='dashed', color='k'
    )
    
    ax2.plot(
        phases, np.mean(fracs_full_means, axis=0), label="Macro", color='k'
    )
    ax2.plot(
        phases, np.mean(fracs_early_means, axis=0), linestyle='dashed', color='k'
    )

    ax.set_ylabel("Completeness")
    ax.set_ylim((0, 1))
    
    ax2.set_ylabel("Estimated Purity")
    ax2.set_ylim((0, 1))
    
    ax3.set_ylabel("Estimated F1")
    ax3.set_ylim((0, 1))
        
    ax.axvline(x=0.0, color="grey", linestyle="dotted")
    ax2.axvline(x=0.0, color="grey", linestyle="dotted")
    ax3.axvline(x=0.0, color="grey", linestyle="dotted")
    ax3.set_xlabel(r"Phase (days)")
    fig.legend(legend_lines, [*[classes_to_labels[x] for x in allowed_types], "Macro"], loc="lower center", ncol=3)
    plt.savefig(os.path.join(save_dir, "phase_vs_accuracy.pdf"), bbox_inches="tight")
    plt.close()


def plot_redshifts_abs_mags(probs_snr_csv, training_csv, fits_dir, save_dir, sampler="dynesty"):
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

    training_df = pd.read_csv(training_csv)

    # labels = np.array([classes_to_labels[int(x)] for x in classes])
    probs_dataframe = pd.read_csv(probs_snr_csv)
    names = probs_dataframe.Name.to_numpy()
    labels = probs_dataframe.Label.to_numpy()
    labels = np.array([classes_to_labels[x] for x in labels])
    redshifts = []
    amplitudes = []
    
    for n in names:
        z = training_df[training_df.NAME == n].Z.iloc[0]
        redshifts.append(z)
        
        ps = PosteriorSamples.from_file(
            name=n,
            input_dir=fits_dir,
            sampling_method=sampler
        )
        amplitudes.append(ps.max_flux)
    
    redshifts = np.array(redshifts)
    amplitudes = np.array(amplitudes)
    
    app_mags = -2.5 * np.log10(amplitudes) + 26.3

    k_correction = 2.5 * np.log10(1.0 + redshifts)
    dist = cosmo.luminosity_distance([redshifts]).value[0]  # returns dist in Mpc
    abs_mags = app_mags - 5.0 * np.log10(dist * 1e6 / 10.0) + k_correction

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    z_ax = axes[0]
    mag_ax = axes[1]

    _, bin_edges = np.histogram(-abs_mags, bins=40, density=True, range=(15, 25))
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    legend_lines = []

    for allowed_type in allowed_types:
        features_1_t = -abs_mags[labels == allowed_type]
        feature_hist, bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        (legend_line,) = mag_ax.step(-bin_centers, cumsum, where="mid", label=allowed_type)
        legend_lines.append(legend_line)

    mag_ax.set_xlabel("Absolute Magnitude")
    mag_ax.invert_xaxis()

    _, bin_edges = np.histogram(redshifts, bins=40, density=True, range=(-0.1, 0.6))
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    for allowed_type in allowed_types:
        features_1_t = redshifts[labels == allowed_type]
        feature_hist, bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        z_ax.step(bin_centers, cumsum, where="mid", label=allowed_type)

    z_ax.set_xlabel("Redshift")
    z_ax.set_ylabel("Cumulative Fraction")

    for ax in axes:
        ratio = 1.0
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    fig.legend(legend_lines, [*allowed_types, "Combined"], loc="lower center", ncol=3)
    plt.savefig(os.path.join(save_dir, "abs_mag_hist.pdf"), bbox_inches="tight")
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

    _, classes_to_labels = SnClass.get_type_maps()

    names, true_type, _, pred_classes, folds, _ = read_probs_csv(probs_snr_csv)
    correct_class = np.where(true_type == pred_classes, 1, 0)

    df = pd.read_csv(probs_snr_csv)
    snr, n_high_snr = df.SNR90, df.nSNR3
          
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 10)) #gridspec_kw = {'hspace': 0.3})
    ax1, ax2 = ax
    
    for unique_type in np.unique(true_type):
        snr_t = snr[true_type == unique_type]
        correct_t = correct_class[true_type == unique_type]

        nbins = 8
        snr_vs_accuracy, snr_bin_edges, _ = binned_statistic(
            snr_t, correct_t, "mean", bins=histedges_equalN(snr_t, nbins)
        )
        snr_vs_accuracy[np.isnan(snr_vs_accuracy)] = 1.0

        ax1.step(
            snr_bin_edges,
            np.append(snr_vs_accuracy, snr_vs_accuracy[-1]),
            #label=classes_to_labels[unique_type],
            where="post",
        )

    ax1.set_xlim((8, 30))
    #ax1.set_ylim((0.15, 1.05))
    ax1.set_xscale('log')
    ax1.set_xlabel("90th Percentile SNR")
    ax1.set_ylabel("Class Completeness")
    #plt.legend()
    #plt.savefig(os.path.join(save_dir, "snr_vs_accuracy.pdf"))
    
    #plt.close()

    #fig, ax = plt.subplots(figsize=(6.4, 6.0))
    # second plot
    for unique_type in np.unique(true_type):
        correct_t = correct_class[true_type == unique_type]
        n_high_t = n_high_snr[true_type == unique_type]

        nbins = 8
        n_vs_accuracy, n_bin_edges, _ = binned_statistic(
            n_high_t, correct_t, "mean", bins=histedges_equalN(n_high_t, nbins)
        )
        
        if nbins < 1:
            continue

        n_vs_accuracy[np.isnan(n_vs_accuracy)] = 1.0

        ax2.step(
            n_bin_edges,
            np.append(n_vs_accuracy, n_vs_accuracy[-1]),
            label=classes_to_labels[unique_type],
            where="post",
        )

    ax2.set_xlim((10, 200))
    #ax2.set_ylim((0.15, 1.05))
    ax2.set_xscale('log')
    ax2.set_xlabel(r"Number of $\geq 3\sigma$ Datapoints")
    ax2.set_ylabel("Class Completeness")
    fig.legend(loc="lower center", ncols=3)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, bottom=0.2, top=0.95)
    
    plt.savefig(os.path.join(save_dir, "n_snr_vs_accuracy.pdf"))
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
    n_snr_3, n_snr_5, n_snr_10 = df.iloc[:, -3:].to_numpy().T
    skip_mask = (df.iloc[:, 1] == "SKIP").to_numpy()
    bins = np.arange(0, 603, 3)

    plt.hist(n_snr_3[~skip_mask], histtype="step", label=r"$SNR \geq 3$", bins=bins)
    plt.hist(n_snr_5[~skip_mask], histtype="step", label=r"$SNR \geq 5$", bins=bins)
    plt.hist(n_snr_10[~skip_mask], histtype="step", label=r"$SNR \geq 10$", bins=bins)
    plt.loglog()
    plt.xlabel("Number of Datapoints at Given SNR")
    plt.ylabel("Number of Light Curves")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "snr_hist.pdf"), bbox_inches="tight")
    plt.close()


def compare_mag_distributions(
    probs_classified,
    probs_unclassified,
    all_spec_csv,
    all_phot_csv,
    fits_dir,
    fits_dir_phot,
    save_dir,
    zeropoint=26.3,
    sampler='dynesty',
    allowed_types = SnClass.get_alternative_namings().keys()
):
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
    classified_names = classified_df.Name.to_numpy()
    all_names = pd.read_csv(all_spec_csv).NAME.to_numpy()
    all_labels = pd.read_csv(all_spec_csv).CLASS.to_numpy()
    label_mask = [SnClass.canonicalize(l) in allowed_types for l in all_labels]
    all_names = all_names[label_mask]
    
    max_flux = []
    for n in all_names:
        ps = PosteriorSamples.from_file(
            name = n,
            input_dir = fits_dir,
            sampling_method =sampler
        )
        max_flux.append(ps.max_flux)
    max_flux = np.array(max_flux)
    max_r_classified_all = -2.5 * np.log10(max_flux) + zeropoint
    mask_high_chisquared = np.isin(all_names, classified_names)
    max_r_classified = max_r_classified_all[mask_high_chisquared]
    max_r_classified_skipped = max_r_classified_all[~mask_high_chisquared]
    
    unclassified_df = pd.read_csv(probs_unclassified)
    unclassified_names = unclassified_df.Name.to_numpy()
    all_phot_names = pd.read_csv(all_phot_csv).NAME.to_numpy()
    
    max_flux = []
    for n in all_phot_names:
        ps = PosteriorSamples.from_file(
            name = n,
            input_dir = fits_dir_phot,
            sampling_method =sampler
        )
        max_flux.append(ps.max_flux)
    max_flux = np.array(max_flux)
    
    max_r_unclassified_all = -2.5 * np.log10(np.asarray(max_flux)) + zeropoint
    mask_high_chisquared = np.isin(all_phot_names, unclassified_names)
    max_r_unclassified = max_r_unclassified_all[mask_high_chisquared]
    max_r_unclassified_skipped = max_r_unclassified_all[~mask_high_chisquared]

    plt.hist(
        max_r_classified,
        histtype="step",
        bins=np.arange(5.0, 21.0, 0.5),
        label="Spec. (included)",
        density=True,
        linewidth=2,
    )
    plt.hist(
        max_r_classified_skipped,
        histtype="step",
        bins=np.arange(5.0, 21.0, 0.5),
        label="Spec. (excluded)",
        density=True,
        linewidth=2,

    )
    plt.hist(
        max_r_unclassified,
        histtype="step",
        bins=np.arange(5.0, 21.0, 0.5),
        label="Phot. (included)",
        density=True,
        linewidth=2,

    )
    plt.hist(
        max_r_unclassified_skipped,
        histtype="step",
        bins=np.arange(5.0, 21.0, 0.5),
        label="Phot. (excluded)",
        density=True,
        linewidth=2,

    )

    plt.yscale("log")
    plt.legend(loc="upper left")
    plt.xlabel("Peak Apparent Magnitude")
    plt.ylabel("Fraction of Light Curves")
    plt.savefig(
        os.path.join(save_dir, "appm_hist_compare.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def plot_chisquared_vs_accuracy(
    pred_spec_fn,
    all_spec_csv,
    all_phot_csv,
    fits_dir,
    fits_dir_phot,
    save_dir,
    sampler=None,
    allowed_types = SnClass.get_alternative_namings().keys()
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
    sn_names, true_classes, _, pred_classes, _, _ = read_probs_csv(pred_spec_fn)
    sn_names_all = pd.read_csv(all_spec_csv).NAME.to_numpy()
    sn_types_all = pd.read_csv(all_spec_csv).CLASS.to_numpy()

    if allowed_types is not None:
        mask = [
            SnClass.canonicalize(y) in allowed_types for y in sn_types_all
        ]
        sn_names_all = sn_names_all[mask]
        sn_types_all = sn_types_all[mask]

    correctly_classified = np.where(true_classes == pred_classes, 1, 0)
    ps_set = retrieve_posterior_set(sn_names_all, fits_dir, sampler=sampler)
    spec_mask = np.isin(sn_names_all, sn_names)
    train_chis = np.array([np.median(x.samples[:, -1]) for x in ps_set])
    train_chis_spec = np.array([np.mean(x.samples[:, -1]) for x in ps_set[spec_mask]])
    
    sn_names_phot = pd.read_csv(all_phot_csv).NAME.to_numpy()
    ps_set = retrieve_posterior_set(sn_names_phot, fits_dir_phot, sampler=sampler)
    spec_mask = np.isin(sn_names_all, sn_names)
    train_chis_phot = np.array([np.median(x.samples[:, -1]) for x in ps_set])

    # plot
    _, ax2 = plt.subplots(figsize=(7, 4.8))
    ax1 = ax2.twinx()
    bins = np.arange(0, 4.0, 0.1)

    correct_hist, bin_edges, _ = binned_statistic(
        train_chis_spec, correctly_classified, statistic="sum", bins=bins
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    all_hist_spec, _, _ = binned_statistic(train_chis, np.ones(len(train_chis)), statistic="sum", bins=bins)
    
    all_hist, _, _ = binned_statistic(train_chis_spec, np.ones(len(train_chis_spec)), statistic="sum", bins=bins)

    all_hist_phot, _, _ = binned_statistic(
        train_chis_phot, np.ones(len(train_chis_phot)), statistic="sum", bins=bins
    )

    ax2.hist(bin_centers, bin_edges, weights=all_hist_spec, alpha=0.5, label="Spectroscopic")
    ax2.hist(bin_centers, bin_edges, weights=all_hist_phot, alpha=0.5, label="Photometric")
    
    ax2.set_yscale("log")

    all_hist[all_hist == 0] = np.inf

    acc_hist = correct_hist / all_hist
    acc_cut = acc_hist[bins[:-1] < 1.2]
    acc_cut = np.append(acc_cut, acc_cut[-1])
    ax1.step(
        bins[bins < 1.215],
        acc_cut,
        where="post", color='#228833', linewidth=3, label="Accuracy"
    )
    ax1.axvline(x=1.2, color="black", linestyle="--", linewidth=4, label=r"Reduced $\chi^2$ cutoff")
    
    ax2.set_xlabel(r"Reduced $\chi^2$")
    ax1.set_ylabel("Accuracy", va="bottom", rotation=270)
    ax2.set_ylabel("Counts")
    
    h2, l2 = ax2.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    ax2.legend(np.append(h2, h1), np.append(l2, l1))

    ax1.yaxis.label.set_color('#228833')
    ax1.spines["right"].set_color('#228833')
    ax1.tick_params(axis="y", colors='#228833')
    ax1.set_ylim((0, 1))

    #ax1.legend(loc="center right")
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    plt.savefig(os.path.join(save_dir, "chisq_vs_accuracy.pdf"), bbox_inches="tight")
    plt.close()


def plot_model_metrics(metrics, plot_name, metrics_dir):
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

    num_epochs = len(train_acc)
    # Plot accuracy
    plt.plot(np.arange(0, num_epochs), train_acc, label="Training")
    plt.plot(np.arange(0, num_epochs), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        os.path.join(metrics_dir, f"accuracy_{plot_name}.pdf"),
        bbox_inches="tight",
    )
    plt.close()

    # Plot loss
    plt.plot(np.arange(0, num_epochs), train_loss, label="Training")
    plt.plot(np.arange(0, num_epochs), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, f"loss_{plot_name}.pdf"), bbox_inches="tight")
    plt.close()


def plot_calibration_curve(probs_csv, save_dir):
    """Plot calibration curve."""
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    colors = CUSTOM_COLORSET
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ratio = 1.0
    ax.set_ylabel("True Fraction")
    plt.locator_params(axis="x", nbins=3)

    legend_lines = []
    for ref_class, ref_label in enumerate(classes_to_labels):
        _, true_classes, probs, _, folds, _ = read_probs_csv(probs_csv)
        y_true = np.where(true_classes == ref_class, 1, 0)
        y_score = probs[:, ref_class]

        t, f, ferr = calc_calibration_curve(y_true, y_score, folds)

        (legend_line,) = ax.step(
            t, f, label=ref_label, c=colors[ref_class], where='post'
        )
        ax.fill_between(t, f-ferr, f+ferr, alpha=0.2, color=colors[ref_class], step='post')
        legend_lines.append(legend_line)

    ax.plot([0,1], [0,1], linestyle='dotted', color='k', linewidth=1)
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()

    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel("Confidence")

    #legend_lines.append(legend_line)
    legend_keys = list(labels_to_classes.keys())
    fig.legend(legend_lines, legend_keys, loc="lower center", ncol=3)
    plt.savefig(os.path.join(save_dir, "calibration_curve.pdf"), bbox_inches="tight")
    plt.close()
    
    
    
def plot_f1_curve(probs_csv, save_dir, ref_class):
    """Plot calibration curve."""
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    colors = CUSTOM_COLORSET
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim([0.0, 1.05])
    ratio = 1.0
    ax.set_ylabel(r"F$_1$")

    thresholds = np.linspace(0, 1, 1000)
    f1_mu = []
    f1_sig = []
    df = pd.read_csv(probs_csv)
    true_classes = df['Label'].to_numpy()
    y_score = df['pSNIa'].to_numpy()
    folds = df['Fold'].to_numpy()
    y_true = np.where(true_classes == ref_class, 1, 0).astype(bool)

    for t in thresholds:
        y_pred = y_score > t
        intersect = (y_pred & y_true).astype(int)
        intersect_other = (~y_pred & ~y_true).astype(int)
        f1s = []
        for f in np.unique(folds):
            f_idx = folds == f
            if sum(y_pred[f_idx].astype(int)) == 0:
                precision = 1.0
            else:
                precision = sum(intersect[f_idx]) / sum(y_pred[f_idx].astype(int))
            recall = sum(intersect[f_idx]) / sum(y_true[f_idx].astype(int))
            Ia_f1 = 2 * precision * recall / (precision + recall)
            
            if sum((~y_pred[f_idx]).astype(int)) == 0:
                prec2 = 1.0
            else:
                prec2 = sum(intersect_other[f_idx]) / sum((~y_pred[f_idx]).astype(int))
            recall2 = sum(intersect_other[f_idx]) / sum((~y_true[f_idx]).astype(int))
            other_f1 = 2 * prec2 * recall2 / (prec2 + recall2)
            f1s.append((Ia_f1 + other_f1)/2)
        f1_mu.append(np.nanmean(f1s))
        f1_sig.append(np.nanstd(f1s))

    f1_mu = np.asarray(f1_mu)
    f1_sig = np.asarray(f1_sig)
    ax.plot(thresholds, f1_mu, c=colors[0])
    ax.fill_between(thresholds, f1_mu-f1_sig, f1_mu+f1_sig, alpha=0.2, color=colors[0])

    # retrieve optimal F1 score
    best_idx = np.argmax(f1_mu)
    best_t = thresholds[best_idx]
    ax.axvline(x=best_t, linestyle='dotted', color=colors[0])
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()

    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel("Confidence Threshold")

    plt.savefig(os.path.join(save_dir, f"f1_curve_{ref_class}.pdf"), bbox_inches="tight")
    plt.close()
    
    print(best_t)
    return best_t