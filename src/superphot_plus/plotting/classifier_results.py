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
from snapi import Formatter

from superphot_plus.plotting.utils import histedges_equalN

def plot_redshifts_abs_mags(fig, z_ax, mag_ax, transient_group, classifier_result, formatter=Formatter()):
    """
    Plot redshift and absolute magnitude distributions used in
    certain classifier.
    
    UPDATED
    """
    if 'abs_mag' not in transient_group.metadata:
        raise ValueError("abs_mag is not an attribute associated with transient group. Please first add.")
        
    legend_lines = []
    meta_df = transient_group.filter(classifier_result.df.index, inplace=True)

    _, bin_edges = np.histogram(-abs_mags, bins=40, density=True, range=(15, 25))
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    for l in np.unique(meta_df['spec_class']):
        features_1_t = -meta_df['abs_mag'][meta_df['spec_class'] == l]
        feature_hist, bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        (legend_line,) = mag_ax.step(-bin_centers, cumsum, where="mid", label=l)
        legend_lines.append(legend_line)

    mag_ax.set_xlabel("Absolute Magnitude")
    mag_ax.invert_xaxis()

    _, bin_edges = np.histogram(redshifts, bins=40, density=True, range=(-0.1, 0.6))
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    for l in np.unique(meta_df['spec_class']):
        features_1_t = meta_df['redshift'][meta_df['spec_class'] == l]
        feature_hist, bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        z_ax.step(bin_centers, cumsum, where="mid", label=l)

    z_ax.set_xlabel("Redshift")
    z_ax.set_ylabel("Cumulative Fraction")

    for ax in axes:
        formatter.set_aspect_ratio(ax, 1.0)

    fig.legend(legend_lines, [*allowed_types, "Combined"], loc="lower center", ncol=3)
    return fig, ax


def plot_snr_npoints_vs_accuracy(fig, ax, ax2, transient_group, classifier_result):
    """
    Generate plots of number of SNR > 5 points versus
    accuracy, and top 10% SNR versus accuracy.
    
    UPDATED
    """
    legend_lines = []
    merged_df = pd.merge(
        transient_group.metadata,
        classifier_result.df,
        how='inner',
        left_index=True,
        right_index=True
    )
    
    for attr in ['snr_90', 'n_obs_snr5']:
        if attr not in merged_df:
            raise ValueError(f"{attr} is not an attribute associated with transient group. Please first add.")

    filt_df['correct'] = (filt_df['pred_class'] == filt_df['true_class']).astype(int)
    
    for t in np.unique(filt_df['spec_class']):
        filt_df = merged_df[merged_df['spec_class' == t]]

        nbins = min(len(filt_df)-1, 8)
            
        snr_vs_accuracy, snr_bin_edges, _ = binned_statistic(
            filt_df['snr_90'], filt_df['correct'], "mean",
            bins=histedges_equalN(filt_df['snr_90'], nbins)
        )
        snr_vs_accuracy[np.isnan(snr_vs_accuracy)] = 1.0

        ax.step(
            snr_bin_edges,
            np.append(snr_vs_accuracy, snr_vs_accuracy[-1]),
            where="post",
        )
        
        n_vs_accuracy, n_bin_edges, _ = binned_statistic(
            filt_df['n_obs_snr5'], filt_df['correct'], "mean",
            bins=histedges_equalN(filt_df['n_obs_snr5'], nbins)
        )
        n_vs_accuracy[np.isnan(n_vs_accuracy)] = 1.0

        ax2.step(
            n_bin_edges,
            np.append(n_vs_accuracy, n_vs_accuracy[-1]),
            label=t,
            where="post",
        )

    for axis in (ax, ax2):
        axis.set_xscale('log')
        axis.set_ylabel("Class Completeness")
        
    ax.set_xlim((8, 30))
    ax.set_xlabel("90th Percentile SNR")
    ax2.set_xlim((10, 200))
    ax2.set_xlabel(r"Number of $\geq 3\sigma$ Datapoints")
    
    # TODO: move to formatter
    #fig.tight_layout()
    #fig.subplots_adjust(hspace=0.3, bottom=0.2, top=0.95)
    
    return fig, (ax, ax2)


def plot_snr_hist(ax, transient_group):
    """
    Replicates SNR plots needed for publication.
    
    UPDATED
    """
    meta_df = transient_group.metadata
    for attr in ['n_obs_snr3', 'n_obs_snr5', 'n_obs_snr10']:
        if attr not in meta_df:
            raise ValueError(f"{attr} is not an attribute associated with transient group. Please first add.")
            
    bins = np.arange(0, 603, 3)
    
    for snr in [3,5,10]:
        ax.hist(meta_df[f'n_obs_snr{snr}'], histtype="step", label=rf"$SNR \geq {{snr}}$", bins=bins)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of Datapoints at Given SNR")
    ax.set_ylabel("Number of Light Curves")
    
    return ax


def compare_mag_distributions(
    cr_spec,
    cr_phot,
    sr_spec,
    sr_phot,
    transient_data_spec,
    transient_data_phot
):
    """
    Generate overlaid magnitude distributions of the classified and unclassified datasets.
    
    UPDATED
    """
    spec_df = pd.merge(
        pd.merge(
            sr_spec.metadata,
            transient_data_spec.metadata,
            how='inner',
            left_index = True,
            right_index = True
        ),
        cr_spec.df, how='outer',
        left_index = True, right_index = True
    )
    phot_df = pd.merge(
        pd.merge(
            sr_phot.metadata,
            transient_data_phot.metadata,
            how='inner',
            left_index = True,
            right_index = True
        ),
        cr_phot.df, how='outer',
        left_index = True, right_index = True
    )
    
    if ('abs_mag' not in spec_df) or ('abs_mag' not in phot_df):
        raise ValueError(f"{attr} is not an attribute associated with transient group. Please first add.")
        
    for (label, df) in [("Spec", spec_df), ("Phot", phot_df)]:
        classified_mask = ~pd.isna(df['pred_class'])
        ax.hist(
            df[classified_mask,'abs_mag'],
            histtype="step",
            bins=np.arange(5.0, 21.0, 0.5),
            label=f"{label}. (included)",
            density=True,
            linewidth=2,
        )
        ax.hist(
            df[~classified_mask,'abs_mag'],
            histtype="step",
            bins=np.arange(5.0, 21.0, 0.5),
            label=f"{label}. (excluded)",
            density=True,
            linewidth=2,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Peak Apparent Magnitude")
    ax.set_ylabel("Fraction of Light Curves")
    return ax


def plot_chisquared_vs_accuracy(
    ax,
    cr_spec,
    cr_phot,
    sr_spec,
    sr_phot,
):
    """
    Plot chi-squared value histograms for both the spectroscopic and photometric
    datasets, and plot spec chi-squared as a function of classification accuracy.
    
    UPDATED
    """
    spec_df = pd.merge(
        sr_spec.metadata,
        cr_spec.df, how='outer',
        left_index = True, right_index = True
    )
    phot_df = pd.merge(
        sr_phot.metadata,
        cr_phot.df, how='outer',
        left_index = True, right_index = True
    )
    spec_df['correct'] = (spec_df['true_class'] == spec_df['pred_class']).astype(int)
   
    bins = np.arange(0, 4.0, 0.1)

    acc_hist, bin_edges, _ = binned_statistic(
        spec_df[~pd.isna(spec_df['pred_class']), 'score'], spec_df[~pd.isna(spec_df['pred_class']), 'correct'], statistic="mean", bins=bins
    )
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    
    ax2 = ax.twinx()
    
    ax.hist(spec_df['score'], bins=bins, alpha=0.5, label="Spectroscopic")
    ax.hist(phot_df['score'], bins=bins, alpha=0.5, label="Photometric")
    ax.set_yscale("log")
    
    acc_cut = acc_hist[bins[:-1] < 1.2]
    acc_cut = np.append(acc_cut, acc_cut[-1])
    ax2.step(
        bins[bins < 1.215],
        acc_cut,
        where="post", color='#228833', linewidth=3, label="Accuracy"
    )
    ax2.axvline(x=1.2, color="black", linestyle="--", linewidth=4, label=r"Reduced $\chi^2$ cutoff")
    
    ax.set_xlabel(r"Reduced $\chi^2$")
    ax2.set_ylabel("Accuracy", va="bottom", rotation=270)
    ax.set_ylabel("Counts")
    
    h2, l2 = ax.get_legend_handles_labels()
    h1, l1 = ax2.get_legend_handles_labels()
    ax.legend(np.append(h2, h1), np.append(l2, l1))

    ax2.yaxis.label.set_color('#228833')
    ax2.spines["right"].set_color('#228833')
    ax2.tick_params(axis="y", colors='#228833')
    ax2.set_ylim((0, 1))

    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    return ax, ax2



