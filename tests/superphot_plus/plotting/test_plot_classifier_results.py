import os

import numpy as np
import pytest

from superphot_plus.plotting.classifier_results import (
    compare_mag_distributions,
    generate_roc_curve,
    plot_chisquared_vs_accuracy,
    plot_redshifts_abs_mags,
    plot_snr_hist,
    plot_snr_npoints_vs_accuracy,
    plot_class_fractions,
)


def test_plot_class_fractions(test_class_frac_csv, tmp_path):
    """Test plotting of class fractions."""
    plot_class_fractions(test_class_frac_csv, tmp_path, "test_cf.png")
    filepath = os.path.join(tmp_path, "test_cf.png")
    assert os.path.exists(filepath)


def test_generate_roc_curve(class_probs_csv, tmp_path):
    """Test ROC curve generation."""
    generate_roc_curve(class_probs_csv, tmp_path)
    filepath = os.path.join(tmp_path, "roc_all.pdf")
    assert os.path.exists(filepath)


def test_plot_redshifts_abs_mags(class_probs_snr_csv, training_csv, test_data_dir, tmp_path):
    """Test redshift and abs magnitude plots are being generated."""
    # TODO: find away around not having all fits saved
    """
    plot_redshifts_abs_mags(class_probs_snr_csv, training_csv, test_data_dir, tmp_path)
    filepath = os.path.join(tmp_path, "abs_mag_hist.pdf")
    assert os.path.exists(filepath)
    """


def test_plot_snr_npoints_vs_accuracy(class_probs_snr_csv, tmp_path):
    """Test whether SNR vs Npoints trends are being plotted."""
    plot_snr_npoints_vs_accuracy(class_probs_snr_csv, tmp_path)
    filepath = os.path.join(tmp_path, "n_snr_vs_accuracy.pdf")
    assert os.path.exists(filepath)


def test_plot_snr_hist(class_probs_snr_csv, tmp_path):
    """Test whether SNR histograms are being plotted."""
    plot_snr_hist(class_probs_snr_csv, tmp_path)
    filepath = os.path.join(tmp_path, "snr_hist.pdf")
    assert os.path.exists(filepath)


def test_compare_mag_distributions(class_probs_snr_csv, training_csv, test_data_dir, tmp_path):
    """Test magnitude comparison plots are generated.
    TODO: add example photometric SNR CSV for proper generation.
    """
    compare_mag_distributions(
        class_probs_snr_csv,
        class_probs_snr_csv,
        training_csv,
        training_csv,
        test_data_dir,
        test_data_dir,
        tmp_path
    )
    filepath = os.path.join(tmp_path, "appm_hist_compare.pdf")
    assert os.path.exists(filepath)


def test_plot_chisquared_vs_accuracy(class_probs_csv, tmp_path):
    """Test plot generation showing chisquared histograms overlaid
    over chisquared vs accuracy trend.
    """
    """
    plot_chisquared_vs_accuracy(class_probs_csv, class_probs_csv, tmp_path)
    filepath = os.path.join(tmp_path, "chisq_vs_accuracy.pdf")
    assert os.path.exists(filepath)
    """
    # IN PROGRESS
