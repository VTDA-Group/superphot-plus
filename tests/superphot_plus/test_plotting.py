import os

import numpy as np
import pytest

from superphot_plus.plotting import plot_posterior_hist, plot_sampling_trace_numpyro, plot_high_confidence_confusion_matrix


def generate_dummy_posterior_sample_dict(batch=False):
    """Create a posterior sample dictionary for r and g bands with random values"""
    param_list = [
        "logA",
        "beta",
        "log_gamma",
        "t0",
        "log_tau_rise",
        "log_tau_fall",
        "log_extra_sigma",
        "A_g",
        "beta_g",
        "gamma_g",
        "t0_g",
        "tau_rise_g",
        "tau_fall_g",
        "extra_sigma_g",
    ]
    return {
        param: np.random.rand(3, 20) if batch else np.random.rand(1, 20).flatten() for param in param_list
    }


def test_plot_posterior_hist(tmp_path):
    """Test that we can plot a posterior samples histogram."""
    samples = generate_dummy_posterior_sample_dict()

    # Check that plot is created for a valid parameter.
    parameter = "log_tau_fall"

    plot_posterior_hist(samples, parameter, tmp_path)

    filepath = os.path.join(tmp_path, f"test_hist_{parameter}.png")
    assert os.path.exists(filepath)

    # Check that plot throws an error if parameter is not valid.
    with pytest.raises(ValueError):
        plot_posterior_hist(samples, "test", tmp_path)
        plot_posterior_hist(samples, None, tmp_path)


def test_plot_posterior_hist_batch(tmp_path):
    """Test that we can plot a histogram for a batch of posterior samples."""
    samples = generate_dummy_posterior_sample_dict(batch=True)

    # Check that plot is created for a valid parameter.
    parameter = "log_tau_fall"

    plot_posterior_hist(samples, parameter, tmp_path)

    filepath = os.path.join(tmp_path, f"test_hist_{parameter}.png")
    assert os.path.exists(filepath)


def test_plot_sampling_trace_numpyro(tmp_path):
    """Test that we can plot the trace of posterior samples."""
    samples = generate_dummy_posterior_sample_dict(batch=True)

    plot_sampling_trace_numpyro(samples, tmp_path)

    filepath = os.path.join(tmp_path, "test_trace.png")
    assert os.path.exists(filepath)


def test_plot_confusion_matrices(class_probs_csv, tmp_path):
    """Test functions that plot confusion matrices.
    """
    test_filename_high_confidence = os.path.join(tmp_path, "test_cm_high_confidence")
    plot_high_confidence_confusion_matrix(class_probs_csv, test_filename_high_confidence)
    assert os.path.exists(test_filename_high_confidence + "_c.pdf")
    assert os.path.exists(test_filename_high_confidence + "_p.pdf")
    
    