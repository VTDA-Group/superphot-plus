import os

import numpy as np
import pytest

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.fit_numpyro import main_loop_directory
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.plotting import (
    plot_posterior_hist,
    plot_sampling_lc_fit_numpyro,
    plot_sampling_trace_numpyro,
)


def generate_dummy_posterior_sample_dict(batch=False):
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


def test_plot_posterior_hist(test_data_dir):
    """Test that we can plot a posterior samples histogram."""
    samples = generate_dummy_posterior_sample_dict()

    # Check that plot is created for a valid parameter.
    parameter = "log_tau_fall"

    plot_posterior_hist(samples, parameter, test_data_dir)

    filepath = os.path.join(test_data_dir, f"test_hist_{parameter}.png")
    assert os.path.exists(filepath)
    os.remove(filepath)

    # Check that plot throws an error if parameter is not valid.
    with pytest.raises(ValueError):
        plot_posterior_hist(samples, "test", test_data_dir)
        plot_posterior_hist(samples, None, test_data_dir)


def test_plot_posterior_hist_batch(test_data_dir):
    """Test that we can plot a histogram for a batch of posterior samples."""
    samples = generate_dummy_posterior_sample_dict(batch=True)

    # Check that plot is created for a valid parameter.
    parameter = "log_tau_fall"

    plot_posterior_hist(samples, parameter, test_data_dir)

    filepath = os.path.join(test_data_dir, f"test_hist_{parameter}.png")
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_plot_sampling_trace_numpyro(test_data_dir):
    """Test that we can plot the trace of posterior samples."""
    samples = generate_dummy_posterior_sample_dict(batch=True)

    plot_sampling_trace_numpyro(samples, test_data_dir)

    filepath = os.path.join(test_data_dir, "test_trace.png")
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_plot_sampling_lc_fit_numpyro(test_data_dir, single_ztf_lightcurve_compressed):
    """Test that we can plot the lightcurve sampling fit."""
    samples = generate_dummy_posterior_sample_dict()

    lc = Lightcurve.from_file(single_ztf_lightcurve_compressed)

    # For non-existent t0 limit
    plot_sampling_lc_fit_numpyro(
        samples,
        [lc.times],
        [lc.fluxes],
        [lc.flux_errors],
        [lc.bands],
        np.random.rand(1, len(lc.times)).flatten(),
        [lc],
        None,
        test_data_dir,
    )

    filepath = os.path.join(test_data_dir, "%s.pdf" % lc.name)
    assert os.path.exists(filepath)
    os.remove(filepath)

    # For existent t0 limit
    t0_lim = np.random.uniform(0, 2)
    plot_sampling_lc_fit_numpyro(
        samples,
        [lc.times],
        [lc.fluxes],
        [lc.flux_errors],
        [lc.bands],
        np.random.rand(1, len(lc.times)).flatten(),
        [lc],
        t0_lim,
        test_data_dir,
    )

    filepath = os.path.join(test_data_dir, "%s_%.02f.pdf" % (lc.name, t0_lim))
    assert os.path.exists(filepath)
    os.remove(filepath)
