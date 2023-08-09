import os, glob

import numpy as np
import pytest

from superphot_plus.plotting.sampling_results import (
    plot_posterior_hist_numpyro_dict,
    plot_sampling_trace_numpyro,
    plot_corner_plot_all,
    compare_oversampling,
    plot_combined_posterior_space,
    plot_param_distributions,
    plot_oversampling_1d,
)


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


def test_plot_posterior_hist_numpyro_dict(tmp_path):
    """Test that we can plot a posterior samples histogram."""
    samples = generate_dummy_posterior_sample_dict()

    # Check that plot is created for a valid parameter.
    parameter = "log_tau_fall"

    plot_posterior_hist_numpyro_dict(samples, parameter, tmp_path)

    filepath = os.path.join(tmp_path, f"test_hist_{parameter}.pdf")
    assert os.path.exists(filepath)

    # Check that plot throws an error if parameter is not valid.
    with pytest.raises(ValueError):
        plot_posterior_hist_numpyro_dict(samples, "test", tmp_path)
        plot_posterior_hist_numpyro_dict(samples, None, tmp_path)


def test_plot_posterior_hist_numpyro_batch(tmp_path):
    """Test that we can plot a histogram for a batch of posterior samples."""
    samples = generate_dummy_posterior_sample_dict(batch=True)

    # Check that plot is created for a valid parameter.
    parameter = "log_tau_fall"

    plot_posterior_hist_numpyro_dict(samples, parameter, tmp_path)

    filepath = os.path.join(tmp_path, f"test_hist_{parameter}.pdf")
    assert os.path.exists(filepath)

    
def test_plot_sampling_trace_numpyro(tmp_path):
    """Test that we can plot the trace of posterior samples."""
    samples = generate_dummy_posterior_sample_dict(batch=True)

    plot_sampling_trace_numpyro(samples, tmp_path)

    filepath = os.path.join(tmp_path, "test_trace.pdf")
    assert os.path.exists(filepath)
    
    
def test_corner_plot_all(single_ztf_sn_id, test_data_dir, ztf_priors, tmp_path):
    """Test that we can generate corner plots for combined LCs.
    """
    aux_bands = ztf_priors.aux_bands
    plot_corner_plot_all([single_ztf_sn_id,], ["SN Ia",], test_data_dir, tmp_path, aux_bands)
    
    filepath = os.path.join(tmp_path, "corner_all.pdf")
    assert os.path.exists(filepath)
    
    
def test_compare_oversampling(test_data_dir, tmp_path):
    """Test plots that compare SMOTE and multifit oversampling.
    """
    """
    names = ["ZTF22aarqrxf", "ZTF22abcesfo", "ZTF22abvdwik"] # TODO: de-hardcode this
    compare_oversampling(names, len(names) * ["SN Ia",], test_data_dir, ["SN Ia",], "dynesty")
    
    filepath = os.path.join(tmp_path, "oversample_compare", "logA_vs_beta.pdf")
    os.path.exists(filepath)
    
    fp2 = os.path.join(tmp_path, "oversample_compare", "A_g_vs_beta_g.pdf")
    os.path.exists(fp2)
    """
    # RE-ADD WHEN OVERSAMPLE_MINORITY_CLASSES REIMPLEMENTED
    
    
def test_combined_parameter_space(single_ztf_sn_id, test_data_dir, tmp_path):
    """Test plotting 2d parameter spaces of all LCs.
    """
    plot_combined_posterior_space([single_ztf_sn_id,], ["SN Ia",], test_data_dir, tmp_path)
    
    filepath = os.path.join(tmp_path, "combined_2d_posteriors", "logA_vs_beta.pdf")
    assert os.path.exists(filepath)
    
    
def test_plot_param_distributions(single_ztf_sn_id, test_data_dir, tmp_path):
    """Test plot_param_distributions() with Gaussian overlays.
    """
    plot_param_distributions([single_ztf_sn_id,], ["SN Ia",], test_data_dir, tmp_path)
    
    filepath = os.path.join(tmp_path, "posterior_hists", "beta.pdf")
    assert os.path.exists(filepath)
    
    
def test_plot_oversampling_1d(single_ztf_sn_id, test_data_dir, tmp_path):
    """Test plot_oversampling_1d(), which essentially does the same as
    plot_param_distributions but all in one paper-quality plot.
    """
    plot_oversampling_1d([single_ztf_sn_id,], ["SN Ia",], test_data_dir, tmp_path)
    
    filepath = os.path.join(tmp_path, "all_1d_hists.pdf")
    assert os.path.exists(filepath)