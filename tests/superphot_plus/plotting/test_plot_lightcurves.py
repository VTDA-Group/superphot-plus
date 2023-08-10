import os

import numpy as np
import pytest

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.plotting.lightcurves import (
    plot_lc_fit,
    plot_sampling_lc_fit,
    plot_sampling_lc_fit_numpyro,
    plot_lightcurve_clipping,
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


def test_plot_lc_fit(single_ztf_sn_id, ztf_priors, test_data_dir, tmp_path):
    """Tests plot_lc_fit() and my extension, plot_sampling_lc_fit()."""
    ref_band = ztf_priors.reference_band
    ordered_bands = ztf_priors.ordered_bands
    data_dir = test_data_dir
    fit_dir = test_data_dir
    out_dir = tmp_path

    plot_lc_fit(
        single_ztf_sn_id,
        ref_band,
        ordered_bands,
        data_dir,
        fit_dir,
        out_dir,
        "dynesty",
    )
    out_fn = os.path.join(out_dir, single_ztf_sn_id + "_dynesty.pdf")
    assert os.path.exists(out_fn)


def test_plot_sampling_lc_fit_numpyro(single_ztf_sn_id, ztf_priors, single_ztf_lightcurve_object, tmp_path):
    """Test plot_sampling_lc_fit_numpyro()."""
    posterior_samples = generate_dummy_posterior_sample_dict()
    lc = single_ztf_lightcurve_object
    ref_band = ztf_priors.reference_band
    max_flux = lc.find_max_flux(band=ref_band)[0]
    lcs = [
        single_ztf_sn_id,
    ]

    plot_sampling_lc_fit_numpyro(
        posterior_samples,
        lc.times,
        lc.fluxes,
        lc.flux_errors,
        lc.bands,
        max_flux,
        lcs,
        ref_band,
        "svi",
        output_folder=tmp_path,
    )

    out_fn = os.path.join(tmp_path, single_ztf_sn_id + "_svi.pdf")
    assert os.path.exists(out_fn)


def test_plot_lightcurve_clipping(single_ztf_sn_id, test_data_dir, tmp_path):
    """Test plot_lightcurve_clipping()."""
    
    # tests case with no points clipped
    plot_lightcurve_clipping(single_ztf_sn_id, test_data_dir, tmp_path)
    out_fn = os.path.join(tmp_path, f"lc_clip_demo_{single_ztf_sn_id}.pdf")
    assert os.path.exists(out_fn)