import os

import numpy as np
import pytest

from superphot_plus.plotting.lightcurves import (
    plot_lc_fit,
    plot_lightcurve_clipping,
    plot_sampling_lc_fit,
    plot_sampling_lc_fit_numpyro,
)


def test_plot_lc_fit(single_ztf_sn_id, ztf_priors, test_data_dir, tmp_path):
    """Tests plot_lc_fit() and by extension, plot_sampling_lc_fit()."""
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


def test_plot_sampling_lc_fit_numpyro(
    single_ztf_sn_id, ztf_priors, single_ztf_lightcurve_object, tmp_path, dummy_posterior_sample_dict
):
    """Test plot_sampling_lc_fit_numpyro()."""
    posterior_samples = dummy_posterior_sample_dict
    lc = single_ztf_lightcurve_object
    ref_band = ztf_priors.reference_band
    ordered_bands = ztf_priors.ordered_bands
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
        ordered_bands,
        tmp_path,
        "svi",
    )

    out_fn = os.path.join(tmp_path, single_ztf_sn_id + "_svi.pdf")
    assert os.path.exists(out_fn)


def test_plot_lightcurve_clipping(single_ztf_sn_id, test_data_dir, tmp_path):
    """Test plot_lightcurve_clipping()."""

    # tests case with no points clipped
    plot_lightcurve_clipping(single_ztf_sn_id, test_data_dir, tmp_path)
    out_fn = os.path.join(tmp_path, f"lc_clip_demo_{single_ztf_sn_id}.pdf")
    assert os.path.exists(out_fn)
