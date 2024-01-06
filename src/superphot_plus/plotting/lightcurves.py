"""This module includes scripts to plot fit light curves."""

import os

import matplotlib.pyplot as plt
import numpy as np

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.import_utils import clip_lightcurve_end, import_lc
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.plotting.utils import lighten_color
from superphot_plus.utils import flux_model, get_numpyro_cube
from superphot_plus.plotting.format_params import *
from superphot_plus.posterior_samples import PosteriorSamples

set_global_plot_formatting()

def plot_lc_fit(
    ztf_name,
    ref_band,
    ordered_bands,
    data_dir,
    fit_dir,
    out_dir,
    sampling_method="dynesty",
    file_type="pdf",
    custom_formatting=None,
):
    """Plot an existing light curve fit.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the object.
    data_dir : str
        Directory containing the data files.
    fit_dir : str
        Directory containing the fit files.
    out_dir : str
        Directory for saving the plot.
    sampling_method : str, optional
        Sampling method used for the fit. Default is "dynesty".
    file_type : str, optional
        Type of file to output the resulting image to (e.g. png, pdf). Default is pdf.
    """
    data_fn = os.path.join(data_dir, ztf_name + ".npz")

    lightcurve = Lightcurve.from_file(data_fn)

    eq_wt_samples = PosteriorSamples.from_file(
        name=ztf_name, input_dir=fit_dir, sampling_method=sampling_method
    ).samples

    print(np.mean(eq_wt_samples, axis=0))
    plot_sampling_lc_fit(
        ztf_name,
        out_dir,
        lightcurve.times,
        lightcurve.fluxes,
        lightcurve.flux_errors,
        lightcurve.bands,
        eq_wt_samples,
        ordered_bands,
        ref_band,
        sampling_method,
        file_type,
        custom_formatting,
    )


def plot_sampling_lc_fit(
    ztf_name,
    out_dir,
    tdata,
    fdata,
    ferrdata,
    bdata,
    eq_wt_samples,
    band_order,
    ref_band,
    sampling_method="dynesty",
    file_type="pdf",
    custom_formatting=None,
):
    """
    Plot lightcurve sampling fit using in-memory samples.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the object.
    out_dir : str
        Directory for saving the plot.
    tdata : array-like
        Time of data.
    fdata : array-like
        Flux of data.
    ferrdata : array-like
        Error in flux of data.
    bdata : array-like
        Band of data.
    eq_wt_samples: array-like
        Equally weighted samples from data.
    sampling_method : str, optional
        Sampling method used for the fit. Default is "dynesty".
    """
    os.makedirs(out_dir, exist_ok=True)
    
    band_order = [x for x in band_order] # convert to arrray
    trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

    _, axis = plt.subplots()
    
    max_flux = np.max(fdata[bdata == ref_band] - ferrdata[bdata == ref_band])
    
    for e, band in enumerate(np.unique(bdata)):  # TODO: handle case where band name isnt a valid color
        face_color, edge_color = band_colors(band)
        axis.errorbar(
            tdata[bdata == band],
            fdata[bdata == band],
            yerr=ferrdata[bdata == band],
            c=edge_color,
            fmt="none",
            zorder=500,
        )
        axis.scatter(
            tdata[bdata == band],
            fdata[bdata == band],
            c=face_color,
            edgecolor=edge_color,
            label=band,
            marker="o",
            zorder=1000,
        )

        for sample in eq_wt_samples[:30]:
            axis.plot(
                trange_fine,
                flux_model(sample, trange_fine, [band] * len(trange_fine), max_flux, band_order, ref_band),
                c=face_color,
                lw=1,
                alpha=0.1,
            )

    axis.legend(loc="upper right")
    axis.set_xlabel("Phase (days)")
    axis.set_ylabel("Flux (arbitrary units)")

    axis.set_title(ztf_name + ": " + sampling_method)

    if custom_formatting is not None:
        custom_formatting(axis)

    plt.savefig(
        os.path.join(out_dir, ztf_name + "_" + sampling_method + "." + file_type), bbox_inches="tight"
    )

    plt.close()


def plot_sampling_lc_fit_numpyro(
    posterior_samples,
    tdata,
    fdata,
    ferrdata,
    bdata,
    max_flux,
    lcs,
    ref_band,
    ordered_bands,
    output_folder,
    sampling_method="svi",
):
    """
    Plot lightcurve sampling fit using in-memory samples.

    Parameters
    ----------
    posterior_samples :
        Posterior samples from the MCMC run.
    tdata : array-like
        Time of data.
    fdata : array-like
        Flux of data.
    ferrdata : array-like
        Error in flux of data.
    bdata : array-like
        Band of data.
    max_flux : array-like
        Max flux of data.
    lcs: array-like
        Light curve objects on which sampling was run.
    output_folder : str or FITS_PLOTS_FOLDER, optional
        Directory where to store the light curve sampling fit.
    """
    lcs = np.atleast_1d(lcs)
    tdata = np.atleast_2d(tdata)
    fdata = np.atleast_2d(fdata)
    ferrdata = np.atleast_2d(ferrdata)
    bdata = np.atleast_2d(bdata)
    max_flux = np.atleast_1d(max_flux)

    for i, tdata_i in enumerate(tdata):
        ignore_idx = ferrdata[i] == 1e10
        tdata_i = tdata_i[~ignore_idx]
        fdata_i = fdata[i][~ignore_idx]
        ferrdata_i = ferrdata[i][~ignore_idx]
        bdata_i = bdata[i][~ignore_idx]

        models = np.array(
            [
                {
                    k: posterior_samples[k][j, i]
                    if len(posterior_samples[k].shape) > 1
                    else posterior_samples[k][j]
                    for k in posterior_samples.keys()
                }
                for j in range(len(posterior_samples["log_tau_fall"]))
            ]
        )

        cubes = np.array([get_numpyro_cube(
            single_model,
            max_flux[i],
            ref_band,
            ordered_bands
        )[0] for single_model in models])

        plot_sampling_lc_fit(
            lcs[i],
            output_folder,
            tdata_i,
            fdata_i,
            ferrdata_i,
            bdata_i,
            cubes,
            ordered_bands,
            ref_band,
            sampling_method=sampling_method,
        )


def plot_lightcurve_clipping(ztf_name, data_folder, save_dir):
    """Plot the lightcurve WITH clipped points and lines demonstrating
    how those points are clipped.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the plotted object.
    data_folder: str
        The path to the folder holding the CSV data.
    save_dir: str
        Directory path where to store the plot figure.
    """
    data_fn = os.path.join(data_folder, f"{ztf_name}.csv")
    times, fluxes, flux_errs, bands, _, _ = import_lc(
        data_fn, clip_lightcurve=False
    )  # pylint: disable=unused-variable
    t_clip, f_clip, ferr_clip, b_clip = clip_lightcurve_end(times, fluxes, flux_errs, bands)

    idx_clip = ~np.isin(times, t_clip)
    t_clip = times[idx_clip]
    f_clip = fluxes[idx_clip]
    ferr_clip = flux_errs[idx_clip]
    b_clip = bands[idx_clip]

    t_notclip = times[~idx_clip]
    f_notclip = fluxes[~idx_clip]
    ferr_notclip = flux_errs[~idx_clip]
    b_notclip = bands[~idx_clip]

    for b_name in np.unique(bands):
        all_b_idx = bands == b_name
        clip_b_idx = b_clip == b_name
        notclip_b_idx = b_notclip == b_name
        t_b = times[all_b_idx]
        f_b = fluxes[all_b_idx]
        t_clip_b = t_clip[clip_b_idx]
        f_clip_b = f_clip[clip_b_idx]
        t_notclip_b = t_notclip[notclip_b_idx]
        f_notclip_b = f_notclip[notclip_b_idx]
        ferr_notclip_b = ferr_notclip[notclip_b_idx]

        face_color, edge_color= band_colors(b_name)
        
        plt.errorbar(
            t_notclip_b,
            f_notclip_b,
            yerr=ferr_notclip_b,
            fmt="none",
            c=edge_color,
        )

        plt.scatter(
            t_notclip_b,
            f_notclip_b,
            color=face_color,
            edgecolor=edge_color,
            marker='o',
            zorder=1000,
        )

        # overlay clipped points
        plt.errorbar(
            t_clip_b,
            f_clip_b,
            yerr=ferr_clip[clip_b_idx],
            fmt="^",
            c=lighten_color(face_color, 1.2),
            zorder=500,
        )

        # plot lines from last to max flux point
        t_range_b = np.linspace(t_b[np.argmax(f_b)], np.max(t_b), num=10)
        m_b = (f_b[np.argmax(t_b)] - np.max(f_b)) / (np.max(t_b) - t_b[np.argmax(f_b)])
        y_b = f_b[np.argmax(t_b)] + m_b * (t_range_b - np.max(t_b))

        plt.plot(
            t_range_b,
            y_b,
            c=edge_color,
            label=f"Max {b_name}-band slope",
            linestyle="dashed",
            linewidth=2,
        )

        if len(t_clip_b) == 0:
            t_range_b = []
            y_b = []
        else:
            # plot slope of clipped portion
            t_range_b = np.linspace(t_clip_b[np.argmax(f_clip_b)], np.max(t_clip_b), num=10)
            m_b = (f_clip_b[np.argmax(t_clip_b)] - np.max(f_clip_b)) / (
                np.max(t_clip_b) - t_clip_b[np.argmax(f_clip_b)]
            )
            y_b = f_clip_b[np.argmax(t_clip_b)] + m_b * (t_range_b - np.max(t_clip_b))

        plt.plot(
            t_range_b,
            y_b,
            c=lighten_color(face_color, 1.2),
            linestyle="dotted",
            label=f"Clipped {b_name}-band slope",
            linewidth=2,
        )

    plt.title(ztf_name)
    plt.xlabel("MJD")
    plt.ylabel("Flux (arbitrary scaling)")
    plt.legend()

    plt.savefig(os.path.join(save_dir, f"lc_clip_demo_{ztf_name}.pdf"), bbox_inches="tight")
    plt.close()
