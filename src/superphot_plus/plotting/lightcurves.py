import matplotlib.pyplot as plt
import numpy as np
import os

from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.file_paths import FIT_PLOTS_FOLDER
from superphot_plus.import_utils import clip_lightcurve_end, import_lc
from superphot_plus.utils import flux_model

from superphot_plus.plotting.format_params import *


def plot_lc_fit(
    ztf_name,
    ref_band,
    ordered_bands,
    data_dir,
    fit_dir,
    out_dir,
    sampling_method="dynesty"
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
    """
    data_fn = os.path.join(data_dir, ztf_name + ".npz")

    lightcurve = Lightcurve.from_file(data_fn)

    eq_wt_samples = get_posterior_samples(ztf_name, fit_dir, sampling_method)

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
        sampling_method
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

    trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)
    
    for b in np.unique(bdata): # TODO: handle case where band name isnt a valid color
        plt.errorbar(
            tdata[bdata == b],
            fdata[bdata == b],
            yerr=ferrdata[bdata == b],
            c=b,
            label=b,
            fmt="o",
        )

        for sample in eq_wt_samples[:30]:
            plt.plot(
                trange_fine,
                flux_model(sample, trange_fine, [b] * len(trange_fine), band_order, ref_band),
                c=b,
                lw=1,
                alpha=0.1,
            )
            

    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title(ztf_name + ": " + sampling_method)

    plt.savefig(os.path.join(out_dir, ztf_name + "_" + sampling_method + ".png"))

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
    sampling_method="svi",
    t0_lim=None,
    output_folder=FIT_PLOTS_FOLDER,
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
    t0_lim:  float or None, optional
        Upper time limit for the data.
    output_folder : str or FITS_PLOTS_FOLDER, optional
        Directory where to store the light curve sampling fit.
    """

    for i in range(len(tdata)):
        ignore_idx = ferrdata[i] == 1e10  # pylint: ignore-superfluous parens
        tdata = tdata[i][~ignore_idx]
        fdata = fdata[i][~ignore_idx]
        ferrdata = ferrdata[i][~ignore_idx]
        bdata = bdata[i][~ignore_idx]

        model_i = np.array(
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
        
        cubes = np.array([get_numpyro_cube(single_model, max_flux)[0] for single_model in model_i])
        aux_bands = get_numpyro_cube(model_i[0], max_flux)[1]
        
        plot_sampling_lc_fit(
            lcs[i],
            output_folder,
            tdata,
            fdata,
            ferrdata,
            bdata,
            cubes,
            aux_bands,
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
    data_fn = f"{data_folder}/{ztf_name}.csv"
    t, f, ferr, b, ra, dec = import_lc(data_fn)  # pylint: disable=unused-variable
    t_clip, f_clip, ferr_clip, b_clip = clip_lightcurve_end(t, f, ferr, b)

    idx_clip = ~np.isin(t, t_clip)
    t_clip = t[idx_clip]
    f_clip = f[idx_clip]
    ferr_clip = ferr[idx_clip]
    b_clip = b[idx_clip]

    plt.errorbar(t[b == "r"], f[b == "r"], yerr=ferr[b == "r"], fmt="o", c="r")
    plt.errorbar(t[b == "g"], f[b == "g"], yerr=ferr[b == "g"], fmt="o", c="g")

    # overlay clipped points
    plt.errorbar(
        t_clip[b_clip == "r"],
        f_clip[b_clip == "r"],
        yerr=ferr_clip[b_clip == "r"],
        fmt="o",
        c="orange",
    )
    plt.errorbar(
        t_clip[b_clip == "g"],
        f_clip[b_clip == "g"],
        yerr=ferr_clip[b_clip == "g"],
        fmt="o",
        c="blue",
    )

    # plot lines from last to max flux point
    t_r = t[b == "r"]
    f_r = f[b == "r"]
    t_g = t[b == "g"]
    f_g = f[b == "g"]

    t_range_r = np.linspace(t_r[np.argmax(f_r)], np.max(t_r), num=10)
    m_r = (f_r[np.argmax(t_r)] - np.max(f_r)) / (np.max(t_r) - t_r[np.argmax(f_r)])
    y_r = f_r[np.argmax(t_r)] + m_r * (t_range_r - np.max(t_r))

    t_range_g = np.linspace(t_g[np.argmax(f_g)], np.max(t_g), num=10)
    m_g = (f_g[np.argmax(t_g)] - np.max(f_g)) / (np.max(t_g) - t_g[np.argmax(f_g)])
    y_g = f_g[np.argmax(t_g)] + m_g * (t_range_g - np.max(t_g))

    plt.plot(t_range_r, y_r, c="r", label="Max r-band slope", linewidth=1)
    plt.plot(t_range_g, y_g, c="g", label="Max g-band slope", linewidth=1)

    # plot slope of clipped portion
    t_clip_r = t_clip[b_clip == "r"]
    f_clip_r = f_clip[b_clip == "r"]
    t_clip_g = t_clip[b_clip == "g"]
    f_clip_g = f_clip[b_clip == "g"]

    t_range_r = np.linspace(t_clip_r[np.argmax(f_clip_r)], np.max(t_clip_r), num=10)
    m_r = (f_clip_r[np.argmax(t_clip_r)] - np.max(f_clip_r)) / (
        np.max(t_clip_r) - t_clip_r[np.argmax(f_clip_r)]
    )
    y_r = f_clip_r[np.argmax(t_clip_r)] + m_r * (t_range_r - np.max(t_clip_r))

    t_range_g = np.linspace(t_clip_g[np.argmax(f_clip_g)], np.max(t_clip_g), num=10)
    m_g = (f_clip_g[np.argmax(t_clip_g)] - np.max(f_clip_g)) / (
        np.max(t_clip_g) - t_clip_g[np.argmax(f_clip_g)]
    )
    y_g = f_clip_g[np.argmax(t_clip_g)] + m_g * (t_range_g - np.max(t_clip_g))

    plt.plot(t_range_r, y_r, c="orange", label="Clipped r-band slope", linewidth=1)
    plt.plot(t_range_g, y_g, c="blue", label="Clipped g-band slope", linewidth=1)

    plt.title(ztf_name, fontsize=20)
    plt.xlabel("MJD", fontsize=15)
    plt.ylabel("Flux (arbitrary scaling)", fontsize=15)
    plt.legend()

    plt.savefig(os.path.join(save_dir, "lc_clip_demo.pdf"))
    plt.close()