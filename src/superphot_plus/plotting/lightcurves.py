"""This module includes scripts to plot fit light curves."""

import os

import matplotlib.pyplot as plt
import numpy as np

from superphot_plus.plotting.utils import lighten_color
from superphot_plus.utils import flux_model, get_numpyro_cube, clip_lightcurve_end
from superphot_plus.plotting.format_params import *

set_global_plot_formatting()

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
