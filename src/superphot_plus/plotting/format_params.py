"""This module introduces uniform plot formatting and parameter label generation."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from superphot_plus.constants import BIGGER_SIZE, MEDIUM_SIZE, SMALL_SIZE


CUSTOM_COLORSET = [
    '#4477AA',
    '#EE6677',
    '#228833',
    '#CCBB44',
    '#66CCEE',
    '#AA3377',
    '#BBBBBB',
]

def set_global_plot_formatting():
    """Set formatting that affects all subsequent plots."""
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    
    # set default color scheme (Paul Tol's Bright):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
        color=CUSTOM_COLORSET
    )
    
    custom_cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_cmap1",
        ["#FFFFFF", '#EE6677']
    )
    custom_cmap2 = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_cmap2",
        ["#FFFFFF", '#4477AA']
    )
    custom_categorical = mpl.colors.ListedColormap(
        CUSTOM_COLORSET,
        name="custom_categorical"
    )
    
    mpl.colormaps.register(cmap=custom_cmap1, force=True)
    mpl.colormaps.register(cmap=custom_cmap2, force=True)
    mpl.colormaps.register(cmap=custom_categorical, force=True)

    
def band_colors(c):
    """Return face and edge colors for light curve plotting."""
    face_dict = {
        "r": '#EE6677',
        "g": '#4477AA',
    }
    edge_dict = {
        'r': '#BB5566',
        'g': '#004488'
    }
    return face_dict[c], edge_dict[c]

def param_labels(aux_bands=None, ref_band=None, log=True):
    """Return properly formatted parameter labels
    for plotting.

    Parameters
    ----------
    aux_bands : array-like, optional
        The auxiliary bands for naming, in order. Defaults to None,
        in which case we assume only the base band.

    Returns
    ----------
    plot_labels : list
        All properly formatted plotting labels.
    """
    if ref_band is not None:
        if log:
            plot_labels = [
                rf"$\log_{{10}}A_{{{ref_band}}}$",
                rf"$\beta_{{{ref_band}}}$",
                rf"$\log_{{10}}\gamma_{{{ref_band}}}$",
                rf"$t_{{0, {ref_band}}}$",
                rf"$\log_{{10}}\tau_\mathrm{{rise, {ref_band}}}$",
                rf"$\log_{{10}}\tau_\mathrm{{fall, {ref_band}}}$",
                rf"$\log_{{10}}\sigma_\mathrm{{extra, {ref_band}}}$",
            ]
        else:
            plot_labels = [
                rf"$A_{{{ref_band}}}$",
                rf"$\beta_{{{ref_band}}}$",
                rf"$\gamma_{{{ref_band}}}$",
                rf"$t_{{0, {ref_band}}}$",
                rf"$\tau_\mathrm{{rise, {ref_band}}}$",
                rf"$\tau_\mathrm{{fall, {ref_band}}}$",
                rf"$\sigma_\mathrm{{extra, {ref_band}}}$",
            ]
    else:
        if log:
            plot_labels = [
                r"$\log_{10}A$",
                r"$\beta$",
                r"$\log_{10}\gamma$",
                r"$t_0$",
                r"$\log_{10}\tau_\mathrm{rise}$",
                r"$\log_{10}\tau_\mathrm{fall}$",
                r"$\log_{10}\sigma_\mathrm{extra}$",
            ]
        else:
            plot_labels = [
                r"$A$",
                r"$\beta$",
                r"$\gamma$",
                r"$t_0$",
                r"$\tau_\mathrm{rise}$",
                r"$\tau_\mathrm{fall}$",
                r"$\sigma_\mathrm{extra}$",
            ]
    save_labels = [
        "logA",
        "beta",
        "loggamma",
        "t0",
        "logtaurise",
        "logtaufall",
        "sigmaextra",
    ]

    if aux_bands is None:
        aux_bands = []

    for aux_b in aux_bands:
        plot_labels.extend(
            [
                rf"$A_{{{aux_b}}}$",
                rf"$\beta_{{{aux_b}}}$",
                rf"$\gamma_{{{aux_b}}}$",
                rf"$t_\mathrm{{0, {aux_b}}}$",
                rf"$\tau_\mathrm{{rise, {aux_b}}}$",
                rf"$\tau_\mathrm{{fall, {aux_b}}}$",
                rf"$\sigma_\mathrm{{extra, {aux_b}}}$",
            ]
        )
        save_labels.extend(
            [
                f"A_{aux_b}",
                f"beta_{aux_b}",
                f"gamma_{aux_b}",
                f"t0_{aux_b}",
                f"taurise_{aux_b}",
                f"taufall_{aux_b}",
                f"extrasigma_{aux_b}",
            ]
        )

    plot_labels.append(r"$\chi^2$")
    save_labels.append("chisquared")

    return np.array(plot_labels), np.array(save_labels)
