import matplotlib.pyplot as plt
import numpy as np
from superphot_plus.constants import BIGGER_SIZE, MEDIUM_SIZE, SMALL_SIZE

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def param_labels(aux_bands=None):
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
    plot_labels = [
        r"$\log_{10}A$",
        r"$\beta$",
        r"$\log_{10}\gamma$",
        r"$t_0$",
        r"$\log_{10}\tau_\mathrm{rise}$",
        r"$\log_{10}\tau_\mathrm{fall}$",
        r"$\log_{10}\sigma_\mathrm{extra}$",
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
        
    for b in aux_bands:
        plot_labels.extend(
            [
                rf"$A_{{{b}}}$",
                rf"$\beta_{{{b}}}$",
                rf"$\gamma_{{{b}}}$",
                rf"$t_\mathrm{{0, {b}}}$",
                rf"$\tau_\mathrm{{rise, {b}}}$",
                rf"$\tau_\mathrm{{fall, {b}}}$",
                rf"$\sigma_\mathrm{{extra, {b}}}$",
            ]
        )
        save_labels.extend(
            [
                f"A_{b}",
                f"beta_{b}",
                f"gamma_{b}",
                f"t0_{b}",
                f"taurise_{b}",
                f"taufall_{b}",
                f"extrasigma_{b}",
            ]
        )

    plot_labels.append(r"$\chi^2$")
    save_labels.append("chisquared")

    return np.array(plot_labels), np.array(save_labels)
