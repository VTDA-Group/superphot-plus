import matplotlib.pyplot as plt
import numpy as np
import os

import arviz as az
import corner

from superphot_plus.format_data_ztf import oversample_using_posteriors, import_labels_only

from superphot_plus.plotting.format_params import *



def corner_plot_all(input_csvs, save_file):
    """Plot combined corner plot of all training set samples, excluding
    the overall scaling A.

    Parameters
    ----------
    input_csvs : list
        List of input CSV file paths containing probability predictions.
    save_file : str
        Path to save the combined corner plot.
    """
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    names, labels = import_labels_only(input_csvs, allowed_types)

    chis = np.ones(len(names))
    features, labels, chis = oversample_using_posteriors(names, labels, chis, 4000)

    figure = corner.corner(
        np.delete(features, [0, 3], axis=1),
        bins=20,
        labels=[
            r"$\beta$",
            r"$\gamma$",
            r"$\tau_r$",
            r"$\tau_f$",
            r"$\sigma_{ex}$",
            r"$A_g$",
            r"$\beta_g$",
            r"$\gamma_g$",
            r"$t_{0,g}$",
            r"$\tau_{r,g}$",
            r"$\tau_{r,g}$",
            r"$\sigma_{ex,g}$",
        ],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 20},
        color="purple",
    )
    # Extract the axes
    axes = np.array(figure.axes)
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)

    figure.savefig(save_file)


def get_numpyro_cube(params, max_flux):
    
    aux_bands = []
    for k in params:
        if k[:4] == "beta" and k != "beta":
            aux_bands.append(k[5:])

    logA, beta, log_gamma = params["logA"], params["beta"], params["log_gamma"]
    t0, log_tau_rise, log_tau_fall, log_extra_sigma = (
        params["t0"],
        params["log_tau_rise"],
        params["log_tau_fall"],
        params["log_extra_sigma"],
    )

    A = max_flux * 10**logA
    gamma = 10**log_gamma
    tau_rise = 10**log_tau_rise
    tau_fall = 10**log_tau_fall
    extra_sigma = 10**log_extra_sigma  # pylint: disable=unused-variable
    
    cube = [A, beta, gamma, t0, tau_rise, tau_fall, extra_sigma]

    for b in aux_bands:
        cube.extend(
            [
                params[f"A_{b}"],
                params[f"beta_{b}"], 
                params[f"gamma_{b}"],
                params[f"t0_{b}"],
                params[f"tau_rise_{b}"],
                params[f"tau_fall_{b}"],
                params[f"extra_sigma_{b}"],
            ]
        )
    return np.array(cube), np.array(aux_bands)



def plot_posterior_hist(posterior_samples, parameter, output_dir=None):
    """
    Plot histogram for a posterior parameter.

    Parameters
    ----------
    posterior_samples :
        Dictionary of posterior samples.
    parameter :
        Posterior parameter for which to plot histogram.
    output_dir :
        The directory where to store the plot.
    """
    if parameter is None or parameter not in posterior_samples:
        raise ValueError("Invalid posterior parameter.")

    output_file = f"test_hist_{parameter}.png"
    if output_dir is not None:
        output_file = os.path.join(output_dir, output_file)

    samples = posterior_samples[parameter]

    if len(samples.shape) > 1:
        samples = samples[:, 0]
    else:
        samples = samples.flatten()

    plt.hist(samples, bins=10)
    plt.savefig(output_file)
    plt.close()


def plot_sampling_trace_numpyro(posterior_samples, output_dir=None):
    """
    Plot trace of all posterior samples.

    Parameters
    ----------
    posterior_samples :
        The lightcurve samples given by MCMC.
    output_dir :
        The directory where to store the plot.
    """

    output_file = "test_trace.png"
    if output_dir is not None:
        output_file = os.path.join(output_dir, output_file)

    post_reformatted = {}
    for p in posterior_samples:
        post_reformatted[p] = np.array(
            [
                posterior_samples[p],
            ]
        )

    az.plot_trace(post_reformatted, compact=True)
    plt.savefig(output_file)
    plt.close()


