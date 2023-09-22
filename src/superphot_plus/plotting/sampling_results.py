"""This module contains scripts to plot sampling results."""
import os

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

from superphot_plus.format_data_ztf import oversample_smote, oversample_using_posteriors
from superphot_plus.plotting.format_params import param_labels
from superphot_plus.plotting.utils import gaussian
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.surveys.surveys import Survey

OVERSAMPLE_SIZE = 4000


def plot_corner_plot_all(
    names,
    labels,
    fits_dir,
    save_dir,
    aux_bands=Survey.ZTF().priors.aux_bands,
):
    """Plot combined corner plot of all training set samples, excluding
    the overall scaling A.

    Parameters
    ----------
    names : array-like
        List of object names.
    labels : array-like
        Class labels associated the objects in names.
    fits_dir : str
        Where object model fits are stored.
    save_dir : str
        Path to save the combined corner plot.
    aux_bands : array-like, optional
        The auxiliary bands of the fits (for plotting lables). Defaults to ZTF's aux bands.
    """
    # allowed_types = SnClass.all_classes()

    features, labels, _ = oversample_using_posteriors(names, labels, OVERSAMPLE_SIZE, fits_dir)
    plotting_labels, _ = param_labels(aux_bands)
    skip_idxs = [0, 3, len(plotting_labels) - 1]

    # remove A, t0, and chisquared
    plotting_labels = [
        x for i, x in enumerate(plotting_labels) if i not in skip_idxs
    ]

    figure = corner.corner(
        np.delete(features[:, :-1], [0, 3], axis=1),
        bins=20,
        labels=plotting_labels,
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

    figure.savefig(os.path.join(save_dir, "corner_all.pdf"))
    plt.close()


def plot_posterior_hist_numpyro_dict(posterior_samples, parameter, output_dir=None):
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

    output_file = f"test_hist_{parameter}.pdf"
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

    output_file = "test_trace.pdf"
    if output_dir is not None:
        output_file = os.path.join(output_dir, output_file)

    post_reformatted = {}
    for param in posterior_samples:
        post_reformatted[param] = np.array(
            [
                posterior_samples[param],
            ]
        )

    az.plot_trace(post_reformatted, compact=True)
    plt.savefig(output_file)
    plt.close()


def compare_oversampling(
    names,
    labels,
    fits_dir,
    save_dir,
    allowed_types=SnClass.all_classes(),
    aux_bands=Survey.ZTF().priors.aux_bands,
    sampler=None,
    goal_per_class=1000,
):
    """
    Compare plots of various oversampling methods.

    Parameters
    ----------
    input_csv : str
        Where supernova list is stored.
    allowed_types : array-like
        Types to include in plot.
    """
    # names, labels = import_labels_only(input_csv, allowed_types)
    _, classes_to_labels = SnClass.get_type_maps()
    labels = np.array([classes_to_labels[x] for x in labels])

    features_gaussian, labels_gaussian, _ = oversample_using_posteriors(
        names, labels, OVERSAMPLE_SIZE, fits_dir, sampler
    )

    feature_means = []
    labels_ordered = []

    start_idx = 0
    for label in np.unique(labels):
        type_idx = labels == label
        labels_ordered.extend(labels[type_idx])
        names_t = names[type_idx]
        samples_per_fit = max(1, int(np.round(goal_per_class / len(names_t))))

        for enum, name in enumerate(names_t):
            ## FIXME - e and name are unused - why is this a loop?
            feature_means.append(
                np.mean(features_gaussian[start_idx : start_idx + samples_per_fit], axis=0)
            )
            start_idx += samples_per_fit

    feature_means = np.array(feature_means)
    feature_means_filler = np.ones((goal_per_class, len(feature_means[0])))
    labels_filler = 100 * np.ones(goal_per_class)

    feature_means_comb = np.vstack((feature_means, feature_means_filler))
    labels_comb = np.append(labels_ordered, labels_filler)
    features_smote_comb, labels_smote_comb = oversample_smote(feature_means_comb, labels_comb)

    features_smote = features_smote_comb[labels_smote_comb == allowed_types[0]]
    labels_smote = labels_smote_comb[labels_smote_comb == allowed_types[0]]

    params, save_labels = param_labels(aux_bands)

    for i, param_1 in enumerate(params):
        for j in range(i):
            if i in [0, 3, len(params)]:
                continue
            if j in [0, 3, len(params)]:
                continue

            _, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={"hspace": 0})

            smote_ax = axes[0]
            gauss_ax = axes[1]

            if i in [2, 4, 5, 6]:
                smote_ax.set_xscale("log")
                gauss_ax.set_xscale("log")
            if j in [2, 4, 5, 6]:
                smote_ax.set_yscale("log")
                gauss_ax.set_yscale("log")

            param_2 = params[j]

            features_1_smote = features_smote[:, i]
            features_2_smote = features_smote[:, j]

            features_1_gauss = features_gaussian[:, i]
            features_2_gauss = features_gaussian[:, j]
            for allowed_t in allowed_types:
                feature_means_t1 = feature_means[:, i][labels_ordered == allowed_t]
                feature_means_t2 = feature_means[:, j][labels_ordered == allowed_t]
                features_smote_t1 = features_1_smote[labels_smote == allowed_t]
                features_smote_t2 = features_2_smote[labels_smote == allowed_t]
                features_gauss_t1 = features_1_gauss[labels_gaussian == allowed_t]
                features_gauss_t2 = features_2_gauss[labels_gaussian == allowed_t]

                smote_ax.scatter(
                    features_smote_t1, features_smote_t2,
                    label=allowed_t, alpha=0.2, s=1
                )
                smote_ax.scatter(
                    feature_means_t1, feature_means_t2,
                    label=allowed_t, s=3, c="black"
                )

                gauss_ax.scatter(
                    feature_means_t1, feature_means_t2,
                    label=allowed_t, s=3, c="black"
                )
                gauss_ax.scatter(
                    features_gauss_t1, features_gauss_t2,
                    label=allowed_t, alpha=0.2, s=1
                )

            # annotate with labels
            smote_ax.text(
                0.99, 0.99, "SMOTE",
                horizontalalignment='right',
                verticalalignment='top',
                c="black",
                transform=smote_ax.transAxes,
            )
            gauss_ax.text(
                0.99, 0.99, "Oversampling\nMultiple Fits\nPer Lightcurve",
                horizontalalignment='right',
                verticalalignment='top',
                c="black",
                transform=gauss_ax.transAxes,
            )
            smote_ax.set_title("Oversampling using SMOTE vs Multiple Fits")
            gauss_ax.set_xlabel(param_1)
            gauss_ax.set_ylabel(param_2)
            smote_ax.set_ylabel(param_2)
            direct_filename = f"{save_labels[i]}_vs_{save_labels[j]}.pdf"
            plt.savefig(
                os.path.join(save_dir, "oversample_compare", direct_filename),
                bbox_inches="tight",
            )
            plt.close()


def plot_oversampling_1d(
    names, labels, fits_dir, save_dir,
    priors=Survey.ZTF().priors, sampler="dynesty"
):
    """
    Save all 1d oversampled histograms for each parameter, in one plot.
    Overlays prior distributions.

    Parameters
    ----------
    names : array-like
        List of all object names.
    labels : array-like
        List of all labels associated with 'names'.
    fits_dir : str
        Directory where model fits are stored.
    save_dir : str
        Where to save figure.
    priors : MultibandPriors, optional
        Prior object to overlay prior distributions. Defaults to ZTF's priors.
    """
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()
    allowed_types = list(classes_to_labels.keys())

    goal_per_class = OVERSAMPLE_SIZE
    features_gaussian, labels_gaussian, _ = oversample_using_posteriors(
        names, labels, goal_per_class, fits_dir, sampler
    )

    params, _ = param_labels(priors.aux_bands, priors.reference_band)

    fig, axes = plt.subplots(3, 4, figsize=(8, 10))
    axes = axes.ravel()

    prior_means = priors.to_numpy()[:, 2]
    prior_stddevs = priors.to_numpy()[:, 3]

    ax_num = 0
    for i in range(1, len(params) - 1):
        leg_lines = []
        param_1 = params[i]
        features_1_gauss = features_gaussian[:, i]

        if i == 3:
            continue
        if i == 10:
            param_1 = r"$10^4\times (t_\mathrm{0, g} - 1)$"
            features_1_gauss = 10000 * (features_1_gauss - 1)
            prior_means[i] = 10000 * (prior_means[i] - 1)
            prior_stddevs[i] = 10000 * prior_stddevs[i]
        if i == 1:
            param_1 = r"$10^3\times \beta_\mathrm{r}$"
            features_1_gauss = 1000 * features_1_gauss
            prior_means[i] = 1000 * prior_means[i]
            prior_stddevs[i] = 1000 * prior_stddevs[i]

        log_scale = False

        if i in [2, 4, 5, 6]:
            log_scale = True
            axes[ax_num].set_xscale("log")
            feature_hist, bin_edges = np.histogram(np.log10(features_1_gauss), bins=20)
            bin_width = bin_edges[1] - bin_edges[0]
            bin_edges = 10**bin_edges

        else:
            feature_hist, bin_edges = np.histogram(features_1_gauss, bins=20)
            bin_width = bin_edges[1] - bin_edges[0]

        feature_hist[np.abs(feature_hist) > 1e5] = 0
        current_area = np.sum(bin_width * feature_hist)
        feature_hist = (1.0 / current_area) * feature_hist

        feature_hist_all = feature_hist
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        for allowed_class in allowed_types:
            allowed_label = classes_to_labels[allowed_class]
            features_1_t = features_1_gauss[labels_gaussian == allowed_class]

            feature_hist, bin_edges = np.histogram(features_1_t, bins=bin_edges)
            current_area = np.sum(bin_width * feature_hist)
            feature_hist = (1.0 / current_area) * feature_hist
            feature_hist[np.abs(feature_hist) > 1e5] = 0
            (legend_line,) = axes[ax_num].step(
                bin_centers, feature_hist, where="mid", label=allowed_label
            )
            leg_lines.append(legend_line)

        ax = axes[ax_num]
        (legend_line,) = ax.step(
            bin_centers,
            feature_hist_all,
            where="mid", c="k", label="Combined", linewidth=2
        )
        leg_lines.append(legend_line)

        amp = 1.0 / np.sqrt(2 * np.pi) / prior_stddevs[i]

        if log_scale:
            bins_fine = np.linspace(np.log10(bin_centers[0]), np.log10(bin_centers[-1]), num=100)
            prior_dist = gaussian(bins_fine, amp, prior_means[i], prior_stddevs[i])
            bins_fine = 10**bins_fine
        else:
            bins_fine = np.linspace(bin_centers[0], bin_centers[-1], num=100)
            prior_dist = gaussian(bins_fine, amp, prior_means[i], prior_stddevs[i])

        (legend_line,) = ax.plot(
            bins_fine, prior_dist,
            linestyle="dashed", linewidth=2,
            label="Prior", c="magenta"
        )
        leg_lines.append(legend_line)
        ax.set_xlabel(param_1)
        ax.set_yticklabels([])
        ax.set_yticks([])

        ratio = 1.2
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()

        if log_scale:
            ax.set_aspect(abs(np.log10(x_right / x_left) / (y_low - y_high)) * ratio)
        else:
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        ax_num += 1

    legend_keys = [*list(labels_to_classes.keys()), "Combined", "Prior"]
    fig.legend(leg_lines, legend_keys, loc="lower center", ncol=4)
    plt.locator_params(axis="x", nbins=3)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)

    plt.savefig(os.path.join(save_dir, "all_1d_hists.pdf"))  # , bbox_inches="tight")
    plt.close()


def plot_combined_posterior_space(
    names, labels, fits_dir, save_dir,
    aux_bands=Survey.ZTF().priors.aux_bands
):
    """
    Plot 2D scatterplots for each pair
    of fit parameters, to identify clustering
    among different subclasses.

    TODO: modify to plot different points for each type.

    Parameters
    ----------
    fits_fn : str
        File path for fit posteriors.
    save_dir : str
        Where to save figure
    """
    os.makedirs(os.path.join(save_dir, "combined_2d_posteriors"), exist_ok=True)
    # pt_colors = ["r", "c", "k", "m", "g"] # keep for TODO

    features, labels, _ = oversample_using_posteriors(names, labels, OVERSAMPLE_SIZE, fits_dir)

    params, save_labels = param_labels(aux_bands)

    # color_arr = [labels_to_vals[l] for l in labels]
    for j in range(1, len(params) - 1):
        for i in range(j):
            param_1 = params[i]
            param_2 = params[j]
            features_1 = features[:, i]
            features_2 = features[:, j]

            if i in [2, 4, 5, 6]:
                features_1 = np.log10(features_1)
            if j in [2, 4, 5, 6]:
                features_2 = np.log10(features_2)

            plt.scatter(features_1, features_2, s=2, alpha=0.005)
            # for t_idx in range(len(allowed_types)):
            #     t = allowed_types[t_idx]
            #     features_1_t = features_1[labels == t]
            #     features_2_t = features_2[labels == t]
            plt.xlabel(param_1)
            plt.ylabel(param_2)
            # leg = plt.legend()
            # for lh in leg.legendHandles:
            #    lh.set_alpha(1)
            direct_filename = f"{save_labels[i]}_vs_{save_labels[j]}.pdf"
            plt.savefig(
                os.path.join(save_dir, "combined_2d_posteriors", direct_filename),
                bbox_inches="tight",
            )
            plt.close()


def plot_param_distributions(
    names, labels, fit_folder, save_dir, overlay_gaussians=True,
    aux_bands=Survey.ZTF().priors.aux_bands
):
    """
    Plot the parameter distributions to get better priors for fitting.

    Parameters
    ----------
    fit_folder : str
        Where the posterior fits are stored.
    save_dir : str
        Where to save the output figures.
    overlay_gaussians : boolean, optional
        Whether to overlay Gaussian estimate of distribution. Defaults to True.
    """
    os.makedirs(os.path.join(save_dir, "posterior_hists"), exist_ok=True)
    posteriors, labels, _ = oversample_using_posteriors(names, labels, OVERSAMPLE_SIZE, fit_folder)

    params, save_labels = param_labels(aux_bands)

    for i in range(1, len(params) - 1):
        feat_i = posteriors[:, i]

        if i in [2, 4, 5, 6]:
            feat_i = np.log10(feat_i)

        num_per_bin, bins, _ = plt.hist(feat_i, bins=100)
        bin_centers = (bins[1:] + bins[:-1]) / 2.0
        bin_centers = bin_centers[num_per_bin != 0]
        num_per_bin = num_per_bin[num_per_bin != 0]
        poisson_err = np.sqrt(num_per_bin)  # assume Poisson statistics
        plt.errorbar(bin_centers, num_per_bin, yerr=poisson_err, fmt="o")

        if overlay_gaussians:
            # estimate with mean and stddev
            mean_est = np.mean(feat_i)
            stddev_est = np.std(feat_i)
            amp_est = np.max(num_per_bin)
            plt.plot(
                bin_centers,
                gaussian(bin_centers, amp_est, mean_est, stddev_est),
                lw=2
            )

        plt.xlabel(params[i])
        plt.ylabel("Count")
        plt.savefig(
            os.path.join(
                save_dir, "posterior_hists", f"{save_labels[i]}.pdf"
            ), bbox_inches="tight"
        )
        plt.close()
