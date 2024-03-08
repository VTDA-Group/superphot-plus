"""This module contains scripts to plot sampling results."""
import os

import pacmap
import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np
import umap
import umap.plot

from superphot_plus.plotting.format_params import (
    param_labels,
    set_global_plot_formatting,
    CUSTOM_COLORSET
)

from superphot_plus.plotting.utils import gaussian
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.surveys.surveys import Survey
from superphot_plus.model.data import PosteriorSamplesGroup
from superphot_plus.format_data_ztf import retrieve_posterior_set

OVERSAMPLE_SIZE = 4000

set_global_plot_formatting()

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
    all_post_objs = retrieve_posterior_set(
        names, fits_dir, sampler='dynesty',
        redshifts=None,
        labels=labels,
        chisq_cutoff=1.2
    )
    psg = PosteriorSamplesGroup(all_post_objs)
    features, labels = psg.oversample()
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

    all_post_objs = retrieve_posterior_set(
        names, fits_dir, sampler=sampler,
        redshifts=None,
        labels=labels,
        chisq_cutoff=1.2
    )
    psg = PosteriorSamplesGroup(all_post_objs)
    psg.canonicalize_labels()
    features_gaussian, labels_gaussian = psg.oversample()
    features_smote, labels_smote = psg.oversample_smote()
    clipped_features = []
    clipped_features_smote = []
    feature_medians = []
    unique_labels = np.unique(labels)
    
    for l in unique_labels:
        random_idxs = np.random.choice(
            np.where(labels_gaussian == l)[0],
            size=goal_per_class,
            replace=False
        )
        clipped_features.append(
            features_gaussian[random_idxs]
        )
        random_idxs_smote = np.random.choice(
            np.where(labels_smote == l)[0],
            size=goal_per_class,
            replace=False
        )
        clipped_features_smote.append(
            features_smote[random_idxs_smote]
        )
        feature_medians.append(
            psg.median_features[psg.labels == l]
        )
    clipped_features = np.asarray(clipped_features)
    clipped_features_smote = np.asarray(clipped_features_smote)
    #feature_medians = np.asarray(feature_medians)
    
    params, save_labels = param_labels(aux_bands)

    for i, param_1 in enumerate(params):
        for j in range(i):
            if i in [0, 3, len(params)]:
                continue
            if j in [0, 3, len(params)]:
                continue

            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={"hspace": 0})

            smote_ax = axes[0]
            gauss_ax = axes[1]

            param_2 = params[j]
            
            for k,l in enumerate(unique_labels):
                if l not in allowed_types:
                    continue
                feature_means_t1 = np.array(feature_medians[k])[:,i]
                feature_means_t2 = np.array(feature_medians[k])[:,j]
                features_smote_t1 = clipped_features_smote[k, :, i]
                features_smote_t2 = clipped_features_smote[k, :, j]
                features_gauss_t1 = clipped_features[k, :, i]
                features_gauss_t2 = clipped_features[k, :, j]

                smote_ax.scatter(
                    features_smote_t1, features_smote_t2,
                    label=l, alpha=0.8, s=1
                )
                
                gauss_ax.scatter(
                    features_gauss_t1, features_gauss_t2,
                    label='Oversampled fits', alpha=0.8, s=1
                )
                
                smote_ax.scatter(
                    feature_means_t1, feature_means_t2,
                    label=l, s=3
                )

                gauss_ax.scatter(
                    feature_means_t1, feature_means_t2,
                    label='Median fits', s=3
                )

            # annotate with labels
            smote_ax.text(
                0.01, 0.99, "SMOTE",
                horizontalalignment='left',
                verticalalignment='top',
                c="black",
                transform=smote_ax.transAxes,
            )
            gauss_ax.text(
                0.01, 0.99, "Oversampling\nMultiple Fits\nPer Lightcurve",
                horizontalalignment='left',
                verticalalignment='top',
                c="black",
                transform=gauss_ax.transAxes,
            )
            #smote_ax.set_title("Oversampling using SMOTE vs Multiple Fits")
            gauss_ax.set_xlabel(param_1)
            gauss_ax.set_ylabel(param_2)
            smote_ax.set_ylabel(param_2)
            
            plt.legend(loc='lower left', fontsize=14)
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
    print(allowed_types)

    #goal_per_class = OVERSAMPLE_SIZE
    all_post_objs = retrieve_posterior_set(
        names, fits_dir, sampler='dynesty',
        redshifts=None,
        labels=labels,
        chisq_cutoff=1.2
    )
    psg = PosteriorSamplesGroup(all_post_objs)
    features_gaussian, labels_gaussian = psg.oversample()

    params, _ = param_labels(priors.aux_bands, priors.reference_band, log=False)

    fig, axes = plt.subplots(3, 4, figsize=(10, 12))
    axes = axes.ravel()

    prior_means = priors.to_numpy()[:, 2]
    prior_stddevs = priors.to_numpy()[:, 3]
        
    ax_num = 0
    for i in range(1, len(params) - 1):
        if i == 3:
            continue
        leg_lines = []
        param_1 = params[i]
        
        features_1_gauss = features_gaussian[:, i]
        labels_gauss = labels_gaussian
        
        if i in [1, 2, 8, 9]:
            """only include samples that actually have a plateau!"""
            constrained_plateau = features_gaussian[:, 2] > 0
            features_1_gauss = features_1_gauss[constrained_plateau]
            labels_gauss = labels_gauss[constrained_plateau]
        
        """
        if i == 10:
            param_1 = r"$10^4\times (t_\mathrm{0, g} - 1)$"
            features_1_gauss = 10000 * (features_1_gauss - 1)
            prior_means[i] = 10000 * (prior_means[i] - 1)
            prior_stddevs[i] = 10000 * prior_stddevs[i]
        """
        if i == 1:
            param_1 = r"$10^3\times \beta_\mathrm{r}$"
            features_1_gauss = 1000 * features_1_gauss
            prior_means[i] = 1000 * prior_means[i]
            prior_stddevs[i] = 1000 * prior_stddevs[i]

        log_scale = False

        if i not in [1, 3, 10, 14]:
            log_scale = True
            axes[ax_num].set_xscale("log")
            feature_hist, bin_edges = np.histogram(features_1_gauss, bins=50)
            bin_width = bin_edges[1] - bin_edges[0]
            bin_edges = 10**bin_edges

        else:
            feature_hist, bin_edges = np.histogram(features_1_gauss, bins=50)
            bin_width = bin_edges[1] - bin_edges[0]

        #feature_hist[np.abs(feature_hist) > 1e5] = 0
        current_area = np.sum(bin_width * feature_hist)
        feature_hist = (1.0 / current_area) * feature_hist

        feature_hist_all = feature_hist
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        for allowed_class in allowed_types:
            allowed_label = classes_to_labels[allowed_class]
            features_1_t = features_1_gauss[labels_gauss == allowed_label]
            
            if log_scale:
                feature_hist, bin_edges = np.histogram(10**features_1_t, bins=bin_edges)

            else:
                feature_hist, bin_edges = np.histogram(features_1_t, bins=bin_edges)
                
            current_area = np.sum(bin_width * feature_hist)
            feature_hist = (1.0 / current_area) * feature_hist
            #feature_hist[np.abs(feature_hist) > 1e5] = 0
            (legend_line,) = axes[ax_num].step(
                bin_centers, feature_hist, where="mid", label=allowed_label, alpha=0.8
            )
            leg_lines.append(legend_line)

        ax = axes[ax_num]
        (legend_line,) = ax.step(
            bin_centers,
            feature_hist_all,
            where="mid", c="k", label="Combined", alpha=0.8,
        )
        leg_lines.append(legend_line)

        if i < 14:
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
                label="Prior",
                alpha=0.8
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
            ax.locator_params(axis="x", numticks=3)
        else:
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
            ax.locator_params(axis="x", nbins=3)
        

        ax_num += 1

    legend_keys = [*list(labels_to_classes.keys()), "Combined", "Prior"]
    fig.legend(leg_lines, legend_keys, loc="lower center", ncol=4)
    

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

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

    all_post_objs = retrieve_posterior_set(
        names, fits_dir, sampler='dynesty',
        redshifts=None,
        labels=labels,
        chisq_cutoff=1.2
    )
    psg = PosteriorSamplesGroup(all_post_objs)
    features, labels = psg.oversample()

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
    all_post_objs = retrieve_posterior_set(
        names, fit_folder, sampler='dynesty',
        redshifts=None,
        labels=labels,
        chisq_cutoff=1.2
    )
    psg = PosteriorSamplesGroup(all_post_objs)
    posteriors, labels = psg.oversample()
    params, save_labels = param_labels(aux_bands)

    for i in range(len(params) - 1):
        feat_i = posteriors[:, i]

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
        plt.title(f"{mean_est} +- {stddev_est}")
        plt.savefig(
            os.path.join(
                save_dir, "posterior_hists", f"{save_labels[i]}.pdf"
            ), bbox_inches="tight"
        )
        plt.close()

        
def plot_feature_umap(psg, save_path):
    """Plot 2D UMAP of sampling features.
    
    Parameters
    ----------
    psg : PosteriorSamplesGroup
        The group of posteriors to map
    save_path : str
        Where to save the resulting figure.
    """
    features, labels = psg.oversample()
    # add jitter
    for i in range(features.shape[1]):
        features[:,i] += np.random.normal(scale=np.std(features) / 1e3, size=len(features))
    nan_features = np.any(np.isnan(features), axis=1)
    print(features)
    mapper = umap.UMAP().fit(features[~nan_features], force_all_finite=False)
    umap.plot.points(
        mapper,
        labels=labels[~nan_features],
        color_key_cmap='custom_categorical'
    )
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    
def plot_feature_pacmap(psg, save_path):
    """Plot 2D PACMAP of sampling features.
    
    Parameters
    ----------
    psg : PosteriorSamplesGroup
        The group of posteriors to map
    save_path : str
        Where to save the resulting figure.
    """
    features, labels = psg.oversample()
    # add jitter
    for i in range(features.shape[1]):
        features[:,i] += np.random.normal(scale=np.nanstd(features) / 100, size=len(features))
    nan_features = np.any(np.isnan(features), axis=1)
    embedding = pacmap.PaCMAP(n_components=2)#, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
    X_transformed = embedding.fit_transform(features[~nan_features], init="pca")

    labels = np.asarray(labels)[~nan_features]
    for l in np.unique(labels):
        subX = X_transformed[labels[~nan_features] == l]
        # visualize the embedding
        plt.scatter(
            subX[:, 0],
            subX[:, 1],
            s=0.6,
            alpha=0.3,
            label=l
        )
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    