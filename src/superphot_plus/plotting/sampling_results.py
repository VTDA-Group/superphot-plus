import matplotlib.pyplot as plt
import numpy as np
import os

import arviz as az
import corner

from superphot_plus.format_data_ztf import oversample_using_posteriors, import_labels_only
from superphot_plus.plotting.utils import get_numpyro_cube

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
    allowed_types = SnClass.all_classes()
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


def compare_oversampling(input_csv):
    """
    Compare plots of various oversampling methods.
    """
    allowed_types = ["SLSN-I",]
    
    input_csv = ["../data/training_set_combined_05_09_2023.csv",]
    names, labels = import_labels_only(input_csv, allowed_types)
    print(names)
    goal_per_class = 4000
    goal_per_name = int(np.round(goal_per_class / len(names)))
    features_gaussian, labels_gaussian, chis = oversample_using_posteriors(names, labels, np.ones(len(labels)), goal_per_class)
    
    feature_means = []
    for i, name in enumerate(names):
        features_single, labels_single, chis_single = oversample_using_posteriors(names[i:i+1], labels[i:i+1], np.ones(1), goal_per_name)
        feature_means.append(np.mean(features_single, axis=0))
        
    feature_means = np.array(feature_means)
    
    feature_means_filler = np.ones((goal_per_class, len(feature_means[0])))
    labels_filler = np.ones(goal_per_class)
    
    feature_means_comb = np.vstack((feature_means, feature_means_filler))
    labels_comb = np.append(labels, labels_filler)
    features_smote_comb, labels_smote_comb = oversample_minority_classes(feature_means_comb, labels_comb)
    
    features_smote = features_smote_comb[labels_smote_comb == allowed_types[0]]
    labels_smote = labels_smote_comb[labels_smote_comb == allowed_types[0]]

    params = [r"$A$", r"$\beta$", r"$\gamma$", r"$t_0$", r"$\tau_{rise}$", r"$\tau_{fall}$", r"$\sigma_{extra}$", \
             r"$A_g$", r"$\beta_g$", r"$\gamma_g$", r"$t_{0, g}$", r"$\tau_{rise, g}$", r"$\tau_{fall, g}$", r"$\sigma_{extra, g}$"]
   
    for i in range(len(params)):
        for j in range(i):
            param_1 = params[i]
            param_2 = params[j]
            
            features_1_smote = features_smote[:,i]
            features_2_smote = features_smote[:,j]
            
            for a in allowed_types:
                features_1_all = feature_means[:,i][labels == a]
                features_2_all = feature_means[:,j][labels == a]
                if param_1 in [r"$A$", r"$\gamma$", r"$\tau_{rise}$", r"$\tau_{fall}$", r"$\sigma_{extra}$"]:
                    #features_1_smote = np.log(features_1_smote)
                    #features_1_all = np.log(features_1_all)
                    plt.xscale("log")
                if param_2 in [r"$A$", r"$\gamma$", r"$\tau_{rise}$", r"$\tau_{fall}$", r"$\sigma_{extra}$"]:
                    #features_2_smote = np.log(features_2_smote)
                    #features_2_all = np.log(features_2_all)
                    plt.yscale("log")
                features_1_t = features_1_smote[labels_smote == a]
                features_2_t = features_2_smote[labels_smote == a]
                plt.scatter(features_1_t, features_2_t, label=a, alpha=0.2, s=1, c="red")
                plt.scatter(features_1_all, features_2_all, label=a, s=3, c="black")
            plt.title("Oversampling using SMOTE")
            plt.xlabel(param_1)
            plt.ylabel(param_2)
            #plt.legend()
            plt.savefig("../figs/oversample_compare/%s_vs_%s_smote.png" % (param_1, param_2), bbox_inches="tight")
            plt.close()

            
            features_1_gauss = features_gaussian[:,i]
            features_2_gauss = features_gaussian[:,j]
            for a in allowed_types:
                features_1_all = feature_means[:,i][labels == a]
                features_2_all = feature_means[:,j][labels == a]
                if param_1 in [r"$A$", r"$\gamma$", r"$\tau_{rise}$", r"$\tau_{fall}$", r"$\sigma_{extra}$"]:
                    #features_1_smote = np.log(features_1_smote)
                    #features_1_all = np.log(features_1_all)
                    plt.xscale("log")
                if param_2 in [r"$A$", r"$\gamma$", r"$\tau_{rise}$", r"$\tau_{fall}$", r"$\sigma_{extra}$"]:
                    #features_2_smote = np.log(features_2_smote)
                    #features_2_all = np.log(features_2_all)
                    plt.yscale("log")
                features_1_t = features_1_gauss[labels_gaussian == a]
                features_2_t = features_2_gauss[labels_gaussian == a]
                plt.scatter(features_1_all, features_2_all, label=a, s=3, c="black")
                plt.scatter(features_1_t, features_2_t, label=a, alpha=0.2, s=1, c="red")
            plt.title("Oversampling using Multiple Fits per Lightcurve")
            plt.xlabel(param_1)
            plt.ylabel(param_2)
            #plt.legend()
            plt.savefig("../figs/oversample_compare/%s_vs_%s_gauss.png" % (param_1, param_2), bbox_inches="tight")
            plt.close()

def plot_all_oversampling(input_csv):
    """
    Compare plots of various oversampling methods.
    """
    allowed_types = ["tensor(0)", "tensor(1)", "tensor(2)", "tensor(3)", "tensor(4)"]
    labels_to_classes = {"tensor(0)": "SN Ia", "tensor(1)": "SN II", "tensor(2)": "SN IIn", "tensor(3)": "SLSN-I", "tensor(4)": "SN Ibc"}
    input_csv = ["../classifier/classified_probs_05_09_2023_paper.csv",]
    names, labels = import_labels_only(input_csv, allowed_types)
    labels = np.array([labels_to_classes[l] for l in labels])
    
    goal_per_class = 4000
    features_gaussian, labels_gaussian, chis = oversample_using_posteriors(names, labels, np.ones(len(labels)), goal_per_class)

    params = [r"$A$", r"$\beta$", r"$\gamma$", r"$t_0$", r"$\tau_{rise}$", r"$\tau_{fall}$", r"$\sigma_{extra}$", \
             r"$A_g$", r"$\beta_g$", r"$\gamma_g$", r"$t_{0, g}$", r"$\tau_{rise, g}$", r"$\tau_{fall, g}$", r"$\sigma_{extra, g}$"]
   
    for i in range(len(params)):
        for j in range(i):
            param_1 = params[i]
            param_2 = params[j]

            features_1_gauss = features_gaussian[:,i]
            features_2_gauss = features_gaussian[:,j]
            for at in allowed_types:
                a = labels_to_classes[at]
                if param_1 in [r"$A$", r"$\gamma$", r"$\tau_{rise}$", r"$\tau_{fall}$", r"$\sigma_{extra}$"]:
                    #features_1_smote = np.log(features_1_smote)
                    #features_1_all = np.log(features_1_all)
                    plt.xscale("log")
                if param_2 in [r"$A$", r"$\gamma$", r"$\tau_{rise}$", r"$\tau_{fall}$", r"$\sigma_{extra}$"]:
                    #features_2_smote = np.log(features_2_smote)
                    #features_2_all = np.log(features_2_all)
                    plt.yscale("log")
                features_1_t = features_1_gauss[labels_gaussian == a]
                features_2_t = features_2_gauss[labels_gaussian == a]
                plt.scatter(features_1_t, features_2_t, label=a, alpha=0.1, s=1)
            plt.xlabel(param_1)
            plt.ylabel(param_2)
            leg = plt.legend()
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
                lh.set_sizes([6.0])
                
            plt.savefig("../figs/oversampled_all/%s_vs_%s.png" % (param_1, param_2), bbox_inches="tight")
            plt.close()
            
def plot_oversampling_1d(input_csv):
    """
    Compare plots of various oversampling methods.
    """
    allowed_types = ["tensor(0)", "tensor(1)", "tensor(2)", "tensor(3)", "tensor(4)"]
    labels_to_classes = {"tensor(0)": "SN Ia", "tensor(1)": "SN II", "tensor(2)": "SN IIn", "tensor(3)": "SLSN-I", "tensor(4)": "SN Ibc"}
    input_csv = ["../classifier/classified_probs_05_09_2023_paper.csv",]
    names, labels = import_labels_only(input_csv, allowed_types)
    labels = np.array([labels_to_classes[l] for l in labels])
    
    goal_per_class = 4000
    features_gaussian, labels_gaussian, chis = oversample_using_posteriors(names, labels, np.ones(len(labels)), goal_per_class)

    params = [r"$A$", r"$\beta$", r"$\gamma$", r"$t_\mathrm{0}$", r"$\tau_\mathrm{rise}$", r"$\tau_\mathrm{fall}$", r"$\sigma_\mathrm{extra}$", \
             r"$A_\mathrm{g}$", r"$\beta_\mathrm{g}$", r"$\gamma_\mathrm{g}$", r"$10^4\times (t_\mathrm{0, g} - 1)$", r"$\tau_\mathrm{rise, g}$", r"$\tau_\mathrm{fall, g}$", r"$\sigma_\mathrm{extra, g}$"]
   

    fig, axes = plt.subplots(3,4,figsize=(8, 9))
    axes = axes.ravel()
    ax_num = 0
    prior_means = [p[2] for p in ALL_PRIORS]
    prior_stddevs = [p[3] for p in ALL_PRIORS]
    for i in range(len(params)):
        leg_lines = []
        param_1 = params[i]
        features_1_gauss = features_gaussian[:,i]
        
        if param_1 == r"$10^4\times (t_\mathrm{0, g} - 1)$":
            features_1_gauss = 10000 * (features_1_gauss - 1)
            
        log_scale = False
        if param_1 in [r"$A$", r"$t_\mathrm{0}$"]:
            continue
        if param_1 in [r"$A$", r"$\gamma$", r"$\tau_\mathrm{rise}$", r"$\tau_\mathrm{fall}$", r"$\sigma_\mathrm{extra}$"]:
            #features_1_smote = np.log(features_1_smote)
            #features_1_all = np.log(features_1_all)
            log_scale = True
            axes[ax_num].set_xscale("log")
            feature_hist,bin_edges = np.histogram(np.log10(features_1_gauss), bins=20)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_edges = 10**bin_edges
            feature_hist[np.abs(feature_hist) > 1e5] = 0
            current_area = np.sum(bin_width * feature_hist)
            feature_hist = (1. / current_area) * feature_hist
        else:
            bin_widths = 0
            feature_hist,bin_edges = np.histogram(features_1_gauss, bins=20, density=True)
                
        feature_hist_all = feature_hist
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        for at in allowed_types:
            a = labels_to_classes[at]
            features_1_t = features_1_gauss[labels_gaussian == a]
            
            if log_scale:
                feature_hist,bin_edges = np.histogram(features_1_t, bins=bin_edges)
                current_area = np.sum(bin_width * feature_hist)
                feature_hist = (1. / current_area) * feature_hist
            else:
                feature_hist,bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
                
            feature_hist[np.abs(feature_hist) > 1e5] = 0
            l, = axes[ax_num].step(bin_centers, feature_hist, where='mid', label=a)
            leg_lines.append(l)

        ax = axes[ax_num]
        l, = ax.step(bin_centers, feature_hist_all, where='mid', c="k", label="Combined", linewidth=2)
        leg_lines.append(l)
        
        # Plot prior distributions over each cell
        def gaussian(x, mu, stddev):
            return np.exp(-(x-mu)**2/(2.*stddev**2)) / np.sqrt(2. * np.pi) / stddev
        
        if log_scale:
            bins_fine = np.linspace(np.log10(bin_centers[0]), np.log10(bin_centers[-1]), num=100)
            prior_dist = gaussian(bins_fine, prior_means[i], prior_stddevs[i])
            bins_fine = 10**bins_fine
        else:
            bins_fine = np.linspace(bin_centers[0], bin_centers[-1], num=100)
            prior_dist = gaussian(bins_fine, prior_means[i], prior_stddevs[i])
        l, = ax.plot(bins_fine, prior_dist, linestyle='dashed', label='Prior', linewidth=2, c='magenta')
        leg_lines.append(l)
        ax.set_xlabel(param_1)
        ax.set_yticklabels([])
        ax.set_yticks([])
        
        ratio = 1.25
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        
        if log_scale:
            ax.set_aspect(abs(np.log10(x_right/x_left)/(y_low-y_high))*ratio)
        else:
            ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
        
        ax_num += 1
        
    fig.legend(leg_lines, ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc", "Combined", "Prior"], loc='lower center', ncol=4)
    plt.locator_params(axis='x', nbins=3)

    plt.savefig("../figs/oversampled_1d/all.pdf", bbox_inches="tight")
    plt.close()
        
