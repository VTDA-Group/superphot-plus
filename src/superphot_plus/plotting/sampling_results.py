import matplotlib.pyplot as plt
import numpy as np
import os

import arviz as az
import corner

from superphot_plus.format_data_ztf import oversample_using_posteriors, import_labels_only
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.surveys.surveys import Survey
from superphot_plus.plotting.utils import get_numpyro_cube
from superphot_plus.plotting.format_params import *


def corner_plot_all(input_csvs, save_dir, aux_bands=["g",]):
    """Plot combined corner plot of all training set samples, excluding
    the overall scaling A.

    Parameters
    ----------
    input_csvs : list
        List of input CSV file paths containing probability predictions.
    save_dir : str
        Path to save the combined corner plot.
    """
    allowed_types = SnClass.all_classes()
    names, labels = import_labels_only(input_csvs, allowed_types)

    features, labels = oversample_using_posteriors(names, labels, 4000)
    plotting_labels = param_labels(aux_bands)
    
    # remove A, t0, and chisquared
    plotting_labels = [x for i, x in enumerate(plotting_labels) if i not in [0,3,len(plotting_labels-1)]]

    figure = corner.corner(
        np.delete(features[:,:-1], [0, 3], axis=1),
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
    for p in posterior_samples:
        post_reformatted[p] = np.array(
            [
                posterior_samples[p],
            ]
        )

    az.plot_trace(post_reformatted, compact=True)
    plt.savefig(output_file)
    plt.close()


def compare_oversampling(input_csv, allowed_types=SnClass.all_classes()):
    """
    Compare plots of various oversampling methods.
    
    Parameters
    ----------
    input_csv : str
        Where supernova list is stored.
    allowed_types : array-like
        Types to include in plot.
    """
    names, labels = import_labels_only(input_csv, allowed_types)
    
    goal_per_class = 4000
    features_gaussian, labels_gaussian = oversample_using_posteriors(names, labels, goal_per_class)
    
    feature_means = []
    labels_ordered = []
    
    start_idx = 0
    for t in np.unique(labels):
        type_idx = (labels == t)
        labels_ordered.extend(labels[type_idx])
        names_t = names[type_idx]
        samples_per_fit = max(1, int(np.round(goal_per_class / len(names_t))))
        
        for e, name in enumerate(names_t):
            feature_means.append(np.mean(features_gaussian[start_idx:start_idx+samples_per_fit], axis=0))
            start_idx += samples_per_fit
        
        
    feature_means = np.array(feature_means)
    feature_means_filler = np.ones((goal_per_class, len(feature_means[0])))
    labels_filler = 100*np.ones(goal_per_class)
    
    feature_means_comb = np.vstack((feature_means, feature_means_filler))
    labels_comb = np.append(labels_ordered, labels_filler)
    features_smote_comb, labels_smote_comb = oversample_minority_classes(feature_means_comb, labels_comb)
    
    features_smote = features_smote_comb[labels_smote_comb == allowed_types[0]]
    labels_smote = labels_smote_comb[labels_smote_comb == allowed_types[0]]

    params = param_labels(["g",])
   
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={'hspace': 0})
    smote_ax = axes[0]
    gauss_ax = axes[1]
    
    for i in range(len(params)):
        for j in range(i):
            
            if i in [0,3,len(params)]:
                continue
            if j in [0,3, len(params)]:
                continue
            
            if i in [2,4,5,6]:
                plt.xscale("log")
            if j in [2,4,5,6]:
                plt.yscale("log")
                
            param_1 = params[i]
            param_2 = params[j]
            
            features_1_smote = features_smote[:,i]
            features_2_smote = features_smote[:,j]
            
            for a in allowed_types:
                feature_means_t1 = feature_means[:,i][labels_ordered == a]
                feature_means_t2 = feature_means[:,j][labels_ordered == a]
                features_smote_t1 = features_1_smote[labels_smote == a]
                features_smote_t2 = features_2_smote[labels_smote == a]
                features_gauss_t1 = features_1_gauss[labels_gaussian == a]
                features_gauss_t2 = features_2_gauss[labels_gaussian == a]
               
                smote_ax.scatter(features_smote_t1, features_smote_t2, label=a, alpha=0.2, s=1, c="red")
                smote_ax.scatter(feature_means_t1, feature_means_t2, label=a, s=3, c="black")
                
                gauss_ax.scatter(feature_means_t1, feature_means_t2, label=a, s=3, c="black")
                gauss_ax.scatter(features_gauss_t1, features_gauss_t2, label=a, alpha=0.2, s=1, c="red")
                
            plt.title("Oversampling using SMOTE vs Multiple Fits")
            gauss_ax.xlabel(param_1)
            gauss_ax.ylabel(param_2)
            smote_ax.ylabel(param_2)
            #plt.legend()
            plt.savefig(os.path.join(save_dir, "oversample_compare", "%s_vs_%s.pdf" % (param_1, param_2)), bbox_inches="tight")
            plt.close()

            
def plot_oversampling_1d(input_csv, priors=Survey.ZTF().priors):
    """
    Save all 1d oversampled histograms for each parameter, in one plot.
    Overlays prior distributions.
    
    Parameters
    ----------
    input_csv : str
        CSV listing all supernovae to include.
    priors : MultibandPriors
        Prior object to overlay prior distributions.
    """
    allowed_types = SnClass.allowed_classes()
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    names, classes = import_labels_only(input_csv, allowed_types)
    labels = np.array([classes_to_labels[c] for c in classes])
    
    goal_per_class = 4000
    features_gaussian, labels_gaussian = oversample_using_posteriors(names, labels, goal_per_class)

    params = param_labels(["g",])

    fig, axes = plt.subplots(3, 4, figsize=(8, 9))
    axes = axes.ravel()

    prior_means = priors.to_numpy()[:,2]
    prior_stddevs = priors.to_numpy()[:,3]
    
    ax_num = 0
    for i in range(1, len(params)):
        leg_lines = []
        param_1 = params[i]
        features_1_gauss = features_gaussian[:,i]
        
        if i == 3:
            continue
        if i == 10:
            param_1 = r"$10^4\times (t_\mathrm{0, g} - 1)$"
            features_1_gauss = 10000 * (features_1_gauss - 1)
            
        log_scale = False

        if i in [2,4,5,6]:
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

        if log_scale:
            bins_fine = np.linspace(np.log10(bin_centers[0]), np.log10(bin_centers[-1]), num=100)
            prior_dist = gaussian(bins_fine, 1., prior_means[i], prior_stddevs[i])
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
        
    fig.legend(leg_lines, [*allowed_types, "Combined", "Prior"], loc='lower center', ncol=4)
    plt.locator_params(axis='x', nbins=3)

    plt.savefig(os.path.join(save_dir, "all_1d_hists.pdf"), bbox_inches="tight")
    plt.close()
        

def plot_combined_posterior_space(fits_dir, save_dir):
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
    labels_to_class, classes_to_labels = SnClass.get_type_maps()
    pt_colors = ["r", "c", "k", "m", "g"]
    
    param_labels = param_labels(["g",])

    for post_fn in glob.glob(os.path.join(fit_folder,"*.npz")):
        new_posts = np.load(post_fn)["arr_0"]
        try:
            features = np.vstack((features, new_posts[:50]))
        except:
            features = new_posts
    
    #color_arr = [labels_to_vals[l] for l in labels]
    for i in range(1,len(param_labels)-1):
        for j in range(i):
            param_1 = param_labels[i]
            param_2 = param_labels[j]
            features_1 = features[:,i]
            features_2 = features[:,j]
            
            if i in [2,4,5,6]:
                features_1 = np.log10(features_1)
            if j in [2,4,5,6]:
                features_2 = np.log10(features_2)
                
            plt.scatter(features_1, features_2, s=2, alpha=0.005)
            """
            for t_idx in range(len(allowed_types)):
                t = allowed_types[t_idx]
                features_1_t = features_1[labels == t]
                features_2_t = features_2[labels == t]
                
            """
            plt.xlabel(param_1)
            plt.ylabel(param_2)
            leg = plt.legend()
            for lh in leg.legendHandles: 
                lh.set_alpha(1)
            plt.savefig(os.path.join(save_dir, "combined_2d_posteriors", f"{param_1}_vs_{param_2}.pdf"))
            plt.close()
        

def plot_param_distributions(fit_folder, save_dir, overlay_gaussians=True):
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
    for post_fn in glob.glob(os.path.join(fit_folder,"*.npz")):
        new_posts = np.load(post_fn)["arr_0"]
        try:
            posteriors = np.vstack((posteriors, new_posts[:50]))
        except:
            posteriors = new_posts

    num_params = posteriors.shape[1]
    for i in range(1,num_params):
            
        feat_i = posteriors[:,i]
        
        if i in [2, 4, 5, 6]:
            feat = np.log10(feat)

        n, bins, patches = plt.hist(feat_i, bins=100)
        bin_centers = (bins[1:] + bins[:-1]) / 2.
        bin_centers = bin_centers[n != 0]
        n = n[n != 0]
        s = np.sqrt(n) # assume Poisson statistics
        plt.errorbar(bin_centers, n, yerr=s, fmt="o")

        if overlay_gaussians:
            
            # estimate with mean and stddev
            mean_est = np.mean(feat_i)
            stddev_est = np.std(feat_i)
            amp_est = np.max(n)
            """
            popt, pcov = curve_fit(gaussian, bin_centers, n, p0=[5000., 1., 0.00005, 0.], sigma=s, bounds=([50., 0., 0., -50.], [100000., 1e20, 1e20, 200.]), maxfev=1e5, ftol=1e-10)
            """
            plt.plot(bin_centers, gaussian(bin_centers, [amp_est, mean_est, stddev_est]), lw=2)
        
        plt.xlabel(i)
        plt.ylabel("Count")
        plt.savefig(os.path.join(save_dir, "posterior_hists", "%d.pdf" % i))
        plt.close()