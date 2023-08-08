import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from superphot_plus.supernova_class import SupernovaClass as SnClass

from superphot_plus.plotting.format_params import *


def read_probs_csv(probs_fn):
    """Helper function to read in a probability csv file
    and return the columns as numpy arrays.
    """
    df = pd.from_csv(probs_fn)
    names = spec_df.name.to_numpy()
    labels = spec_df.label.to_numpy()
    probs = spec_df.iloc[2:7]
    pred_classes = np.argmax(probs, axis=1)
    
    return names, labels, probs, pred_classes
    
    
def save_class_fractions(spec_probs_csv, phot_probs_csv, save_path):
    """Save class fractions from spectroscopic, photometric, and
    corrected photometric.

    Parameters
    ----------
    spec_probs_csv : str
        Path to the CSV file containing spectroscopic probability
        predictions.
    phot_probs_csv : str
        Path to the CSV file containing photometric probability
        predictions.
    save_fn : str
        Filename + dir for saving the class fractions.
    """
    labels_to_class, _ = SnClass.get_type_maps()

    # import spec dataframe
    names_spec, true_class_spec, _, pred_class_spec = read_probs_csv(spec_probs_csv)
    
    true_class_alerce = labels_true
    true_class_alerce[true_class_alerce == 2] = 1  
    pred_class_spec_alerce = np.array([labels_to_class[x] for x in get_pred_class(names_spec, reflect_style=True)])
    
    # import phot dataframe
    names_phot, pred_label_alerce, _, pred_class_phot = read_probs_csv(phot_probs_csv)
    pred_class_phot_alerce = np.array([labels_to_class[x] for x in pred_label_alerce])

    cm_p = confusion_matrix(true_class_spec, pred_class_spec, normalize="pred")
    cm_p_alerce = confusion_matrix(true_class_alerce, pred_class_spec_alerce, normalize="pred")

    true_fracs = np.array([len(true_classes[true_class_spec == i]) / len(true_class_spec) for i in range(5)])
    pred_fracs = np.array([len(pred_classes[pred_class_phot == i]) / len(pred_class_phot) for i in range(5)])
    alerce_fracs = np.array([len(alerce_preds[pred_class_phot_alerce == i]) / len(pred_class_phot_alerce) for i in range(5)])

    pred_fracs_corr = []
    alerce_fracs_corr = []
    for i in range(5):
        pred_fracs_corr.append(np.sum(pred_fracs * cm_p[i]))
        if i == 2:
            alerce_fracs_corr.append(0.0)
        elif i > 2:
            alerce_fracs_corr.append(np.sum(np.delete(alerce_fracs, 2) * cm_p_alerce[i - 1]))
        else:
            alerce_fracs_corr.append(np.sum(np.delete(alerce_fracs, 2) * cm_p_alerce[i]))

    pred_fracs_corr = np.array(pred_fracs_corr)
    alerce_fracs_corr = np.array(alerce_fracs_corr)

    save_df = pd.DataFrame(
        {
            'spec_fracs': true_fracs,
            'phot_fracs': pred_fracs,
            'phot_fracs_corr': pred_fracs_corr,
            'alerce_fracs': alerce_fracs,
            'alerce_fracs_corr': alerce_fracs_corr
        }
    )
    save_df.to_csv(save_path, index=False)


def plot_class_fractions(saved_cf_file, fig_dir, filename):
    """Plot class fractions saved from 'save_class_fractions'.

    Parameters
    ----------
    saved_cf_file : str
        Path to the saved class fractions file.
    fig_dir : str
        Directory for saving the class fractions plot.
    filename: str
        Filename for the class fractions plot figure.
    """
    _, classes_to_labels = SnClass.get_type_maps()
    labels = [
        "Spec (ZTF)",
        "Spec (YSE)",
        "Spec (PS1-MDS)",
        "Phot",
        "Phot (corr.)",
        "ALeRCE",
        "ALeRCE (corr.)",
    ]
    width = 0.6

    frac_df = pd.from_csv(saved_cf_file)

    true_fracs, pred_fracs, pred_fracs_corr, alerce_fracs, alerce_fracs_corr = frac_df.to_numpy().T

    # Plot YSE class fractions too
    yse_counts = np.array([314, 107, 15, 2, 32])
    yse_fracs = yse_counts / np.sum(yse_counts)

    # Plot PS-MDS
    psmds_counts = np.array([404, 94, 24, 17, 19])
    psmds_fracs = psmds_counts / np.sum(psmds_counts)

    combined_fracs = np.array(
        [
            true_fracs,
            yse_fracs,
            psmds_fracs,
            pred_fracs,
            pred_fracs_corr,
            alerce_fracs,
            alerce_fracs_corr,
        ]
    ).T
    fig, ax = plt.subplots(figsize=(11, 16))  # pylint: disable=unused-variable
    bar = ax.bar(labels, combined_fracs[0], width, label=classes_to_labels[0])
    for i in range(len(combined_fracs[0])):
        bari = bar.patches[i]
        ax.annotate(
            round(combined_fracs[0][i], 3),
            (bari.get_x() + bari.get_width() / 2, bari.get_y() + bari.get_height() / 2),
            ha="center",
            va="center",
            color="white",
        )

    for i in range(1, 5):
        bar = ax.bar(
            labels,
            combined_fracs[i],
            width,
            bottom=np.sum(combined_fracs[0:i], axis=0),
            label=classes_to_labels[i],
        )
        for j in range(len(combined_fracs[0])):
            barj = bar.patches[j]
            # Create annotation
            ax.annotate(
                round(combined_fracs[i][j], 3),
                (barj.get_x() + barj.get_width() / 2, barj.get_y() + barj.get_height() / 2),
                ha="center",
                va="center",
                color="white",
            )

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=5, fontsize=15
    )

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    # plt.legend(loc=3)
    plt.ylabel("Fraction", fontsize=20)
    plt.savefig(os.path.join(fig_dir, filename))
    plt.close()
    
    
    
def generate_roc_curve(probs_csv, save_dir):
    """Generate a combined ROC curve of all SN classes.
    
    Parameters
    ----------
    probs_csv : str
        CSV file where class probabilities are stored.
    save_dir : str
        Where to save the figure.
    """
    labels_to_class, classes_to_labels = SnClass.get_type_maps()

    colors = [plt.cm.Set1(i) for i in range(10)]
    fig, ax = plt.subplots(1,2,figsize=(8, 7))
    ax1, ax2 = ax
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 0.1])
    ax2.set_ylim([0.0, 1.05])
    ax1.set_ylabel("True Positive Rate")
    ratio = 1.2
    plt.locator_params(axis='x', nbins=3)

    legend_lines = []
    for ref_label in range(len(classes_to_labels)):
        names, true_labels, probs, preds = read_probs_csv(probs_csv)
        y_true = np.where(true_labels == ref_label, 1, 0)
        y_score = probs[:,ref_label]

        f, t, threshholds = roc_curve(y_true, y_score)
        idx_50 = np.argmin((threshholds - 0.5)**2)
        l, = ax1.plot(
            f,
            t,
            label="%s" % classes_to_labels[ref_label], c=colors[ref_label]
        )
        ax2.plot(
            f,
            t,
            label="%s" % classes_to_labels[ref_label], c=colors[ref_label]
        )
        legend_lines.append(l)
        #ax1.scatter(f[idx_50], t[idx_50], color=colors[ref_label], s=40)
        ax2.scatter(f[idx_50], t[idx_50], color=colors[ref_label], s=100, marker="d")
        fpr.append(f)
        tpr.append(t)
        
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(len(classes_to_labels)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes_to_labels)

    for ax_i in ax:
        l, = ax_i.plot(
            all_fpr,
            mean_tpr,
            label="Macro-averaged",
            linewidth=3,
            linestyle="dashed",
            c="black"
        )
        
        x_left, x_right = ax_i.get_xlim()
        y_low, y_high = ax_i.get_ylim()
        
        ax_i.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
        ax_i.yaxis.set_minor_locator(AutoMinorLocator())
        ax_i.xaxis.set_minor_locator(AutoMinorLocator())
        ax_i.set_xlabel("False Positive Rate")
    
    legend_lines.append(l)
    
    fig.legend(leg_lines, [*list(labels_to_classes.keys()), "Combined"], loc='lower center', ncol=3)
    plt.savefig(os.path.join(save_dir, "roc_all.pdf"), bbox_inches='tight')
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
    """
    labels_to_class, classes_to_labels = SnClass.get_type_maps()
    pt_colors = ["r", "c", "k", "m", "g"]
    
    param_labels = [
        r"$A$", r"$\beta$", r"$\log_{10}\gamma$", r"$t_0$", r"$\log_{10}\tau_\mathrm{rise}$", \
        r"$\log_{10}\tau_\mathrm{fall}", r"$\log_{10}\sigma_\mathrm{extra}$", \
        r"$A_\mathrm{g}$", r"$\beta_\mathrm{g}$", r"$\gamma_\mathrm{g}$", \
        r"$t_{0,\mathrm{g}}$", r"$\tau_\mathrm{rise, g}$", r"$\tau_\mathrm{fall, g}$",
        r"$\sigma_\mathrm{extra, g}$"
    ]

    for post_fn in glob.glob(os.path.join(fit_folder,"*.npz")):
        new_posts = np.load(post_fn)["arr_0"]
        try:
            features = np.vstack((features, new_posts[:50]))
        except:
            features = new_posts
    
    #color_arr = [labels_to_vals[l] for l in labels]
    for i in range(1,len(param_labels)):
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

                    
def plot_phase_vs_accuracy(phased_probs_csv, save_dir):
    """Plot classification accuracy as a function of phase.
    
    Parameters
    ----------
    phased_probs_csv : str
        Where classification probabilities and LC truncated phases are saved.
    save_dir : str
        Where to save the output figures.
    """
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={'hspace': 0})
    ax, ax2 = axes
    _, classes_to_labels = SnClass.get_type_maps()
    allowed_types = np.arange(len(classes_to_labels))
    
    true_type, phase, _, pred_type = read_probs_csv(phased_probs_csv)
    correct_class = (true_type == pred_type).astype(int)
        
    legend_lines = []
    for at in allowed_types:
        correct_t = correct_class[true_type == at]
        phase_t = phase[true_type == at]
        
        bins = np.arange(-16, 52, 4)
        #bins = histedges_equalN(phase_t[phase_t > -18.], 20)

        correct_hist,bin_edges,_ = binned_statistic(phase_t, correct_t, statistic='sum', bins=bins)
        all_hist,_,_ = binned_statistic(phase_t, np.ones(len(phase_t)), statistic='sum', bins=bins)
        acc_hist_t = correct_hist / all_hist
        #acc_hist_comb += acc_hist_t
        l, = ax.step(bins, np.append(acc_hist_t, acc_hist_t[-1]), where='post', label=at)
        legend_lines.append(l)
    
    #acc_hist = acc_hist_comb / 3
    #l, = ax.step(bins, np.append(acc_hist, acc_hist[-1]), where='post', color="k", label="Average")
    #leg_lines.append(l)
    
    ax.axvline(x=0.0, color="grey", linestyle="dotted")
    ax.set_ylabel("Classification Accuracy")
    ax.set_xlim((-18., 48.))
 
    # also plot the over/under-classification fraction of each type compared to final classification
    legend_lines = []
    #bins_eq=histedges_equalN(phase[phase > -30.], 20) # all points
    bins_eq = np.arange(-16, 52, 4)
    all_hist,_,_ = binned_statistic(phase, np.ones(len(true_type)), statistic='sum', bins=bins_eq)

    idxs_type = []

    for at in allowed_types:
        eff_num = np.zeros(len(bins_eq) - 1) # effective numerator
        for at2 in allowed_types:
            idx_sub = (true_type == at2)
            phase_t = phase[idx_sub]
            
            bins_eq = np.arange(-16, 52, 4)

            true_hist,_,_ = binned_statistic(phase_t, np.ones(len(phase_t)), statistic='sum', bins=bins_eq)
            frac_hist = (true_hist / all_hist) # within each bin, fraction that is that true type

            normed_const = 0.2 / frac_hist
            
            # get fraction of true type at2 classified as at, and add it to total 'at' fraction
            idx_sub2 = ( (true_type == at2) & (pred_type == at) )
            phase_sub = phase[idx_sub2]
            if len(phase_sub) == 0:
                continue
            pred_hist,_,_ = binned_statistic(phase_sub, np.ones(len(phase_sub)), statistic='sum', bins=bins_eq)
            
            eff_num += normed_const * pred_hist
            
            #acc_hist_comb += acc_hist_t
        pred_frac = eff_num / all_hist
        pred_frac_normed = pred_frac / pred_frac[-1]
        l, = ax2.step(bins_eq, np.append(pred_frac_normed, pred_frac_normed[-1]), where='post', label=at)
        legend_lines.append(l)
        
    ax2.axhline(y=1.0, color="k", xmin=-30, xmax=50, linestyle="--")
    ax2.axvline(x=0.0, color="grey", linestyle="dotted")
    ax2.set_xlabel(r"Phase (days)")
    ax2.set_ylabel("Overprediction Fraction")
    ax2.set_xlim((-18., 48.))
    fig.legend(leg_lines, [classes_to_labels[x] for x in allowed_types], loc='lower center', ncol=3)
    plt.savefig(os.path.join(save_dir, "phase_vs_accuracy.pdf"), bbox_inches="tight")
    plt.close()
        
    

def plot_redshifts_abs_mags(probs_snr_csv, save_dir):
    """
    Plot redshift and absolute magnitude distributions used in the
    redshift-inclusive classifier.
    
    Parameters
    ----------
    probs_snr_csv : str
        Where probabilities + SNRs are stored.
    save_dir : str
        Where to save figures.
    """
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()
    allowed_types = list(labels_to_classes.keys())
    
    names, labels, redshifts = import_labels_only([probs_snr_csv,], allowed_types, redshift=True)
    
    df = pd.from_csv(probs_snr_csv)
    amplitudes = df.iloc[:,-5].to_numpy()
    app_mags = -2.5 * np.log10(amplitudes) + 26.3

    k_correction = 2.5 * np.log10(1.+redshifts)
    dist = cosmo.luminosity_distance([redshifts]).value[0]  # returns dist in Mpc
    abs_mags = app_mags - 5.0 * np.log10(dist * 1e6 / 10.0) + k_correction
    
    fig, axes = plt.subplots(1, 2)
    z_ax = axes[0]
    mag_ax = axes[1]

    mag_hist, bin_edges = np.histogram(-abs_mags, bins=40, density=True, range=(15,25))
    bin_width = bin_edges[1] - bin_edges[0]
    mag_cumsum = np.cumsum(mag_hist) * bin_width
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    legend_lines = []
    
    for at in allowed_types:
        a = at
        features_1_t = -abs_mags[labels == a]
        feature_hist,bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        l, = mag_ax.step(-bin_centers, cumsum, where='mid', label=a)
        legend_lines.append(l)

    #l, = mag_ax.step(bin_centers, mag_cumsum, where='mid', c="k", label="Combined", linewidth=2)
    #leg_lines.append(l)
    mag_ax.set_xlabel("Absolute Magnitude")
    #mag_ax.set_yticklabels([])
    #mag_ax.set_yticks([])
    mag_ax.invert_xaxis()

    z_hist,bin_edges = np.histogram(redshifts, bins=40, density=True, range=(-0.1, 0.6))
    bin_width = bin_edges[1] - bin_edges[0]
    z_cumsum = np.cumsum(z_hist) * bin_width
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    
    for at in allowed_types:
        a = at
        features_1_t = redshifts[labels == a]
        feature_hist,bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        l, = z_ax.step(bin_centers, cumsum, where='mid', label=a)

    #l, = z_ax.step(bin_centers, z_cumsum, where='mid', c="k", label="Combined", linewidth=2)
    z_ax.set_xlabel("Redshift")
    z_ax.set_ylabel("Cumulative Fraction")
    #z_ax.set_yticklabels([])
    #z_ax.set_yticks([])

    for ax in axes:
        ratio = 1.0
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
        
    fig.legend(legend_lines, [*allowed_types, "Combined"], loc='lower center', ncol=3)
    plt.savefig(os.path.join(save_dir, "abs_mag_hist.pdf"), bbox_inches="tight")
    plt.close()

    
    
def plot_snr_npoints_vs_accuracy(probs_snr_csv, save_dir):
    """
    Generate plots of number of SNR > 5 points versus
    accuracy, and top 10% SNR versus accuracy.
    
    TODO: add functionality for only one type.
    
    Parameters
    ----------
    probs_snr_csv : str
        Where probabilities + SNRs are stored.
    save_dir : str
        Where to save figures.
    """

    labels_to_classes, classes_to_labels = SnClass
    
    names, true_type, _, pred_classes = read_probs_csv(probs_snr_csv)
    correct_class = np.where(true_classes == pred_classes, 1, 0)
    
    df = pd.from_csv(probs_snr_csv)
    snr, n_high_snr = df.iloc[:,-4:-2]
    
    for t in np.unique(true_type):
        snr_t = snr[true_type == t]
        correct_t = correct_class[true_type == t]
        
        snr_vs_accuracy, snr_bin_edges, _ = binned_statistic(snr_t, correct_t, 'mean', bins=histedges_equalN(snr_t, 8))
        cts_per_bin, _, _ = binned_statistic(snr_t, np.ones(len(correct_t)), 'sum', bins=snr_bin_edges)

        snr_vs_accuracy[np.isnan(snr_vs_accuracy)] = 1.

        plt.step(snr_bin_edges, np.append(snr_vs_accuracy, snr_vs_accuracy[-1]), label=classes_to_labels[t], where="post")
        
    plt.xlim((5, 30))

    plt.xlabel("90th Percentile SNR")
    plt.ylabel("Classification Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "snr_vs_accuracy.pdf"))
    plt.close()
    
    # second plot
    for t in np.unique(true_type):
        correct_t = correct_class[true_type == t]
        n_high_t = n_high_snr[true_type == t]

        n_vs_accuracy, n_bin_edges, _ = binned_statistic(n_high_t, correct_t, 'mean', bins=histedges_equalN(n_high_t, 8))
        n_vs_accuracy[np.isnan(n_vs_accuracy)] = 1.

        plt.step(n_bin_edges, np.append(n_vs_accuracy, n_vs_accuracy[-1]), label=classes_to_labels[t], where="post")
        
    plt.xlim((8, 100))

    plt.xlabel(r"Number of $\geq 3\sigma$ Datapoints")
    plt.ylabel("Classification Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "n_vs_accuracy.pdf"))
    plt.close()


def plot_snr_hist(probs_snr_csv, save_dir):
    """
    Replicates SNR plots needed for publication.
    
    Parameters
    ----------
    probs_snr_csv : str
        Where probability + SNR info is stored.
    save_dir : str
        Where to save figure.
    """
    df = pd.from_csv(probs_snr_csv)
    snr, n_snr_3, n_snr_5, n_snr_10 = df.iloc[:,-4:].to_numpy().T
    skip_mask = (df.iloc[:,1] == "SKIP").to_numpy()

    plt.hist(n_snr_3[~skip_mask], histtype='step', label=r'$3\sigma$', bins=np.arange(0, 603, 3))
    plt.hist(n_snr_5[~skip_mask], histtype='step', label=r'$5\sigma$', bins=np.arange(0, 603, 3))
    plt.hist(n_snr_10[~skip_mask], histtype='step', label=r'$10\sigma$', bins=np.arange(0, 603, 3))
    plt.loglog()
    plt.xlabel("Number of Datapoints at Given SNR")
    plt.ylabel("Number of Lightcurves")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "snr_hist.pdf"))
    plt.close()

    
def compare_mag_distributions(probs_classified, probs_unclassified, save_dir, zeropoint=26.3):
    """
    Generate overlaid magnitude distributions of the classified and unclassified datasets.
    Assumes that unclassified LCs that did not pass the chi-squared cut are marked as "SKIP".
    
    Parameters
    ----------
    probs_classified : str
        CSV filename where probs of spectroscopic set are stored.
    probs_unclassified : str
        CSV filename where probs of photometric set are stored.
    save_dir : str
        Where to save figure.
    zeropoint : float, optional
        Zeropoint used when converting mags to fluxes. Defaults to 26.3.
    """
    classified_df = pd.from_csv(probs_classified)
    max_flux = classified_df.iloc[:,-5].to_numpy()
    max_r_classified = -2.5*np.log10(max_flux) + zeropoint
    
    unclassified_df = pd.from_csv(probs_classified)
    max_flux = unclassified_df.iloc[:,-5].to_numpy()
    max_r_unclassified_all = -2.5*np.log10(max_flux) + zeropoint
        
    mask_high_chisquared = (unclassified_df.iloc[:,1] == "SKIP").to_numpy()
    max_r_unclassified = max_r_unclassified_all[~mask_high_chisquared]
    max_r_skipped = max_r_unclassified_all[mask_high_chisquared]
                
    plt.hist(max_r_classified, histtype='stepfilled', bins=np.arange(5., 21., 0.5), alpha = 0.5, label="Classified", density=True)
    plt.hist(max_r_unclassified, histtype='stepfilled', bins=np.arange(5., 21., 0.5), alpha = 0.5, label="Unclassified (included)", density=True)
    plt.hist(max_r_skipped, histtype='stepfilled', bins=np.arange(5., 21., 0.5), alpha = 0.5, label="Unclassified (excluded)", density=True)
    
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.xlabel("Apparent Magnitude (m)")
    plt.ylabel("Fraction of Lightcurves")
    plt.savefig(os.path.join(save_dir, "appm_hist_compare.pdf"))
    plt.close()
    
    
def plot_chisquared_vs_accuracy(pred_spec_fn, pred_phot_fn, save_dir):
    """
    Plot chi-squared value histograms for both the spectroscopic and photometric
    datasets, and plot spec chi-squared as a function of classification accuracy.
    
    Parameters
    ----------
    pred_spec_fn : str
        CSV filename where probs of spectroscopic set are stored.
    pred_phot_fn : str
        CSV filename where probs of photometric set are stored.
    save_dir : str
        Where to save figure.
    """
    sn_names, true_classes, _, pred_classes = read_probs_csv(pred_spec_fn)
    
    correctly_classified = np.where(true_classes == pred_classes, 1, 0)
    train_chis = -1*calculate_neg_chi_squareds(sn_names, FITS_DIR, DATA_DIRS)
    
    sn_names, _, _, _ = read_probs_csv(pred_phot_fn)
    train_chis_phot = -1*calculate_neg_chi_squareds(sn_names, FITS_DIR, DATA_DIRS)
    
    # plot
    fig, ax2 = plt.subplots(figsize=(7,4.8))
    ax1 = ax2.twinx()
    bins=np.arange(3.5,14,0.5)
    
    correct_hist,bin_edges,_ = binned_statistic(train_chis, correctly_classified, statistic='sum', bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    bin_width = bin_edges[1] - bin_edges[0]

        
    all_hist,_,_ = binned_statistic(train_chis, np.ones(len(train_chis)), statistic='sum', bins=bins)
    all_hist_phot,_,_ = binned_statistic(train_chis_phot, np.ones(len(train_chis_phot)), statistic='sum', bins=bins)
    
    ax2.hist(bin_centers, bin_edges, weights=all_hist, color="purple", alpha=0.5, label="Spectroscopic")
    ax2.hist(bin_centers, bin_edges, weights=all_hist_phot, color="red", alpha=0.5, label="Photometric")
    
    all_hist[all_hist == 0] = np.inf
    
    acc_hist = correct_hist / all_hist
    
    idx_keep = (bin_centers < 10) & (bin_centers > 5)
    ax1.step(bin_centers[idx_keep], acc_hist[idx_keep], where='mid', color="blue", linewidth=3, label="Accuracy")
    ax1.axvline(x = 10, color='black', linestyle="--", linewidth=4, label= r"Phot. $\chi^2$ cutoff")

    
    # put bin counts on top of bars 
    """
    for bin_i in range(len(bins)-1):
        try:
            height = acc_hist[bin_i]
            plt.annotate(
                '%d' % all_hist[bin_i],
                xy=(bin_centers[bin_i], height),
                xytext=(1, 1), # 3 points vertical offset
                textcoords="offset points",
                fontsize=10,
                ha='center', va='bottom'
            )
        except:
            plt.annotate(
                '0',
                xy=(bin_centers[bin_i], height),
                xytext=(1, 1), # 3 points vertical offset
                textcoords="offset points",
                fontsize=10,
                ha='center', va='bottom'
            )
            
    """
    
    ax2.set_xlabel(r"Reduced $\chi^2$")
    ax1.set_ylabel("Accuracy", va='bottom', rotation=270)
    ax2.set_ylabel("Counts")
    ax2.legend()
    
    ax1.yaxis.label.set_color('blue')
    ax1.spines['right'].set_color('blue')
    ax1.tick_params(axis='y', colors='blue')

    ax1.legend(loc="lower right")
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.savefig(os.path.join(save_dir, "chisq_vs_accuracy.pdf"), bbox_inches="tight")
    plt.close()
    