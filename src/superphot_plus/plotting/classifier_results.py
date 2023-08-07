import matplotlib.pyplot as plt
import numpy as np
import csv
import os

from superphot_plus.supernova_class import SupernovaClass as SnClass

from superphot_plus.plotting.format_params import *

def save_class_fractions(spec_probs_csv, phot_probs_csv, save_fn):
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
        Filename for saving the class fractions.
    """
    labels_to_class, _ = SnClass.get_type_maps()
    true_classes = []
    pred_classes = []
    pred_classes_spec = []
    alerce_preds = []
    alerce_preds_spec = []
    true_classes_alerce = []

    ct = 0
    with open(spec_probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            ct += 1
            print(ct)
            try:
                alerce_pred = labels_to_class[get_pred_class(row[0], reflect_style=True)]
            except:
                continue
            alerce_preds_spec.append(alerce_pred)
            l = int(row[1][-2])
            true_classes.append(l)
            if l == 2:
                true_classes_alerce.append(1)
            else:
                true_classes_alerce.append(l)
            pred_classes_spec.append(np.argmax(np.array(row[2:])))

    with open(phot_probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for e, row in enumerate(csvreader):
            print(e)
            name = row[0]
            if row[1] == "SKIP":
                continue
            try:
                alerce_pred = labels_to_class[get_pred_class(name, reflect_style=True)]
                # print(alerce_pred, e)
            except:
                print(name, " skipped")
                continue
            alerce_preds.append(alerce_pred)
            pred_classes.append(np.argmax(np.array(row[2:])))

    true_classes = np.array(true_classes)
    pred_classes = np.array(pred_classes)
    alerce_preds = np.array(alerce_preds)

    cm_p = confusion_matrix(true_classes, pred_classes_spec, normalize="pred")
    cm_p_alerce = confusion_matrix(true_classes_alerce, alerce_preds_spec, normalize="pred")

    true_fracs = np.array([len(true_classes[true_classes == i]) / len(true_classes) for i in range(5)])
    pred_fracs = np.array([len(pred_classes[pred_classes == i]) / len(pred_classes) for i in range(5)])
    alerce_fracs = np.array([len(alerce_preds[alerce_preds == i]) / len(alerce_preds) for i in range(5)])

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

    with open(save_fn, "a+") as sf:
        csvwriter = csv.writer(sf)
        csvwriter.writerow(true_fracs)
        csvwriter.writerow(pred_fracs)
        csvwriter.writerow(pred_fracs_corr)
        csvwriter.writerow(alerce_fracs)
        csvwriter.writerow(alerce_fracs_corr)


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

    fracs = []
    with open(saved_cf_file, "r") as sf:
        csvreader = csv.reader(sf)
        for row in csvreader:
            fracs.append([float(x) for x in row])

    true_fracs = fracs[0]
    pred_fracs = fracs[1]
    pred_fracs_corr = fracs[2]
    alerce_fracs = fracs[3]
    alerce_fracs_corr = fracs[4]

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
    
    
    
    
def generate_roc_curve(probs_csv):
    classes_to_labels = {0: "SN Ia", 1: "SN II", 2: "SN IIn", 3: "SLSN-I", 4: "SN Ibc"}
    fpr = []
    tpr = []
    leg_lines = []
    colors = [plt.cm.Set1(i) for i in range(10)]

    fig, ax = plt.subplots(1,2,figsize=(8, 7))
    ax1, ax2 = ax
    for ref_label in range(0,5):
        true_labels = []
        simplified_probs = []
        with open(probs_csv, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if int(row[1][-2]) == ref_label:
                    true_labels.append(1)
                else:
                    true_labels.append(0)
                simplified_probs.append(float(row[2+ref_label]))
        y_true = np.array(true_labels)
        y_score = np.array(simplified_probs)

        print(y_true, y_score)
        f, t, threshholds = roc_curve(y_true, y_score)
        idx_50 = np.argmin((threshholds - 0.5)**2)
        #print(classes_to_labels[ref_label], t[threshholds > 0.7][-1])
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
        leg_lines.append(l)
        #ax1.scatter(f[idx_50], t[idx_50], color=colors[ref_label], s=40)
        ax2.scatter(f[idx_50], t[idx_50], color=colors[ref_label], s=100, marker="d")
        fpr.append(f)
        tpr.append(t)
        
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 5
    
    
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    
    ax2.set_xlim([0.0, 0.1])
    ax2.set_ylim([0.0, 1.05])
    
    for ax_i in ax:
        l, = ax_i.plot(
            all_fpr,
            mean_tpr,
            label="Macro-averaged",
            linewidth=3,
            linestyle="dashed",
            c="black"
        )
        
        ratio = 1.2
        x_left, x_right = ax_i.get_xlim()
        y_low, y_high = ax_i.get_ylim()
        
        ax_i.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    
    plt.locator_params(axis='x', nbins=3)

    leg_lines.append(l)
    
    for ax_i in ax:
        ax_i.yaxis.set_minor_locator(AutoMinorLocator())
        ax_i.xaxis.set_minor_locator(AutoMinorLocator())
        ax_i.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    
    fig.legend(leg_lines, ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc", "Combined"], loc='lower center', ncol=3)
    plt.savefig("../figs/roc_all.pdf", bbox_inches='tight')
            

def generate_roc_curve_binary(probs_csv):
    fpr = []
    tpr = []
    true_labels = []
    simplified_probs = []
    with open(probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if float(row[1]) == 0:
                true_labels.append(1)
            else:
                true_labels.append(0)
            simplified_probs.append(float(row[2]))
    y_true = np.array(true_labels)
    y_score = np.array(simplified_probs)
    print(y_true, y_score)
    f, t, threshholds = roc_curve(y_true, y_score)
    plt.plot(
        f,
        t,
        label="Non-Recurring Transients",
    )
    for cutoff in [0.2, 0.3, 0.4, 0.5]:
        print(cutoff, f[np.abs(threshholds - cutoff) < 1e-3], t[np.abs(threshholds - cutoff) < 1e-3])

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig("../figs/roc_top_level.png")

    
def parameter_2d_plots():
    """
    Plot 2D scatterplots for each pair
    of fit parameters, to identify clustering
    among different subclasses.
    """
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    labels_to_vals = {allowed_types[i]: i for i in range(len(allowed_types))}
    pt_colors = ["r", "c", "k", "m", "g"]
    input_csv = "training_data.csv"
    params = ["A", "beta", "gamma", "t0", "tau_rise", "tau_fall", "extra_sigma", \
             "A_g", "beta_g", "gamma_g", "t0_g", "tau_rise_g", "tau_fall_g", "extra_sigma_g"]
    names, feature_means, feature_std, labels = import_features_and_labels(input_csv, allowed_types)
    features, labels = oversample_using_posteriors(names, labels, 15000)
    color_arr = [labels_to_vals[l] for l in labels]
    for i in range(len(params)):
        print(i)
        for j in range(i):
            print(i, j)
            param_1 = params[i]
            param_2 = params[j]
            features_1 = features[:,i]
            features_2 = features[:,j]
            
            if param_1 == "A":
                features_1 = np.log10(features_1)
            if param_2 == "A":
                features_2 = np.log10(features_2)
            for t_idx in range(len(allowed_types)):
                t = allowed_types[t_idx]
                features_1_t = features_1[labels == t]
                features_2_t = features_2[labels == t]
                plt.scatter(features_1_t[0], features_2_t[0], s=2, alpha=1, label=allowed_types[t_idx])
                plt.scatter(features_1_t, features_2_t, s=2, alpha=0.005)
            plt.xlabel(param_1)
            plt.ylabel(param_2)
            plt.legend()
            plt.savefig("../figs/param_compare/%s_vs_%s.png" % (param_1, param_2))
            plt.close()
    """
    print(features[:10])
    model = TSNE(n_components=2, perplexity=15, verbose=2, init="random", random_state=1, n_iter=5000)
    y_space = model.fit_transform(features)
    x_vals = np.array([x[0] for x in y_space])
    y_vals = np.array([x[1] for x in y_space])
    
    for t_idx in range(len(allowed_types)):
        t = allowed_types[t_idx]
        x_vals_t = x_vals[labels == t]
        y_vals_t = y_vals[labels == t]
        plt.scatter(x_vals_t[0], y_vals_t[0], s=2, alpha=1, label=allowed_types[t_idx])
        plt.scatter(x_vals_t, y_vals_t, s=2, alpha=0.005)
    plt.legend()
    plt.savefig("../figs/tsne_15000.png")
    plt.close()
    """
        

def plot_param_distributions(plot_priors=True):
    """
    Plot the parameter distributions to get better priors for fitting.
    """
    #allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ib/c"]
    #input_csv = "training_data.csv"
    #num_params = 42
    num_params = 14
    post_folder = "/gpfs/group/vav5084/default/kdesoto/ztf/dynesty_fits_11_2022/"
    #post_folder = "/gpfs/group/vav5084/default/kdesoto/elasticc/fits/"
    
    for post_fn in glob.glob(post_folder+"*.npz"):
        A_cutoff = 0.
        new_posts = import_elasticc_features(post_fn)
        new_posts = new_posts[new_posts[:,0] >= A_cutoff]
        try:
            posteriors = np.vstack((posteriors, new_posts[:50]))
        except:
            print("REPLACING")
            posteriors = new_posts
        
    features = posteriors
    #features, labels = oversample_minority_classes(features, labels) #smote
    #max_fluxes = features[:,14] / (1. + np.exp(-features[:,16] / features[:,18])) * (1. - features[:,15] * features[:,16])
    #max_fluxes = features[:,0] / (1. + np.exp(-features[:,2] / features[:,4])) * (1. - features[:,1] * features[:,2])


    os.makedirs("../figs/param_dist_ztf_final/", exist_ok=True)
    
    for i in range(num_params):
        feat = features[:,i]
        
        #if i == 20.:
        #if i == 6:
        #    feat = feat / max_fluxes

        #if i in [14, 16, 18, 19, 20]:
        if i in [0, 2, 4, 5, 6]:
            print(feat[feat <= 0])
            feat = np.log10(feat)

        
        #if param == "extra_sigma_g":
        #    n, bins, patches = plt.hist(feat, bins=100, range=(0.6, 1.2))
        #else:
        n, bins, patches = plt.hist(feat, bins=100)
        bin_centers = (bins[1:] + bins[:-1]) / 2.
        bin_centers = bin_centers[n != 0]
        n = n[n != 0]
        s = np.sqrt(n)

        popt, pcov = curve_fit(gaussian, bin_centers, n, p0=[5000., 1., 0.00005, 0.], sigma=s, bounds=([50., 0., 0., -50.], [100000., 1e20, 1e20, 200.]), maxfev=1e5, ftol=1e-10)
        plt.errorbar(bin_centers, n, yerr=s, fmt="o")
        plt.plot(bin_centers, gaussian(bin_centers, *popt), lw=2)
        print(i, popt[1], popt[2])
        plt.xlabel(i)
        plt.ylabel("Count")
        plt.savefig("../figs/param_dist_elasticc_final/%d.png" % i)
        plt.close()

def plot_phase_vs_accuracy(phased_probs_csv):
    """
    Plot classification accuracy as a function of phase.
    """
    
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={'hspace': 0})
    ax, ax2 = axes
    classes_to_labels = {0: "SN Ia", 1: "SN II", 2: "SN IIn", 3: "SLSN-I", 4: "SN Ibc"}
    allowed_types = np.arange(5)
    correct_class = [] # 1 if yes, 0 if no
    true_type = []
    phase = []
    pred_type = []
    
    with open(phased_probs_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            correct = (int(row[0][-2]) == np.argmax([float(row[i]) for i in range(2,7)]))
            correct_class.append(float(correct))
            phase.append(float(row[1]))
            true_type.append(int(row[0][-2]))
            pred_type.append(np.argmax([float(row[i]) for i in range(2,7)]))
    
    correct_class = np.array(correct_class)
    true_type = np.array(true_type)
    phase = np.array(phase)
    pred_type = np.array(pred_type)
    
    #acc_hist_comb = np.zeros(len(bins)-1)
    
    leg_lines = []
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
        leg_lines.append(l)
    
    #acc_hist = acc_hist_comb / 3
    #l, = ax.step(bins, np.append(acc_hist, acc_hist[-1]), where='post', color="k", label="Average")
    #leg_lines.append(l)
    
    ax.axvline(x=0.0, color="grey", linestyle="dotted")
    ax.set_ylabel("Classification Accuracy")
    ax.set_xlim((-18., 48.))
 
    # also plot the over/under-classification fraction of each type compared to final classification
    leg_lines = []
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
        leg_lines.append(l)
        
    ax2.axhline(y=1.0, color="k", xmin=-30, xmax=50, linestyle="--")
    ax2.axvline(x=0.0, color="grey", linestyle="dotted")
    ax2.set_xlabel(r"Phase (days)")
    ax2.set_ylabel("Overprediction Fraction")
    ax2.set_xlim((-18., 48.))
    fig.legend(leg_lines, ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"], loc='lower center', ncol=3)
    plt.savefig("../figs/phase_vs_accuracy_5_9.pdf", bbox_inches="tight")
    plt.close()
        
   
   
    

def plot_redshifts_abs_mags(probs_csv):
    """
    Plot redshift and absolute magnitude distributions used in the
    redshift-inclusive classifier.
    """
    #labels_to_classes = {"tensor(0)": "SN Ia", "tensor(1)": "SN II", "tensor(2)": "SN IIn", "tensor(3)": "SLSN-I", "tensor(4)": "SN Ibc"}
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    names, labels, redshifts = import_labels_only([probs_csv,], allowed_types, redshift=True)
    goal_per_class = 4000
    goal_per_name = int(np.round(goal_per_class / len(names)))

    feature_means = []
    
    for i, name in enumerate(names):
        features_single, labels_single, chis_single = oversample_using_posteriors(names[i:i+1], labels[i:i+1], np.ones(1), goal_per_name)
        feature_means.append(np.mean(features_single, axis=0))
        
    feature_means = np.array(feature_means)
    amplitudes = feature_means[:,0]
    
    app_mags = -2.5 * np.log10(amplitudes) + 26.3

    k_correction = 2.5 * np.log10(1.+redshifts)
    dist = cosmo.luminosity_distance([redshifts]).value[0]  # returns dist in Mpc
    abs_mags = app_mags - 5.0 * np.log10(dist * 1e6 / 10.0) + k_correction
    
    fig, axes = plt.subplots(1, 2)
    z_ax = axes[0]
    mag_ax = axes[1]

    mag_hist,bin_edges = np.histogram(-abs_mags, bins=40, density=True, range=(15,25))
    bin_width = bin_edges[1] - bin_edges[0]
    mag_cumsum = np.cumsum(mag_hist) * bin_width
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    leg_lines = []
    
    for at in allowed_types:
        a = at
        features_1_t = -abs_mags[labels == a]
        feature_hist,bin_edges = np.histogram(features_1_t, bins=bin_edges, density=True)
        cumsum = np.cumsum(feature_hist) * bin_width
        l, = mag_ax.step(-bin_centers, cumsum, where='mid', label=a)
        leg_lines.append(l)

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

        
    fig.legend(leg_lines, ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc", "Combined"], loc='lower center', ncol=3)
    plt.savefig("../figs/abs_mag_hist.pdf", bbox_inches="tight")
    plt.close()

    
    
def plot_snr_trends(probs_snr_csv, specific_class=None):
    """
    Generate plots of number of SNR > 5 points versus
    accuracy, and top 10% SNR versus accuracy.
    """

    classes_to_labels = {0: "SN Ia", 1: "SN II", 2: "SN IIn", 3: "SLSN-I", 4: "SN Ibc"}
    correct_class = [] # 1 if yes, 0 if no
    snr = []
    true_type = []
    n_high_snr = []
    with open(probs_snr_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if specific_class is not None and int(row[1]) != specific_class:
                continue
            correct = (int(row[1][-2]) == np.argmax([float(row[i]) for i in range(2,7)]))
            #print(int(row[1]), np.argmax([float(row[i]) for i in range(2,7)]))
            correct_class.append(float(correct))
            snr.append(float(row[-4]))
            n_high_snr.append(int(row[-3]))
            true_type.append(int(row[1][-2]))
            
    true_type = np.array(true_type)
    snr = np.array(snr)
    correct_class = np.array(correct_class)
    n_high_snr = np.array(n_high_snr)
    
    for t in np.unique(true_type):
        snr_t = snr[true_type == t]
        correct_t = correct_class[true_type == t]
        
        snr_vs_accuracy, snr_bin_edges, _ = binned_statistic(snr_t, correct_t, 'mean', bins=histedges_equalN(snr_t, 8))
        cts_per_bin, _, _ = binned_statistic(snr_t, np.ones(len(correct_t)), 'sum', bins=snr_bin_edges)
        
        print(cts_per_bin)

        snr_vs_accuracy[np.isnan(snr_vs_accuracy)] = 1.

        plt.step(snr_bin_edges, np.append(snr_vs_accuracy, snr_vs_accuracy[-1]), label=classes_to_labels[t], where="post")
        
    plt.xlim((5, 30))

    plt.xlabel("90th Percentile SNR")
    plt.ylabel("Classification Accuracy")
    plt.legend()
    if specific_class is not None:
        plt.savefig("../figs/snr_vs_accuracy_5_9_%d.pdf" % specific_class)
    else:
        plt.savefig("../figs/snr_vs_accuracy_5_9.pdf")
    plt.close()
    
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
    if specific_class is not None:
        plt.savefig("../figs/n_vs_accuracy_ztf_5_9_%d.pdf" % specific_class)
    else:
        plt.savefig("../figs/n_vs_accuracy_ztf_5_9_.pdf")
    plt.close()
    
    """
    plt.hist(snr, bins=20)
    if specific_class is not None:
        plt.savefig("../figs/snr_dist_elasticc_%d.png" % specific_class)
    else:
        plt.savefig("../figs/snr_dist_elasticc.png")
    plt.close()
    """

def plot_snr_trends2(probs_snr_csv, filename_prefix):
    """
    Replicates SNR plots needed for publication.
    """
    snr = []
    n_snr_3 = []
    n_snr_5 = []
    n_snr_10 = []
    max_r_flux = []
    exclude_ct = 0
    with open(probs_snr_csv, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            snr.append(float(row[-4]))
            n_snr_3.append(int(row[-3]))
            n_snr_5.append(int(row[-2]))
            n_snr_10.append(int(row[-1]))
            mf = -2.5*np.log10(float(row[-5]))+26.3
            if mf < 11.:
                exclude_ct += 1
                continue
            else:
                max_r_flux.append(mf)
    print(exclude_ct, len(n_snr_3))

    plt.hist(n_snr_3, histtype='step', label=r'$3\sigma$', bins=np.arange(0, 603, 3))
    plt.hist(n_snr_5, histtype='step', label=r'$5\sigma$', bins=np.arange(0, 603, 3))
    plt.hist(n_snr_10, histtype='step', label=r'$10\sigma$', bins=np.arange(0, 603, 3))
    plt.loglog()
    plt.xlabel("Number of Datapoints at Given SNR")
    plt.ylabel("Number of Lightcurves")
    plt.legend()
    plt.savefig("../figs/"+filename_prefix+"_snr_hist.pdf")
    plt.close()
    
    plt.hist(max_r_flux, histtype='stepfilled', bins=30)
    plt.yscale('log')
    plt.xlabel("Apparent Magnitude (m)")
    plt.ylabel("Number of Lightcurves")
    plt.savefig("../figs/"+filename_prefix+"_appm_hist.pdf")
    plt.close()

def compare_mag_distributions(probs_classified, probs_unclassified, filename_prefix):
    """
    Generate overlaid magnitude distributions of the classified and unclassified datasets.
    """
    max_r_classified = []
    max_r_unclassified = []
    max_r_skipped = []
    exclude_ct = 0
    with open(probs_classified, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            mf = -2.5*np.log10(float(row[-5]))+26.3
            #if mf < 11.:
            #    exclude_ct += 1
            #    continue
            #else:
            max_r_classified.append(mf)
                
    with open(probs_unclassified, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            mf = -2.5*np.log10(float(row[-5]))+26.3

            if mf < 0.:
                exclude_ct += 1
                continue
                
            if row[1] == "SKIP":
                max_r_skipped.append(mf)
                
            else:
                max_r_unclassified.append(mf)
                
                
    plt.hist(max_r_classified, histtype='stepfilled', bins=np.arange(5., 21., 0.5), alpha = 0.5, label="Classified", density=True)
    plt.hist(max_r_unclassified, histtype='stepfilled', bins=np.arange(5., 21., 0.5), alpha = 0.5, label="Unclassified (included)", density=True)
    plt.hist(max_r_skipped, histtype='stepfilled', bins=np.arange(5., 21., 0.5), alpha = 0.5, label="Unclassified (excluded)", density=True)
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.xlabel("Apparent Magnitude (m)")
    plt.ylabel("Fraction of Lightcurves")
    plt.savefig("../figs/"+filename_prefix+"_appm_hist_compare.pdf")
    plt.close()
    
    
def save_chisquared_vs_accuracy(pred_class_file):
    """
    Plots classification accuracy as a function of chisquared.
    """
    FITS_DIR = "/storage/group/vav5084/default/superphot+/dynesty_fits_unclassified_5_9_2023/"
    DATA_DIRS = ["/storage/group/vav5084/default/superphot+/data_reformatted_unclassified_05_09_2023/",]
    
    sn_names = []
    true_classes = []
    pred_classes = []
    with open(pred_class_file, "r") as orig:
        csv_reader = csv.reader(orig, delimiter=",")
        for row in csv_reader:
            if row[0] in sn_names:
                continue
            if not os.path.exists(FITS_DIR + row[0] + "_eqwt.npz"):
                continue
            sn_names.append(row[0])
            #true_classes.append(classes_to_labels[row[1]])
            #pred_classes.append(np.argmax([float(x) for x in row[2:7]]))
    
    #true_classes = np.array(true_classes)
    #pred_classes = np.array(pred_classes)
    train_chis = calculate_chi_squareds(sn_names, FITS_DIR, DATA_DIRS)
    
    #correctly_classified = np.where(true_classes == pred_classes, 1, 0)
    
    with open("chisq_vs_accuracy_phot.csv", "w+") as ca:
        ca.write("")
        
    with open("chisq_vs_accuracy_phot.csv", "a") as ca:
        csvwriter = csv.writer(ca, delimiter=",")
        for e, train_chi in enumerate(train_chis):
            #csvwriter.writerow([sn_names[e], true_classes[e], train_chis[e], correctly_classified[e]])
            csvwriter.writerow([sn_names[e], train_chis[e]])
    
    
def plot_chisquared_vs_accuracy(saved_csv_spec, saved_csv_phot):
    
    fig, ax2 = plt.subplots(figsize=(7,4.8))
    ax1 = ax2.twinx()

    train_chis = []
    train_chis_phot = []
    correctly_classified = []
    with open(saved_csv_spec, "r") as ca:
        csvreader = csv.reader(ca, delimiter=",")
        for row in csvreader:
            correctly_classified.append(int(row[3]))
            train_chis.append(float(row[2]))

    with open(saved_csv_phot, "r") as ca:
        csvreader = csv.reader(ca, delimiter=",")
        for row in csvreader:
            train_chis_phot.append(float(row[1]))
    
    train_chis = -1*np.array(train_chis)
    train_chis_phot = -1*np.array(train_chis_phot)
    correctly_classified = np.array(correctly_classified)
    
    #idx_cutoff = ( train_chis < 50 )

    #correctly_classified = correctly_classified[idx_cutoff]
    #train_chis = train_chis[idx_cutoff]
    bins=np.arange(3.5,14,0.5)
    
    correct_hist,bin_edges,_ = binned_statistic(train_chis, correctly_classified, statistic='sum', bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
        
    all_hist,_,_ = binned_statistic(train_chis, np.ones(len(train_chis)), statistic='sum', bins=bins)
    all_hist_phot,_,_ = binned_statistic(train_chis_phot, np.ones(len(train_chis_phot)), statistic='sum', bins=bins)
    
    ax2.hist(bin_centers, bin_edges, weights=all_hist, color="purple", alpha=0.5, label="Spectroscopic")
    ax2.hist(bin_centers, bin_edges, weights=all_hist_phot, color="red", alpha=0.5, label="Photometric")
    
    all_hist[all_hist == 0] = np.inf
    
    acc_hist = correct_hist / all_hist
    
    idx_keep = (bin_centers < 10) & (bin_centers > 5)
    ax1.step(bin_centers[idx_keep], acc_hist[idx_keep], where='mid', color="blue", linewidth=3, label="Accuracy")
    ax1.axvline(x = 10, color='black', linestyle="--", linewidth=4, label= r"Phot. $\chi^2$ cutoff")

    
    bin_width = bin_edges[1] - bin_edges[0]
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
    
    plt.savefig("../figs/chisq_vs_accuracy.pdf", bbox_inches="tight")
    plt.close()
    