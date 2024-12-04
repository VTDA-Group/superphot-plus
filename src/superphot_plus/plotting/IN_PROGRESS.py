def save_class_fractions(spec_probs_csv, probs_alerce_csv, phot_probs_csv, probs_alerce_phot_csv, save_path):
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
    labels_to_class, classes_to_labels = SnClass.get_type_maps()

    # import spec dataframe
    _, true_class_spec, probs_spec, pred_class_spec, _, _ = read_probs_csv(spec_probs_csv)

    num_classes = probs_spec.shape[1]

    true_class_alerce = true_class_spec.copy()
    true_class_alerce[true_class_alerce == 2] = 1

    # read in ALeRCE classes
    df_alerce = pd.read_csv(probs_alerce_csv)
    pred_alerce = df_alerce.alerce_label.to_numpy().astype(str)

    ignore_mask = (pred_alerce == "None") | (pred_alerce == "nan") | (pred_alerce == "SKIP")
    # ignore true SNe IIn
    ignore_mask = ignore_mask | (true_class_alerce == 2)

    true_class_alerce = true_class_alerce[~ignore_mask]
    pred_alerce = pred_alerce[~ignore_mask]

    pred_class_spec_alerce = np.array([labels_to_class[x] for x in pred_alerce])

    # import phot dataframe
    pred_class_phot = read_probs_csv(phot_probs_csv)[3]
    pred_class_phot_alerce = retrieve_four_class_info(phot_probs_csv, probs_alerce_phot_csv)[4]

    cm_p = confusion_matrix(true_class_spec, pred_class_spec, normalize="pred")
    cm_p_alerce = confusion_matrix(true_class_alerce, pred_class_spec_alerce, normalize="pred")

    true_fracs = np.array(
        [len(true_class_spec[true_class_spec == i]) / len(true_class_spec) for i in range(num_classes)]
    )
    pred_fracs = np.array(
        [len(pred_class_phot[pred_class_phot == i]) / len(pred_class_phot) for i in range(num_classes)]
    )
    alerce_fracs = np.array(
        [
            len(pred_class_phot_alerce[pred_class_phot_alerce == classes_to_labels[i]]) / len(pred_class_phot_alerce) for i in range(num_classes)
        ]
    )

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
            "spec_fracs": true_fracs,
            "phot_fracs": pred_fracs,
            "phot_fracs_corr": pred_fracs_corr,
            "alerce_fracs": alerce_fracs,
            "alerce_fracs_corr": alerce_fracs_corr,
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

    frac_df = pd.read_csv(saved_cf_file)

    true_fracs, pred_fracs, pred_fracs_corr, alerce_fracs, alerce_fracs_corr = frac_df.to_numpy().T

    survey_sn_fracs = get_survey_fracs()
    yse_fracs, psmds_fracs = survey_sn_fracs["YSE"], survey_sn_fracs["PS-MDS"]

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
    _, ax = plt.subplots(figsize=(11, 16))

    for i in range(5):
        if i == 0:
            bottom = 0
        else:
            bottom = np.sum(combined_fracs[0:i], axis=0)
        stacked_bar = ax.bar(
            labels,
            combined_fracs[i],
            width,
            bottom=bottom,
            label=classes_to_labels[i],
        )
        for j, fracs_j in enumerate(combined_fracs[i]):
            if fracs_j == 0.0:
                continue
            barj = stacked_bar.patches[j]
            # Create annotation
            ax.annotate(
                round(fracs_j, 3),
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

    
def plot_metrics_over_mjd(mjd_bins, p_matrix, c_matrix, save_dir):
    labels_to_classes, classes_to_labels = SnClass.get_type_maps()

    colors = CUSTOM_COLORSET
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 10), gridspec_kw={"hspace": 0.1})
    ax, ax2 = axes
    ratio = 1.0
    ax2.set_ylabel("Purity")
    ax.set_ylabel("Completeness")
    ax.set_ylim((0, 1))
    ax2.set_ylim((0, 1))
    plt.locator_params(axis="x", nbins=3)

    legend_lines = []
    for i in range(5):
        mean_p = np.mean(p_matrix[i], axis=0)
        mean_c = np.mean(c_matrix[i], axis=0)
        p_err = np.std(p_matrix[i], axis=0)
        c_err = np.std(c_matrix[i], axis=0)
        
        (legend_line,) = ax.step(
            mjd_bins, mean_c, where='post',
            c=colors[i], label=classes_to_labels[i]
        )
        ax.fill_between(
            mjd_bins, mean_c - c_err, mean_c + c_err,
            color=colors[i], alpha=0.2, step='post'
        )
        ax2.step(
            mjd_bins, mean_p, where='post', c=colors[i]
        )
        ax2.fill_between(
            mjd_bins, mean_p - p_err, mean_p + p_err,
            color=colors[i], alpha=0.2, step='post'
        )
        legend_lines.append(legend_line)
    
    ax2.set_xlabel("MJD")
    fig.legend(loc="lower center", ncol=3)
    plt.savefig(os.path.join(save_dir, "metrics_over_mjd.pdf"), bbox_inches="tight")
    plt.close()
    
    
def plot_phase_vs_accuracy(phased_probs_dir, all_probs_csv, save_dir):
    """Plot classification accuracy as a function of phase.

    Parameters
    ----------
    phased_probs_dir : str
        Where classification probabilities and LC truncated phases are saved.
    save_dir : str
        Where to save the output figures.
    """
    fig, axes = plt.subplots(5, 3, sharex=True, figsize=(16, 16), gridspec_kw = {'hspace':0.15, 'wspace':0.3})
    _, classes_to_labels = SnClass.get_type_maps()
    allowed_types = np.arange(len(classes_to_labels))

    phases = []
    accs_full_means = []
    accs_full_stddevs = []
    accs_early_means = []
    accs_early_stddevs = []
    fracs_early_means = []
    fracs_early_stddevs = []
    fracs_full_means = []
    fracs_full_stddevs = []
    f1_early_means = []
    f1_early_stddevs = []
    f1_full_means = []
    f1_full_stddevs = []
    
    all_probs_files = glob.glob(
        os.path.join(
            phased_probs_dir,
            "full_*_concat.csv"
        )
    )
    
    full_probs_df = pd.read_csv(all_probs_csv)
    all_true_labels = full_probs_df.Label.to_numpy()
    class_counts = [len(all_true_labels[l == all_true_labels]) for l in allowed_types]
    class_fracs = np.asarray(class_counts) / np.sum(class_counts)
    
    n_phases = len(all_probs_files)
    
    for probs_file_full in all_probs_files:
        _, true_type, _, pred_type, folds, _ = read_probs_csv(probs_file_full)
        phase_counts = [len(true_type[l == true_type]) for l in allowed_types]
        phase_fracs = np.asarray(phase_counts) / np.sum(phase_counts)
    
        correct_class = (true_type == pred_type).astype(int)
        acc_mu_single = []
        acc_std_single = []
        fracs_mu_single = []
        fracs_std_single = []
        f1_mu_single = []
        f1_std_single = []
        
        for i, allowed_type in enumerate(allowed_types):
            accs = []
            fracs = []
            f1s = []
            for f in range(10):
                correct_t = correct_class[(folds == f) & (true_type == allowed_type)]
                if len(correct_t) == 0.0:
                    correct_sum = 0.0
                else:
                    correct_sum = np.sum(correct_t)
                if len(true_type[(true_type == allowed_type) & (folds == f)]) == 0:
                    completeness = 1.0
                else:
                    completeness = correct_sum / len(true_type[(true_type == allowed_type) & (folds == f)])
                all_preds = pred_type[(pred_type == allowed_type) & (folds == f)]
                all_trues = true_type[(pred_type == allowed_type) & (folds == f)]
                adj_pred = np.sum([
                    class_fracs[j] * len(all_preds[all_trues == at2]) / phase_fracs[j] for j, at2 in enumerate(allowed_types)
                ])
                if adj_pred == 0.0:
                    purity = 1.0
                else:
                    purity = class_fracs[i] * correct_sum / adj_pred / phase_fracs[i]
                accs.append(completeness)
                fracs.append(purity)
                if purity == 0 and completeness == 0:
                    f1s.append(0)
                else:
                    f1s.append(2 * purity * completeness / (purity + completeness))
            
            fracs_mu_single.append(np.nanmean(fracs))
            fracs_std_single.append(np.nanstd(fracs))
            acc_mu_single.append(np.nanmean(accs))
            acc_std_single.append(np.nanstd(accs))
            f1_mu_single.append(np.nanmean(f1s))
            f1_std_single.append(np.nanstd(f1s))
            
            
        accs_full_means.append(acc_mu_single)
        accs_full_stddevs.append(acc_std_single)
        fracs_full_means.append(fracs_mu_single)
        fracs_full_stddevs.append(fracs_std_single)
        f1_full_means.append(f1_mu_single)
        f1_full_stddevs.append(f1_std_single)
        
        phase = probs_file_full.split("/")[-1].split("_")[1]

        if round(float(phase), 2) == 0.61:
            print("PHASE ZERO FULL")
            print(acc_mu_single, acc_std_single)
            print(fracs_mu_single, fracs_std_single)
            
        if round(float(phase), 2) == 70.00:
            print("PHASE LATE FULL")
            print(acc_mu_single, acc_std_single)
            print(fracs_mu_single, fracs_std_single)
                
        phases.append(float(phase))
        probs_file_early = os.path.join(
            phased_probs_dir,
            f"early_{phase}_concat.csv"
        )
        _, true_type, _, pred_type, folds, _ = read_probs_csv(probs_file_early)
        correct_class = (true_type == pred_type).astype(int)
        acc_mu_single = []
        acc_std_single = []
        fracs_mu_single = []
        fracs_std_single = []
        f1_mu_single = []
        f1_std_single = []
        
        for i, allowed_type in enumerate(allowed_types):
            accs = []
            fracs = []
            f1s = []
            for f in range(10):
                correct_t = correct_class[(folds == f) & (true_type == allowed_type)]
                if len(correct_t) == 0.0:
                    correct_sum = 0.0
                else:
                    correct_sum = np.sum(correct_t)
                if len(true_type[(true_type == allowed_type) & (folds == f)]) == 0:
                    completeness = 1.0
                else:
                    completeness = correct_sum / len(true_type[(true_type == allowed_type) & (folds == f)])
                all_preds = pred_type[(pred_type == allowed_type) & (folds == f)]
                all_trues = true_type[(pred_type == allowed_type) & (folds == f)]
                adj_pred = np.sum([
                    class_fracs[j] * len(all_preds[all_trues == at2]) / phase_fracs[j] for j, at2 in enumerate(allowed_types)
                ])
                if adj_pred == 0.0:
                    purity = 1.0
                else:
                    purity = class_fracs[i] * correct_sum / adj_pred / phase_fracs[i]
                accs.append(completeness)
                fracs.append(purity)
                if purity == 0 and completeness == 0:
                    f1s.append(0)
                else:
                    f1s.append(2 * purity * completeness / (purity + completeness))
            
            fracs_mu_single.append(np.nanmean(fracs))
            fracs_std_single.append(np.nanstd(fracs))
            acc_mu_single.append(np.nanmean(accs))
            acc_std_single.append(np.nanstd(accs))
            f1_mu_single.append(np.nanmean(f1s))
            f1_std_single.append(np.nanstd(f1s))
            
        accs_early_means.append(acc_mu_single)
        accs_early_stddevs.append(acc_std_single)
        fracs_early_means.append(fracs_mu_single)
        fracs_early_stddevs.append(fracs_std_single)
        f1_early_means.append(f1_mu_single)
        f1_early_stddevs.append(f1_std_single)
        
        if round(float(phase), 2) == 0.61:
            print("PHASE ZERO EARLY")
            print(acc_mu_single, acc_std_single)
            print(fracs_mu_single, fracs_std_single)
            
        if round(float(phase), 2) == 70.00:
            print("PHASE LATE EARLY")
            print(acc_mu_single, acc_std_single)
            print(fracs_mu_single, fracs_std_single)
            
        
    sort_idx = np.argsort(phases)
    phases = np.asarray(phases)[sort_idx]
    accs_early_means = np.asarray(accs_early_means)[sort_idx].T
    accs_early_stddevs = np.asarray(accs_early_stddevs)[sort_idx].T
    accs_full_means = np.asarray(accs_full_means)[sort_idx].T
    accs_full_stddevs = np.asarray(accs_full_stddevs)[sort_idx].T
    fracs_early_means = np.asarray(fracs_early_means)[sort_idx].T
    fracs_early_stddevs = np.asarray(fracs_early_stddevs)[sort_idx].T
    fracs_full_means = np.asarray(fracs_full_means)[sort_idx].T
    fracs_full_stddevs = np.asarray(fracs_full_stddevs)[sort_idx].T
    f1_early_means = np.asarray(f1_early_means)[sort_idx].T
    f1_early_stddevs = np.asarray(f1_early_stddevs)[sort_idx].T
    f1_full_means = np.asarray(f1_full_means)[sort_idx].T
    f1_full_stddevs = np.asarray(f1_full_stddevs)[sort_idx].T
    
    legend_lines = []
    for i, allowed_type in enumerate(allowed_types):
        
        # now plots each in grid pattern
        ax = axes[i,0]
        ax2 = axes[i,1]
        ax3 = axes[i,2]
        
        (legend_line,) = ax.plot(
            phases, accs_full_means[i], label=allowed_type, color=CUSTOM_COLORSET[i]
        )
        ax.plot(
            phases, accs_early_means[i], linestyle='dashed', color=CUSTOM_COLORSET[i]
        )
        ax.fill_between(
            phases, accs_full_means[i]-accs_full_stddevs[i],
            accs_full_means[i]+accs_full_stddevs[i], alpha=0.2, color=CUSTOM_COLORSET[i]
        )
        legend_lines.append(legend_line)
        
        ax2.plot(
            phases, fracs_full_means[i], label=allowed_type, color=CUSTOM_COLORSET[i]
        )
        ax2.plot(
            phases, fracs_early_means[i], linestyle='dashed', color=CUSTOM_COLORSET[i]
        )
        ax2.fill_between(
            phases, fracs_full_means[i]-fracs_full_stddevs[i],
            fracs_full_means[i]+fracs_full_stddevs[i],
            alpha=0.2, color=CUSTOM_COLORSET[i]
        )
        
        ax3.plot(
            phases, f1_full_means[i], label=allowed_type, color=CUSTOM_COLORSET[i]
        )
        ax3.plot(
            phases, f1_early_means[i], linestyle='dashed', color=CUSTOM_COLORSET[i]
        )
        ax3.fill_between(
            phases, f1_full_means[i]-f1_full_stddevs[i],
            f1_full_means[i]+f1_full_stddevs[i],
            alpha=0.2, color=CUSTOM_COLORSET[i]
        )
        ax.set_ylim((0, 1))
        ax2.set_ylim((0, 1))
        ax3.set_ylim((0, 1))
        ax.axvline(x=0.0, color="grey", linestyle="dotted")
        ax2.axvline(x=0.0, color="grey", linestyle="dotted")
        ax3.axvline(x=0.0, color="grey", linestyle="dotted")

        ax.set_ylabel("Completeness")
        ax2.set_ylabel("Estimated Purity")    
        ax3.set_ylabel("Estimated F1")
        
    (dotted_l,) = axes[0,0].plot(np.NaN, np.NaN, color='k', linestyle='dashed', label='Early Phase')
    (solid_l,) = axes[0,0].plot(np.NaN, np.NaN, color='k', label='Full Phase')
    legend_lines.extend([dotted_l, solid_l])
    
    axes[4,0].set_xlabel(r"Phase (days)")
    axes[4,1].set_xlabel(r"Phase (days)")
    axes[4,2].set_xlabel(r"Phase (days)")
    
    fig.legend(legend_lines, [*[classes_to_labels[x] for x in allowed_types], 'Early Phase', 'Full Phase'], loc="lower center", ncol=4)
    plt.savefig(os.path.join(save_dir, "phase_vs_accuracy.pdf"), bbox_inches="tight")
    plt.close()
    
        
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 12), gridspec_kw = {'hspace':0.1})
    ax, ax2, ax3 = axes
    
    (legend_line,) = ax3.plot(
        phases, np.mean(f1_full_means, axis=0), color=CUSTOM_COLORSET[0]
    )
    (legend_line_dotted,) = ax3.plot(
        phases, np.mean(f1_early_means, axis=0), linestyle='dashed', color=CUSTOM_COLORSET[1]
    )
    ax3.fill_between(
        phases, np.mean(f1_full_means, axis=0) - np.mean(f1_full_stddevs, axis=0),
        np.mean(f1_full_means, axis=0) + np.mean(f1_full_stddevs, axis=0),
        alpha=0.2, color=CUSTOM_COLORSET[0]
    )
    
    ax.plot(
        phases, np.mean(accs_full_means, axis=0), color=CUSTOM_COLORSET[0]
    )
    ax.plot(
        phases, np.mean(accs_early_means, axis=0), linestyle='dashed', color=CUSTOM_COLORSET[1]
    )
    ax.fill_between(
        phases, np.mean(accs_full_means, axis=0) - np.mean(accs_full_stddevs, axis=0),
        np.mean(accs_full_means, axis=0) + np.mean(accs_full_stddevs, axis=0),
        alpha=0.2, color=CUSTOM_COLORSET[0]
    )
    
    ax2.plot(
        phases, np.mean(fracs_full_means, axis=0), color=CUSTOM_COLORSET[0]
    )
    ax2.plot(
        phases, np.mean(fracs_early_means, axis=0), linestyle='dashed', color=CUSTOM_COLORSET[1]
    )
    ax2.fill_between(
        phases, np.mean(fracs_full_means, axis=0) - np.mean(fracs_full_stddevs, axis=0),
        np.mean(fracs_full_means, axis=0) + np.mean(fracs_full_stddevs, axis=0),
        alpha=0.2, color=CUSTOM_COLORSET[0]
    )

    ax.set_ylabel("Completeness")
    ax.set_ylim((0, 1))
    
    ax2.set_ylabel("Estimated Purity")
    ax2.set_ylim((0, 1))
    
    ax3.set_ylabel("Estimated F1")
    ax3.set_ylim((0, 1))
        
    ax.axvline(x=0.0, color="grey", linestyle="dotted")
    ax2.axvline(x=0.0, color="grey", linestyle="dotted")
    ax3.axvline(x=0.0, color="grey", linestyle="dotted")
    ax3.set_xlabel(r"Phase (days)")
    fig.legend([legend_line, legend_line_dotted], ["Full Phase", "Early Phase"], loc="lower center", ncol=3)
    plt.savefig(os.path.join(save_dir, "phase_vs_accuracy_macro.pdf"), bbox_inches="tight")
    plt.close()

    def get_survey_fracs():
    """Return catalog with supernova fractions from existing
    catalogue datasets. referenced in papers.
    """
    yse_counts = np.array([314, 107, 15, 2, 32])
    yse_fracs = yse_counts / np.sum(yse_counts)

    psmds_counts = np.array([404, 94, 24, 17, 19])
    psmds_fracs = psmds_counts / np.sum(psmds_counts)

    return {"YSE": yse_fracs, "PS-MDS": psmds_fracs}


def retrieve_four_class_info(probs_csv, probs_alerce_csv, p07=False):
    """Extract Superphot+ and ALeRCE predictions and true class info."""
    _, classes_to_labels = SnClass.get_type_maps()

    (sn_names, true_classes, class_probs, pred_classes, folds, df) = read_probs_csv(probs_csv)

    secondary_pred_classes = np.argsort(class_probs, axis=1)[:,-2]
    
    if true_classes is None:
        true_classes = np.zeros(len(sn_names)) # filler
    if folds is None:
        folds = np.zeros(len(sn_names))
    try:
        true_labels = np.array([classes_to_labels[x] for x in true_classes])
    except:
        true_labels = np.array([SnClass.canonicalize(x) for x in true_classes])
    pred_labels = np.array([classes_to_labels[x] for x in pred_classes])
    pred_labels2 = np.array([classes_to_labels[x] for x in secondary_pred_classes])
    # read in ALeRCE classes
    df_alerce = pd.read_csv(probs_alerce_csv)
    merged_df = df.merge(df_alerce, left_on='Name', right_on='name', how='left')
    pred_alerce = merged_df.alerce_label.to_numpy().astype(str)

    ignore_mask = (pred_alerce == "None") | (pred_alerce == "nan") | (pred_alerce == "SKIP")
    # ignore true SNe IIn
    ignore_mask = ignore_mask | (true_labels == "SN IIn")

    (sn_names, true_labels, class_probs, pred_labels2, pred_labels, pred_alerce, folds) = (
        sn_names[~ignore_mask],
        true_labels[~ignore_mask],
        class_probs[~ignore_mask],
        pred_labels2[~ignore_mask],
        pred_labels[~ignore_mask],
        pred_alerce[~ignore_mask],
        folds[~ignore_mask]
    )
    print(np.unique(pred_labels2[pred_labels == "SN IIn"], return_counts=True))
    # merge SN IIn predictions with SN II
    pred_labels[pred_labels == "SN IIn"] = pred_labels2[pred_labels == "SN IIn"]

    
    if p07:
        p07_mask = np.max(class_probs, axis=1) > 0.7
        (sn_names, true_labels, class_probs, pred_labels, pred_alerce, folds) = (
            sn_names[p07_mask],
            true_labels[p07_mask],
            class_probs[p07_mask],
            pred_labels[p07_mask],
            pred_alerce[p07_mask],
            folds[p07_mask]
        )

    return (sn_names, true_labels, class_probs, pred_labels, pred_alerce, folds)



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
    psg = PosteriorSamplesGroup(all_post_objs, random_seed=42)
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
    
    print(aux_bands)
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
                0.01, 0.99, "Oversampling\nMultiple Fits\nper Light Curve",
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
        names, fits_dir, sampler=sampler,
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
    
    
"""
def plot_high_confidence_confusion_matrix(probs_csv, filename, cutoff=0.7, num_include=None):
    Plot confusion matrices for high-confidence predictions.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    filename : str
        Base filename for saving the confusion matrix plots.
    cutoff : float, optional
        Probability cutoff value for high-confidence predictions.
        Default is 0.7.
    _, classes_to_labels = SnClass.get_type_maps()

    _, true_classes, probs, pred_classes, folds, _ = read_probs_csv(probs_csv)
    
    if num_include is not None:
        confidences = np.max(probs, axis=1)
        conf_ordered = np.sort(confidences)
        print(conf_ordered)
        cutoff = conf_ordered[len(conf_ordered) - num_include - 1]
        
    high_conf_mask = np.max(probs, axis=1) > cutoff

    true_labels = [classes_to_labels[x] for x in true_classes[high_conf_mask]]
    pred_labels = [classes_to_labels[x] for x in pred_classes[high_conf_mask]]
    
    try:
        folds = folds[high_conf_mask]
    except:
        folds = None

    plot_confusion_matrix(true_labels, pred_labels, filename + "_c.pdf", folds, purity=False)
    plot_confusion_matrix(true_labels, pred_labels, filename + "_p.pdf", folds, purity=True)
"""

def plot_binary_confusion_matrix(probs_csv, filename, cutoff=0.5):
    """Merge all non-Ia into one core collapse class and plot resulting
    binary confusion matrix.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    filename : str
        Base filename for saving the confusion matrix plots.
    """
    df = pd.read_csv(probs_csv)
    true_classes = df.Label.to_numpy()
    prob_Ia = df.pSNIa.to_numpy()
    
    pred_binary = np.where(prob_Ia > cutoff, "SN Ia", "SN-Other")
    true_binary = np.where(true_classes == 0, "SN Ia", "SN-Other")
    try:
        folds = df.Fold.to_numpy()
    except:
        folds = None

    plot_confusion_matrix(true_binary, pred_binary, filename + "_c.pdf", folds, purity=False)
    plot_confusion_matrix(true_binary, pred_binary, filename + "_p.pdf", folds, purity=True)


def compare_four_class_confusion_matrices(probs_csv, probs_alerce_csv, save_dir, p07=False):
    """Plots ALeRCE's classifications as confusion matrix, and compare
    to condensed four-class CM of Superphot+.

    Only four classes as SNe IIn is not a label in their transient
    classifier.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing Superphot+ probability predictions.
    probs_alerce_csv : str
        Path to the CSV file containing ALeRCE predicted classes.
    save_dir : str
        Directory for saving the confusion matrix plots.
    p07 : bool, optional
        If True, only include predictions with a probability >= 0.7.
        Default is False.
    """
    (
        _, true_labels, _,
        pred_labels, pred_alerce, folds
    ) = retrieve_four_class_info(probs_csv, probs_alerce_csv, p07)

    plot_confusion_matrix(
        true_labels, pred_labels,
        os.path.join(save_dir, "superphot4_c.pdf"),
        folds=folds, purity=False, cmap='custom_cmap1',
    )
    plot_confusion_matrix(
        true_labels, pred_labels,
        os.path.join(save_dir, "superphot4_p.pdf"),
        folds=folds, purity=True, cmap='custom_cmap1',
    )

    plot_confusion_matrix(
        true_labels, pred_alerce,
        os.path.join(save_dir, "alerce_c.pdf"),
        folds=folds, purity=False, cmap='custom_cmap2',
    )
    plot_confusion_matrix(
        true_labels, pred_alerce,
        os.path.join(save_dir, "alerce_p.pdf"),
        folds=folds, purity=True, cmap='custom_cmap2',
    )


def plot_true_agreement_matrix(probs_csv, probs_alerce_csv, save_dir, spec=True):
    """Plot agreement matrix between ALeRCE and Superphot+
    classifications.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    probs_alerce_csv : str
        Path to the CSV containing ALeRCE predictions.
    save_dir : str
        Directory path for saving the agreement matrix plot.
    """
    pred_labels, pred_alerce, folds = retrieve_four_class_info(
        probs_csv,
        probs_alerce_csv,
        False,
    )[3:6]

    plot_agreement_matrix_from_arrs(pred_labels, pred_alerce, folds, save_dir, spec=spec)


def plot_expected_agreement_matrix(probs_csv, probs_alerce_csv, save_dir, cmap="custom_cmap2"):
    """Plot expected agreement matrix based on independent ALeRCE and
    Superphot+ confusion matrices.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing probability predictions.
    save_dir : str
        Directory for saving the expected agreement matrix plot.
    cmap : matplotlib.colors.Colormap, optional
        Color map for the plot. Default is plt.cm.Purples.
    """
    (_, true_labels, _, pred_labels, alerce_preds, folds) = retrieve_four_class_info(
        probs_csv, probs_alerce_csv
    )

    accs = []
    cm_vals_all = []
    alerce_preds = np.array(alerce_preds)
    classes = unique_labels(alerce_preds, pred_labels)

    for f in np.unique(folds):
        ap_fold = alerce_preds[folds == f]
        cm_purity = confusion_matrix(
            true_labels[folds == f],
            pred_labels[folds == f],
            normalize="pred"
        )
        cm_complete = confusion_matrix(
            true_labels[folds == f],
            ap_fold,
            normalize="true"
        )
        cm_expected = cm_purity.T @ cm_complete

        exp_acc = 0
        # calculate agreement score
        for i, single_class in enumerate(classes):
            num_in_class = len(ap_fold[ap_fold == single_class])
            exp_acc += num_in_class * cm_expected[i, i]

        accs.append(exp_acc / len(ap_fold))
        cm_vals_all.append(cm_expected)
        
    cm_vals_all = np.asarray(cm_vals_all)
    cm_expected = np.median(cm_vals_all, axis=0)
    cm_low = np.abs(cm_expected - np.percentile(cm_vals_all, 10, axis=0))
    cm_high = np.abs(np.percentile(cm_vals_all, 90, axis=0) - cm_expected)

    acc = np.median(accs)
    acc_low = acc - np.percentile(accs, 10)
    acc_high = np.percentile(accs, 90) - acc

    title = f"Expected Agreement Matrix,\nSpec. ($A' = {acc:.2f}^{{+{acc_high:.2f}}}_{{-{acc_low:.2f}}}$)"
    fig, axis = plt.subplots(figsize=(6,6))
    _ = axis.imshow(cm_expected, interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap)

    axis.set(
        xticks=np.arange(cm_expected.shape[1]),
        yticks=np.arange(cm_expected.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="ALeRCE Classification",
        xlabel="Superphot+ Classification",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"
    thresh = cm_expected.max() / 1.5
    for i in range(cm_expected.shape[0]):
        for j in range(cm_expected.shape[1]):
            axis.text(
                j,
                i,
                f"${cm_expected[i, j]:.2f}^{{+{cm_high[i, j]:.2f}}}_{{-{cm_low[i, j]:.2f}}}$",
                ha="center",
                va="center",
                color="white" if cm_expected[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)
    plt.savefig(
        os.path.join(save_dir, "expected_agreement.pdf"),
        bbox_inches='tight',
    )
    plt.close()


def plot_agreement_matrix_from_arrs(our_labels, alerce_labels, folds, save_dir, spec=True, cmap="custom_cmap2"):
    """Helper function to plot agreement matrices.

    Plot agreement matrix based on input arrays of ALeRCE and Superphot+
    classifications.

    Parameters
    ----------
    our_labels : list
        List of our predicted labels.
    alerce_labels : list
        List of ALeRCE predicted labels.
    filename : str
        Base filename for saving the agreement matrix plot.
    cmap : matplotlib.colors.Colormap, optional
        Color map for the plot. Default is plt.cm.Purples.
    """
    if spec:
        suffix_title = "Spec."
        suffix = "spec"
    else:
        suffix_title = "Phot."
        suffix = "phot"

    accs = []
    cm_vals_all = []
    classes = unique_labels(alerce_labels, our_labels)
    our_labels = np.array(our_labels)
    alerce_labels = np.array(alerce_labels)
    
    for f in np.unique(folds):
        cm_vals_all.append(
            confusion_matrix(
                alerce_labels[folds == f],
                our_labels[folds == f], normalize="true"
            )
        )
        
        accs.append(
            calc_accuracy(
                alerce_labels[folds == f],
                our_labels[folds == f]
            )
        )

    cm_vals_all = np.asarray(cm_vals_all)
    cm = np.median(cm_vals_all, axis=0)
    cm_low = np.abs(cm - np.percentile(cm_vals_all, 10, axis=0))
    cm_high = np.abs(np.percentile(cm_vals_all, 90, axis=0) - cm)

    acc = np.median(accs)
    acc_low = acc - np.percentile(accs, 10)
    acc_high = np.percentile(accs, 90) - acc
        
    title = "True Agreement Matrix,\n" + fr"{suffix_title} ($A' = {acc:.2f}^{{+{acc_high:.2f}}}_{{-{acc_low:.2f}}}$)"
        
    fig, ax = plt.subplots(figsize=(6,6))
    _ = ax.imshow(cm, interpolation="nearest", vmin=0.0, vmax=1.0, cmap=cmap)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="ALeRCE Classification",
        xlabel="Superphot+ Classification",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 1.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            class_i = classes[i]
            class_j = classes[j]
            num_in_cell = len(our_labels[(our_labels == class_j) & (alerce_labels == class_i)])
            #if spec:
            ax.text(
                j,
                i,
                f"${cm[i, j]:.2f}^{{+{cm_high[i, j]:.2f}}}_{{-{cm_low[i, j]:.2f}}}$" + f"\n({num_in_cell})",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
            #else:
            #    ax.text(
            #    j,
            #    i,
            #    f"${cm[i, j]:.2f}$" + f"\n({num_in_cell})",
            #    ha="center",
            #    va="center",
            #    color="white" if cm[i, j] > thresh else "black",
            #)
                
    fig.tight_layout()
    plt.xlim(-0.5, len(classes) - 0.5)
    plt.ylim(len(classes) - 0.5, -0.5)

    plt.savefig(
        os.path.join(save_dir, f"true_agreement_{suffix}.pdf"),
        bbox_inches='tight'
    )
    plt.close()
