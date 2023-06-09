import numpy as np
import corner
import torch
import csv, os
import sklearn
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import glob
#from sklearn.cross_validation import train_test_split


def import_labels_only(input_csvs, allowed_types, fits_dir=None, redshift=False):
    """
    Import all features and labels, convert to label and features
    numpy arrays.
    """

    if fits_dir is None:
        fits_dir = FITS_DIR
    labels = []
    labels_orig = []
    repeat_ct = 0
    names = []
    redshifts = []
    sn1bc_alts = ["SN Ic", "SN Ib", "SN Ic-BL", "SN Ib-Ca-rich", "SN Ib/c", \
                 "SNIb", "SNIc", "SNIc-BL", "21", "20", "27", "26", "25"]
    snIIn_alts = ["SNIIn", "35"]
    snIa_alts = ["SN Ia-91T-like", "SN Ia-CSM", "SN Ia-91bg-like", "SNIa", "SN Ia-91T", "SN Ia-91bg", "10", "11", "12"]
    snII_alts = ["SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"]
    slsnI_alts = ["40", "SLSN",]
    slsnII_alts = ["SLSN-II",]
    tde_alts = ["42",]
    for input_csv in input_csvs:
        with open(input_csv, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                name = row[0]
                if not os.path.isfile(fits_dir+name+"_eqwt.npz"):
                    continue
                label_orig = row[1]
                l = row[1]
                z = float(row[2])
                if redshift and z <= 0.:
                    print(name, l)
                    continue
                if l in sn1bc_alts:
                    l = "SN Ibc"
                elif l in snIIn_alts:
                    l = "SN IIn"
                elif l in snIa_alts:
                    l = "SN Ia"
                elif l in snII_alts:
                    l = "SN II"
                elif l in slsnI_alts:
                    l = "SLSN-I"
                elif l in slsnI_alts:
                    l = "SLSN-II"
                elif l in tde_alts:
                    l = "TDE"
                if l not in allowed_types:
                    #print(l)
                    continue
                if name not in names:
                    names.append(name)
                    labels.append(l)
                    labels_orig.append(label_orig)
                    if redshift:
                        redshifts.append(z)
                else:
                    repeat_ct += 1

    tally_each_class(labels_orig)
    print(repeat_ct)
    if redshift:
        return np.array(names), np.array(labels), np.array(redshifts)
    return np.array(names), np.array(labels)
    

def import_features_and_labels(input_csv, allowed_types):
    """
    Import all features and labels, convert to label and features
    numpy arrays.
    """
    feature_means = []
    feature_stddevs = []
    labels = []
    names = []
    sn1bc_alts = ["SN Ic", "SN Ib", "SN Ic-BL", "SN Ib-Ca-rich", "SN Ib/c"]
    snIIn_alts = ["SLSN-II"]
    snIa_alts = ["SN Ia-91 T-like", "SN Ia-CSM", "SN Ia-91bg-like"]
    snII_alts = ["SN IIP", "SN IIL"]
    with open(input_csv, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            l = row[1]
            if l in sn1bc_alts:
                l = "SN Ibc"
            elif l in snIIn_alts:
                l = "SN IIn"
            elif l in snIa_alts:
                l = "SN Ia"
            elif l in snII_alts:
                l = "SN II"
            if l not in allowed_types:
                continue
            names.append(row[0])
            feature_means.append(row[2:16])
            feature_stddevs.append(row[16:])
            labels.append(l)

    return np.array(names), np.array(feature_x).astype(float), np.array(feature_stddevs).astype(float), np.array(labels)

def return_names_from_med_arrays(input_csv, med_arr):
    
    names = [""] * len(med_arr)
    
    t_0_expected = med_arr[3]
    best_diff = np.inf
    best_features = None
    best_match = None
    ct = 0
    for fn in glob.glob(FITS_DIR+"/*.npz"):
        try:
            name = fn.split("/")[-1].split("_")[0]
            #print(name)
            features = get_posterior_samples(name, output_dir=None)
            med_features = np.median(features, axis=0)
            t_0 = med_features[3]
            diff = np.abs(t_0_expected - t_0)
            if diff < best_diff:
                best_diff = diff
                best_match = name
                best_features = med_features
            ct += 1
        except:
            pass
    print(ct)
    print(best_match,best_features)
            

def divide_into_training_test_set(features, labels, test_fraction):
    """
    Divides dataset into set fraction of test samples and remaining as
    training data.
    """
    return train_test_split(features, labels, test_size=test_fraction, random_state=42)

def generate_K_fold(features, classes, num_folds):
    """
    Generates set of K test sets and corresponding training sets
    """
    if num_folds == -1:
        kf = StratifiedKFold(n_splits=len(features), shuffle=True) # cross-one out validation
    else:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    return kf.split(features, classes)

def tally_each_class(labels):
    """
    Print number of samples with each class label.
    """
    tally_dict = {}
    for label in labels:
        if label not in tally_dict:
            tally_dict[label] = 1
        else:
            tally_dict[label] += 1
    for tally_label in tally_dict:
        print(tally_label,": ", str(tally_dict[tally_label]))
    print()

def generate_two_class_labels(labels):
    """
    For the binary classification problem.
    """
    labels_copy = np.copy(labels)
    labels_copy[labels_copy != "SN Ia"] = "other"
    return labels_copy

def oversample_minority_classes(features, labels):
    """
    Uses SMOTE to oversample data from rarer classes so
    classifiers are not biased toward SN-1a or SN-II
    """
    oversample = SMOTE()
    features_smote, labels_smote = oversample.fit_resample(features, labels)
    return features_smote, labels_smote

def oversample_using_posteriors_old(feature_means, feature_stddevs, labels, goal_per_class):
    """
    Oversample each class to num_per_class by drawing multiple feature sets from each fit's
    posterior space.
    """
    oversampled_labels = []
    oversampled_features = []
    labels_unique = np.unique(labels)
    for l in labels_unique:
        idxs_in_class = np.asarray(labels == l).nonzero()[0]
        num_in_class = len(idxs_in_class)
        samples_per_fit = max(1, np.round(goal_per_class / num_in_class).astype(int))
        print(l, samples_per_fit)
        for i in idxs_in_class:
            for j in range(samples_per_fit):
                feature_set = np.random.normal(loc=np.array(feature_means[i]), scale=np.array(feature_stddevs[i]))
                while np.any(feature_set <= 0):
                    feature_set = np.random.normal(loc=np.array(feature_means[i]), scale=np.array(feature_stddevs[i]))
                oversampled_features.append(feature_set)
                oversampled_labels.append(l)
    oversampled_features = np.array(oversampled_features)
    oversampled_labels = np.array(oversampled_labels)
    print(oversampled_labels)
    return oversampled_features, oversampled_labels

def get_posterior_samples(ztf_name, output_dir=None):
    """
    Get all EQUAL WEIGHT posterior samples from
    a ZTF lightcurve fit.
    """
    if output_dir is None:
        output_dir = FITS_DIR
    post_fn = os.path.join(output_dir, ztf_name + "_eqwt.npz")
    #output_dir = "../outputs/"
    #post_fn = output_dir + ztf_name +"/" + ztf_name + "post_equal_weights.dat"
    """
    with open(post_fn, "r") as post_ew:
        post_rows = post_ew.read().split("\n")
        post_arr = []
        for row in post_rows[:-1]:
            post_arr.append([float(x) for x in row.split()])
        post_arr = np.array(post_arr)[:,:-1] # exclude the loglikelihoods
    """
    npy_array = np.load(post_fn)
    post_arr = npy_array['arr_0']
    return post_arr

def oversample_using_posteriors(ztf_names, labels, chis, goal_per_class):
    """
    Draws from posteriors of a certain fit.
    """
    oversampled_labels = []
    oversampled_chis = []
    oversampled_features = []
    labels_unique = np.unique(labels)
    for l in labels_unique:
        idxs_in_class = np.asarray(labels == l).nonzero()[0]
        num_in_class = len(idxs_in_class)
        samples_per_fit = max(1, np.round(goal_per_class / num_in_class).astype(int))
        for i in idxs_in_class:
            ztf_name = ztf_names[i]
            all_posts = get_posterior_samples(ztf_name)
            sampled_idx = np.random.choice(np.arange(len(all_posts)), samples_per_fit)
            sampled_features = all_posts[sampled_idx]
            oversampled_features.extend(list(sampled_features))
            oversampled_labels.extend([l] * samples_per_fit)
            oversampled_chis.extend([chis[i]] * samples_per_fit)
    return np.array(oversampled_features), np.array(oversampled_labels), np.array(oversampled_chis)
        
def normalize_features(features, mean=None, std=None):
    """
    Normalize the features for feeding into the neural network.
    """
    if mean is None:
        mean = features.mean(axis=-2)
    if std is None:
        std = features.std(axis=-2)
    
    print(mean, std)
    return (features - mean) / std, mean, std

def corner_plot_all(input_csvs, save_file):
    """
    Plot combined corner plot of all training set samples,
    excluding the overall scaling A.
    """
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    names, labels = import_labels_only(input_csvs, allowed_types)
    print(labels)
    chis = np.ones(len(names))
    features, labels, chis = oversample_using_posteriors(names, labels, chis, 4000)
    print(labels)
    figure = corner.corner(
        np.delete(features, [0,3], axis=1),
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
            r"$\sigma_{ex,g}$"
        ],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 20},
        color="purple"
    )
    # Extract the axes
    axes = np.array(figure.axes)
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
    
    figure.savefig(save_file)
    
def summarize_misc_classification(misc_csv):
    """
    Summarize how miscellaneous types of transients are classified.
    """
    misc_dict = {}
    with open(misc_csv, "r") as mc:
        csv_reader = csv.reader(mc, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            obj_type = row[1]
            probs = np.array(row[2:]).astype(float)
            best_type = np.argmax(probs)
            if obj_type not in misc_dict:
                misc_dict[obj_type] = [0,0,0,0,0]
            misc_dict[obj_type][best_type] += 1
                
    print(misc_dict)
    
def generate_csv_subset(orig_sn_name, new_sn_name, sn_idx, p_cutoff):
    """
    Generate smaller subset with only SNe of one type,
    with confidence above certain threshhold.
    """
    sn_names = []
    with open(orig_sn_name, "r") as orig:
        csv_reader = csv.reader(orig, delimiter=",")
        for row in csv_reader:
            if float(row[sn_idx]) > p_cutoff:
                sn_names.append(row[0])
    
    with open(new_sn_name, "a+") as new:
        csv_writer = csv.writer(new, delimiter=",")
        for sn in sn_names:
            csv_writer.writerow([sn, -1])

def flux_model(cube, t_data, b_data):    
    A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7]

    phase = t_data - t0    
    f_model = A / (1. + np.exp(-phase / tau_rise)) * (1. - beta * gamma) * np.exp((gamma - phase) / tau_fall)
    f_model[phase < gamma] = A / (1. + np.exp(-phase[phase < gamma] / tau_rise)) * (1. - beta * phase[phase < gamma])

    # for secondary band
    start_idx = 7
    A_b = A * cube[start_idx]
    beta_b = beta * cube[start_idx + 1]
    gamma_b = gamma * cube[start_idx + 2]
    t0_b = t0 * cube[start_idx + 3]
    tau_rise_b = tau_rise * cube[start_idx + 4]
    tau_fall_b = tau_fall * cube[start_idx + 5]
    
    inc_band_ix = (np.array(b_data) == "g")
    phase_b = (t_data - t0_b)[inc_band_ix]
    phase_b2 = (t_data - t0_b)[inc_band_ix & (t_data - t0_b < gamma_b)]

    f_model[inc_band_ix] = A_b / (1. + np.exp(-phase_b / tau_rise_b)) \
        * (1. - beta_b * gamma_b) * np.exp((gamma_b - phase_b) / tau_fall_b)
    f_model[inc_band_ix & (t_data - t0_b < gamma_b)] = A_b / (1. + np.exp(-phase_b2 / tau_rise_b)) \
        * (1. - phase_b2 * beta_b)
    return f_model

def calculate_chi_squareds(names, fit_dir, data_dirs):
    """
    Gets the chi-squared of posterior fits from
    the model parameters and original datafiles.
    """
    log_likelihoods = []
    for e, name in enumerate(names):
        data_fn = None
        for d in data_dirs: 
            data_fn = os.path.join(d, name + ".npz")
            if os.path.exists(data_fn):
                break

        npy_array = np.load(data_fn)
        mjd, flux, flux_err, bands = npy_array['arr_0']
        
        flux_err = flux_err.astype(float)
        mjd = mjd.astype(float)[~np.isnan(flux_err)]
        flux = flux.astype(float)[~np.isnan(flux_err)]
        bands = bands[~np.isnan(flux_err)]
        flux_err = flux_err[~np.isnan(flux_err)]
        
        fit_fn = os.path.join(fit_dir, name +"_eqwt.npz")
        npy_array_fit = np.load(fit_fn)
        post_arr = npy_array_fit['arr_0']
        
        post_med = np.median(post_arr, axis=0)
        #print(post_med)

        model_f = flux_model(post_med, mjd, bands)
        extra_sigma_arr = np.ones(len(mjd)) * np.max(flux[bands == "r"]) * post_med[6]
        extra_sigma_arr[bands == "g"] *= post_med[-1]
        sigma_sq = extra_sigma_arr**2 + flux_err**2
        
        logL = np.sum(np.log(1. / np.sqrt(2. * np.pi * sigma_sq)) - 0.5 * (flux - model_f)**2 / sigma_sq) / len(mjd)
        log_likelihoods.append(logL)
            
    return np.array(log_likelihoods)

def generate_csv_subset2(orig_sn_names, new_sn_name, sn_type):
    """
    Generate smaller subset with only SNe of one type,
    from spectroscopic set.
    """
    with open(new_sn_name, "w+") as new:
        new.write("")
            
    FITS_DIR = "/gpfs/group/vav5084/default/kdesoto/ztf/dynesty_fits_11_2022/"
    DATA_DIRS = ["/gpfs/group/vav5084/default/kdesoto/ztf/data_reformatted_11_2022/", "/gpfs/group/vav5084/default/kdesoto/ztf/data_reformatted_classified_ztf_bts"]
    
    alts = {"SN II": ["SN II", "SNII", "SN IIP", "SN IIL", "SNII", "SNIIP", "32", "30", "31"]}
    sn_names = []
    zs = []
    for orig_sn_name in orig_sn_names:
        with open(orig_sn_name, "r") as orig:
            csv_reader = csv.reader(orig, delimiter=",")
            for row in csv_reader:
                if row[1] in alts["SN II"]:
                    if row[0] in sn_names:
                        continue
                    if not os.path.exists(FITS_DIR + row[0] + "_eqwt.npz"):
                        continue
                    sn_names.append(row[0])
                    zs.append(float(row[2]))
                    

    
    train_chis = calculate_chi_squareds(sn_names, FITS_DIR, DATA_DIRS)
    print(len(train_chis), len(sn_names))
    for e, sn_name in enumerate(sn_names):
        train_features, train_classes, train_chis_os = oversample_using_posteriors([sn_name,],[2,], [train_chis[e],], 100)
        med_features = np.median(train_features, axis=0)
        med_features = np.append(med_features, np.median(train_chis_os))
        
        with open(new_sn_name, "a") as new:
            csv_writer = csv.writer(new, delimiter=",")
            #csv_writer.writerow(["Name","Redshift"])
            csv_writer.writerow([sn_name, zs[e], *med_features])

def add_snr_to_prob_csv(probs_csv, new_csv):
    """
    Adds 10% SNR and num of SNR > 5 points columns
    to probability CSV. Useful for plots.
    """
    all_rows = []
    with open(probs_csv, "r") as csvfile:
        with open(new_csv, 'w+') as csvoutput:
            csvreader = csv.reader(csvfile)
            csvwriter = csv.writer(csvoutput)
            for row in csvreader:
                name = row[0]
                for data_dir in DATA_DIRS:
                    try:
                        #data_fn = glob.glob(data_dir + "/*/" + name + ".npz")[0]
                        data_fn = data_dir + "/" + name + ".npz"
                        npy_array = np.load(data_fn)
                        #print(npy_array)
                    except:
                        pass
                    
                arr = npy_array['arr_0']

                ferr = arr[2]
                f = arr[1][ferr != "nan"].astype(float)
                b = arr[3][ferr != "nan"]
                ferr = ferr[ferr != "nan"].astype(float)
                snr = np.abs(f / ferr)

                n_snr_3 = len(snr[(snr > 3.)])
                n_snr_5 = len(snr[(snr > 5.)])
                n_snr_10 = len(snr[(snr > 10.)])
                snr_ten_percent = np.quantile(snr, 0.9)
                max_r_flux = np.max(f[b == "r"])
                row.append(max_r_flux)
                row.append(snr_ten_percent)
                row.append(n_snr_3)
                row.append(n_snr_5)
                row.append(n_snr_10)
                all_rows.append(row)
            csvwriter.writerows(all_rows)
            
def main():
    pass

if __name__ == "__main__":
    main()