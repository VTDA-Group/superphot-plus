import numpy as np
from format_data_ztf import *
from plotting import *
from file_paths import *
from mlp import *
import shutil
from joblib import Parallel, delayed
from ztf_transient_fit import run_mcmc, import_data


def flux_model(cube, t_data, b_data):
    """
    Given "cube" of fit parameters, returns the flux measurements for
    a given set of time and band data.
    """
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


def calc_accuracy(pred_classes, test_labels):
    """
    Calculates the accuracy of the random forest after predicting
    all classes.
    
    Parameters
    ----------
    pred_classes : numpy array (int)
        classes predicted by MLP
    test_labels : numpy array (int)
        true spectroscopic classes
    """
    num_total = len(pred_classes)
    num_correct = np.sum(np.where(pred_classes == test_labels, 1, 0))
    return num_correct/num_total


def f1_score(pred_classes, true_classes, class_average=False):
    """
    Calculates the F1 score for the classifier. If class_average=True, then
    the macro-F1 is used. Else, uses the weighted-F1 score.
    
    Parameters
    ----------
    pred_classes : numpy array (int)
        classes predicted by MLP
    true_classes : numpy array (int)
        true spectroscopic classes
    class_average : bool
        Determines whether F1 score is weighted equally for each class,
        or by number of samples per class. Defaults to False.
    """
    samples_per_class = {}
    for c in true_classes:
        if c in samples_per_class:
            samples_per_class[c] += 1
        else:
            samples_per_class[c] = 1

    f1_sum = 0.
    for c in samples_per_class:
        tp = len(pred_classes[(pred_classes == c) & (true_classes == c)])
        purity = tp / len(pred_classes[pred_classes == c])
        completeness = tp / len(true_classes[true_classes == c])
        f1 = 2. * purity * completeness / (purity + completeness)
        if class_average:
            f1_sum += f1
        else:
            f1_sum += samples_per_class[c] * f1
    if class_average:
        return f1_sum / len(samples_per_class.keys())

    total_samples = np.sum(samples_per_class.values())
    return f1_sum / total_samples


def adjust_log_dists(features):
    """
    Takes log of fit parameters with log-Gaussian priors before
    feeding into classifier. Also removes apparent amplitude and t0.
    
    Parameters
    ----------
    features : numpy array
        Array of fit features of all samples.
    """
    features[:, 4:7] = np.log10(features[:, 4:7])
    features[:, 2] = np.log10(features[:, 2])
    return np.delete(features, [0,3], 1)


def classify(goal_per_class, num_epochs, neurons_per_layer, num_layers, fits_plotted=False):
    """
    Train MLP to classify between supernovae of 'allowed_types'.
    
    Parameters
    ----------
    goal_per_class : int
        Oversampling such that there are this many fits per supernova type.
    num_epochs : int
        Number of training epochs.
    neurons_per_layer : int
        Number of neurons per hidden layer of MLP.
    num_layers : int
        Number of hidden layers in MLP.
    fits_plotted : bool
        If true, assumes all sample fit plots are saved in FIT_PLOTS_FOLDER. Copies
        plots of wrongly classified samples to separate folder for manual followup.
        Defaults to False.
    """
    
    #for file in os.scandir('models'):
    #    os.remove(file.path)
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc", "SLSN-II"]
    output_dim = len(allowed_types) # number of classes
    
    labels_to_classes = {allowed_types[i]: i for i in range(len(allowed_types))}
    classes_to_labels = {i: allowed_types[i] for i in range(len(allowed_types))}

    num_folds = 20
    fn_prefix = "cm_%d_%d_%d_%d" % (goal_per_class, num_epochs, neurons_per_layer, num_layers)
    fn_purity = os.path.join(CM_FOLDER, fn_prefix + "_p.pdf") 
    fn_completeness = os.path.join(CM_FOLDER, fn_prefix + "_c.pdf") 
    fn_purity_07 = os.path.join(CM_FOLDER, fn_prefix + "_p_p07.pdf") 
    fn_completeness_07 = os.path.join(CM_FOLDER, fn_prefix + "_c_p07.pdf")

    names, labels = import_labels_only(input_csvs, allowed_types)
    
    tally_each_class(labels) # original tallies

    kfold = generate_K_fold(np.zeros(len(labels)), labels, num_folds)

    true_classes_mlp = np.array([])
    predicted_classes_mlp = np.array([])
    prob_above_07_mlp = np.array([], dtype=bool)
    #ct = 0
    
    #for MLP
    input_dim = 13
        
    def run_single_fold(x):
        train_index, test_index = x
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        test_names = np.array(names[test_index])
        
        train_index, val_index = train_test_split(train_index, stratify=train_labels, test_size=0.1)
        
        train_names = names[train_index]
        val_names = names[val_index]
        
        train_labels = labels[train_index]
        val_labels = labels[val_index]
        
        train_classes = np.array([labels_to_classes[l] for l in train_labels]).astype(int)
        val_classes = np.array([labels_to_classes[l] for l in val_labels]).astype(int)
        test_classes = np.array([labels_to_classes[l] for l in test_labels]).astype(int)
        
        train_chis = calculate_chi_squareds(train_names, FITS_DIR, DATA_DIRS)
        train_features, train_classes, train_chis = oversample_using_posteriors(train_names, train_classes, train_chis, goal_per_class)
        
        val_chis = calculate_chi_squareds(val_names, FITS_DIR, DATA_DIRS)
        val_features, val_classes, val_chis = oversample_using_posteriors(val_names, val_classes, val_chis, round(0.1 * goal_per_class))
        
        train_features = np.append(train_features, np.array([train_chis,]).T, 1)
        val_features = np.append(val_features, np.array([val_chis,]).T, 1)
        
        test_features = []
        test_classes_os = []
        test_group_idxs = []
        test_names_os = []
        test_chis_os = []
        test_chis = calculate_chi_squareds(test_names, FITS_DIR, DATA_DIRS)
        
        for i in range(len(test_names)):
            test_name = test_names[i]
            test_posts = get_posterior_samples(test_name)
            test_features.extend(test_posts)
            test_classes_os.extend([test_classes[i]] * len(test_posts))
            test_names_os.extend([test_names[i]] * len(test_posts))
            test_chis_os.extend([test_chis[i]] * len(test_posts))
            if len(test_group_idxs) == 0:
                start_idx = 0
            else:
                start_idx = test_group_idxs[-1][-1]+1
            test_group_idxs.append(np.arange(start_idx, start_idx+len(test_posts)))
            
        test_features = np.array(test_features)
        test_chis = np.array([test_chis_os,])
        
        test_features = np.append(test_features, test_chis.T, 1)
        
        # normalize the log distributions
        test_features = adjust_log_dists(test_features)
        test_classes = np.array(test_classes_os)
        test_names = np.array(test_names_os)

        #print(test_names[0])
        train_features = adjust_log_dists(train_features)
        val_features = adjust_log_dists(val_features)
        train_features, mean, std = normalize_features(train_features)
        val_features, mean, std = normalize_features(val_features, mean, std)
        test_features, mean, std = normalize_features(test_features, mean, std)

        #Convert to Torch DataSet objects
        train_data = create_dataset(train_features, train_classes)
        val_data = create_dataset(val_features, val_classes)
        #test_data = create_dataset(test_features, test_classes)
        
        # Train and evaluate multi-layer perceptron
        test_classes, test_names, pred_classes, pred_probs, valid_loss = run_mlp(train_data, val_data, test_features, test_classes, \
                                                                     test_names, test_group_idxs, input_dim, output_dim, \
                                                                    neurons_per_layer, num_layers, num_epochs)
            
        return pred_classes, pred_probs > 0.7, test_classes, test_names, valid_loss
    
   
    r = Parallel(n_jobs=-1)(delayed(run_single_fold)(x) for x in kfold)
    predicted_classes_mlp, prob_above_07_mlp, true_classes_mlp, ztf_test_names, valid_loss_mlp = zip(*r)
    predicted_classes_mlp = np.hstack(tuple(predicted_classes_mlp))
    prob_above_07_mlp = np.hstack(tuple(prob_above_07_mlp))
    true_classes_mlp = np.hstack(tuple(true_classes_mlp))
    ztf_test_names = np.hstack(tuple(ztf_test_names))
    valid_loss_avg = np.mean(valid_loss_mlp)

    true_classes_mlp = np.array([classes_to_labels[l] for l in true_classes_mlp])
    predicted_classes_mlp = np.array([classes_to_labels[l] for l in predicted_classes_mlp])

    if fits_plotted:
        wrongly_classified = np.where(true_classes_mlp != predicted_classes_mlp)[0]
        for wc_idx in wrongly_classified:
            wc = ztf_test_names[wc_idx]
            wc_type = true_classes_mlp[wc_idx]
            wrong_type = predicted_classes_mlp[wc_idx]
            fn = wc+".png"
            fn_new = wc+"_"+wc_type+"_"+wrong_type+".png"
            shutil.copy(os.path.join(FIT_PLOTS_FOLDER, fn), os.path.join(WRONGLY_CLASSIFIED_FOLDER, wc_type+"/"+fn_new))

    
    with open(CLASSIFY_LOG_FILE, 'a+') as the_file:          
        the_file.write(str(goal_per_class)+ " samples per class\n")
        the_file.write(str(neurons_per_layer)+ " neurons per each of "+ str(num_layers)+ " layers\n")
        the_file.write(str(num_epochs) + " epochs\n")
        the_file.write("HOW MANY CERTAIN "+str(len(true_classes_mlp)) + " " + str(len(true_classes_mlp[prob_above_07_mlp]))+"\n")
        the_file.write("MLP class-averaged F1-score: %.04f\n" % f1_score(predicted_classes_mlp, true_classes_mlp, class_average=True))
        the_file.write("Accuracy: %.04f\n" % calc_accuracy(predicted_classes_mlp, true_classes_mlp))
        the_file.write("Validation Loss: %.04f\n\n" % valid_loss_avg)
        
    # Plot full and p > 0.7 confusion matrices
    plot_confusion_matrix(true_classes_mlp, predicted_classes_mlp, fn_purity, True)
    plot_confusion_matrix(true_classes_mlp, predicted_classes_mlp, fn_completeness, False)

    plot_confusion_matrix(true_classes_mlp[prob_above_07_mlp], predicted_classes_mlp[prob_above_07_mlp], fn_purity_07, True)
    plot_confusion_matrix(true_classes_mlp[prob_above_07_mlp], predicted_classes_mlp[prob_above_07_mlp], fn_completeness_07, False)
    

def return_new_classifications(test_csv, data_dirs, fit_dir, include_labels=False):
    """
    Return new classifications based on model, save probs to save_Csv.
    """
    model = MLP(13, 5, 128, 3) # set up empty multi-layer perceptron
    model.load_state_dict(torch.load(glob.glob("models_saved/*ZTF23aagkgnz.pt")[0])) # load trained state dict to the MLP
    means = np.array([5.18928404e-03, 1.27960653e+00, 7.16663998e-01, 1.59438422e+00,
                      -1.45215889e+00, 9.54225774e-01, 1.04234314e+00, 1.00733255e+00,
                      9.99977873e-01, 9.66049977e-01, 5.62890503e-01, 8.63290836e-01,
                      -6.27014662e+00])
    
    stds = np.array([5.00371052e-04, 3.17145223e-01, 3.93625733e-01, 2.58445333e-01,
                     3.72614525e-01, 2.15892037e-01, 2.44030784e-03, 1.27667116e-02,
                     5.11363683e-05, 1.18290613e-02, 6.24454144e-02, 4.99345471e-02,
                     8.98741848e-01])
    
    classes_to_labels = {0: "SN Ia", 1: "SN II", 2: "SN IIn", 3: "SLSN-I", 4: "SN Ibc"} #converts the MLP classes to types
    labels_to_classes = {"SN Ia": 0, "SN II": 1, "SN IIn": 2, "SLSN-I":3, "SN Ibc": 4}
    
    special_labels = {"SN Iax[02cx-like]",}
    test_features = []
    test_classes_os = []
    test_group_idxs = []
    test_names_os = []
    test_chis_os = []
    with open(test_csv, "r") as tc:
        csv_reader = csv.reader(tc, delimiter=",")
        next(csv_reader)
        for e, row in enumerate(csv_reader):
            try:
                test_name = row[0]
            except:
                print(row)
                continue
            if include_labels:
                label = row[1]
                if label not in special_labels: # to classify special types
                    continue
            try:
                print(test_name, fit_dir)
                test_posts = get_posterior_samples(test_name, fit_dir)
            except:
                print("no posts")
                continue
            test_features = test_posts
            test_names = np.array([test_name] * len(test_posts))
            test_chi = calculate_chi_squareds([test_name,], fit_dir, data_dirs)[0]
            #if np.abs(test_chi) > 10: # probably not a SN
            #    print(test_name, "CHISQ TOO HIGH")
            #    label = "SKIP"
            test_chis = np.array([[test_chi] * len(test_posts)])
            
            test_features = np.append(test_features, test_chis.T, 1)

            # normalize the log distributions
            test_features = adjust_log_dists(test_features)
            test_features, means, stds = normalize_features(test_features, means, stds)
            test_data = TensorDataset(torch.Tensor(test_features))
            test_iterator = data.DataLoader(test_data,
                                        batch_size=32)
            images, probs = get_predictions_new(model, test_iterator, 'cpu')
            probs_avg = np.mean(probs.numpy(), axis=0)
            if include_labels:
                save_test_probabilities(test_names[0], label, probs_avg)
            else:
                save_unclassified_test_probabilities(test_names[0], probs_avg)

                
def save_phase_versus_class_probs(probs_csv, data_dir):
    """
    Apply classifier to dataset over different phases, plot overall trends of phase vs confidence,
    phase vs F1 score, phase vs each class accuracy.
    """
    model = MLP(13, 5, 128, 3) # set up empty multi-layer perceptron
    model.load_state_dict(torch.load(glob.glob("models_saved/*ZTF23aagkgnz.pt")[0])) # load trained state dict to the MLP
    means = np.array([5.18928404e-03, 1.27960653e+00, 7.16663998e-01, 1.59438422e+00,
                      -1.45215889e+00, 9.54225774e-01, 1.04234314e+00, 1.00733255e+00,
                      9.99977873e-01, 9.66049977e-01, 5.62890503e-01, 8.63290836e-01,
                      -6.27014662e+00])
    
    stds = np.array([5.00371052e-04, 3.17145223e-01, 3.93625733e-01, 2.58445333e-01,
                     3.72614525e-01, 2.15892037e-01, 2.44030784e-03, 1.27667116e-02,
                     5.11363683e-05, 1.18290613e-02, 6.24454144e-02, 4.99345471e-02,
                     8.98741848e-01])
    
    classes_to_labels = {0: "SN Ia", 1: "SN II", 2: "SN IIn", 3: "SLSN-I", 4: "SN Ibc"} #converts the MLP classes to types
    labels_to_classes = {"SN Ia": 0, "SN II": 1, "SN IIn": 2, "SLSN-I":3, "SN Ibc": 4}
    
    ct = 0
    
    t_cutoffs = np.arange(-18, 54, 4)
    with open(probs_csv, "r") as tc:
        csv_reader = csv.reader(tc, delimiter=",")
        next(csv_reader)
        for e, row in enumerate(csv_reader):
            if ct >= 60:
                break
            test_name = row[0]
            label = row[1]
            if int(label[-2]) != 4:
                continue
                
            ct += 1
            phases = []
            try:
                tarr, farr, _, _ = import_data(os.path.join(data_dir, test_name+".npz"))
            except:
                print("skipping import")
                continue
            
           
            mean_t0 = tarr[np.argmax(farr)]
            def single_loop(phase):
                t = phase + float(mean_t0)
                print(phase)
                if phase > 50.:
                    return None

                try:
                    refit_posts = run_mcmc(os.path.join(data_dir, test_name+".npz"), t)
                    test_chi = calculate_chi_squareds([test_name,], FITS_DIR, [data_dir,])[0]
                    test_chis = np.array([[test_chi] * len(refit_posts)])
                except:
                    print("skipping fitting")
                    return None
                
                test_features = np.append(refit_posts, test_chis.T, 1)

                # normalize the log distributions
                test_features = adjust_log_dists(test_features)
                test_features, means2, stds2 = normalize_features(test_features, means, stds)
                test_data = TensorDataset(torch.Tensor(test_features))
                test_iterator = data.DataLoader(test_data,
                                            batch_size=32)
                images, probs = get_predictions_new(model, test_iterator, 'cpu')
                probs_avg = np.mean(probs.numpy(), axis=0)
                #idx_random = np.random.choice(np.arange(len(probs)))
                save_test_probabilities(str(label), round(phase, 2), probs_avg)
            
            Parallel(n_jobs=-1)(delayed(single_loop)(float(x)) for x in t_cutoffs)
            
                
def main():
    classify(4000, 200, 128, 3)
    
if __name__ == "__main__":
    main()
        
        

