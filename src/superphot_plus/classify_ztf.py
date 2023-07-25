"""This module provides functions to classify supernovae using a
multi-layer perceptron (MLP).

The classification is based on the fit parameters and light curves of
the supernovae."""

import csv
import os
import shutil

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from .constants import MEANS_TRAINED_MODEL, NUM_FOLDS, STDDEVS_TRAINED_MODEL, PAD_SIZE
from .file_paths import *  # pylint: disable=wildcard-import
from .format_data_ztf import (
    generate_K_fold,
    get_posterior_samples,
    import_labels_only,
    normalize_features,
    oversample_using_posteriors,
    tally_each_class,
)
from .lightcurve import Lightcurve
from .mlp import (
    MLP,
    create_dataset,
    get_predictions_new,
    run_mlp,
    save_test_probabilities,
    save_unclassified_test_probabilities,
)
from .plotting import plot_confusion_matrix
from .supernova_class import SupernovaClass as SnClass
from .utils import calc_accuracy, calculate_neg_chi_squareds, f1_score
from .ztf_transient_fit import run_mcmc


def adjust_log_dists(features):
    """Takes log of fit parameters with log-Gaussian priors before
    feeding into classifier. Also removes apparent amplitude and t0.

    Parameters
    ----------
    features : np.ndarray
        Array of fit features of all samples.
    """
    features[:, 4:7] = np.log10(features[:, 4:7])
    features[:, 2] = np.log10(features[:, 2])
    return np.delete(features, [0, 3], 1)


def classify(goal_per_class, num_epochs, neurons_per_layer, num_layers, fits_plotted=False):
    """Train MLP to classify between supernovae of 'allowed_types'.

    Parameters
    ----------
    goal_per_class : int
        Oversampling such that there are this many fits per supernova
        type.
    num_epochs : int
        Number of training epochs.
    neurons_per_layer : int
        Number of neurons per hidden layer of MLP.
    num_layers : int
        Number of hidden layers in MLP.
    fits_plotted : bool
        If true, assumes all sample fit plots are saved in
        FIT_PLOTS_FOLDER. Copies plots of wrongly classified samples to
        separate folder for manual followup. Defaults to False.
    """

    # for file in os.scandir('models'):
    #    os.remove(file.path)
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc", "SLSN-II"]
    output_dim = len(allowed_types)  # number of classes

    labels_to_classes, classes_to_labels = SnClass.get_type_maps(allowed_types)

    fn_prefix = "cm_%d_%d_%d_%d" % (goal_per_class, num_epochs, neurons_per_layer, num_layers)
    fn_purity = os.path.join(CM_FOLDER, fn_prefix + "_p.pdf")
    fn_completeness = os.path.join(CM_FOLDER, fn_prefix + "_c.pdf")
    fn_purity_07 = os.path.join(CM_FOLDER, fn_prefix + "_p_p07.pdf")
    fn_completeness_07 = os.path.join(CM_FOLDER, fn_prefix + "_c_p07.pdf")

    names, labels = import_labels_only(input_csvs, allowed_types)

    tally_each_class(labels)  # original tallies

    kfold = generate_K_fold(np.zeros(len(labels)), labels, NUM_FOLDS)

    true_classes_mlp = np.array([])
    predicted_classes_mlp = np.array([])
    prob_above_07_mlp = np.array([], dtype=bool)

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

        train_chis = calculate_neg_chi_squareds(train_names, FITS_DIR, DATA_DIRS)
        train_features, train_classes, train_chis = oversample_using_posteriors(
            train_names, train_classes, train_chis, goal_per_class
        )
        val_chis = calculate_neg_chi_squareds(val_names, FITS_DIR, DATA_DIRS)
        val_features, val_classes, val_chis = oversample_using_posteriors(
            val_names, val_classes, val_chis, round(0.1 * goal_per_class)
        )

        train_features = np.append(
            train_features,
            np.array(
                [
                    train_chis,
                ]
            ).T,
            1,
        )
        val_features = np.append(
            val_features,
            np.array(
                [
                    val_chis,
                ]
            ).T,
            1,
        )

        test_features = []
        test_classes_os = []
        test_group_idxs = []
        test_names_os = []
        test_chis_os = []
        test_chis = calculate_neg_chi_squareds(test_names, FITS_DIR, DATA_DIRS)

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
                start_idx = test_group_idxs[-1][-1] + 1
            test_group_idxs.append(np.arange(start_idx, start_idx + len(test_posts)))

        test_features = np.array(test_features)
        test_chis = np.array(
            [
                test_chis_os,
            ]
        )

        test_features = np.append(test_features, test_chis.T, 1)

        # normalize the log distributions
        test_features = adjust_log_dists(test_features)
        test_classes = np.array(test_classes_os)
        test_names = np.array(test_names_os)

        # print(test_names[0])
        train_features = adjust_log_dists(train_features)
        val_features = adjust_log_dists(val_features)
        train_features, mean, std = normalize_features(train_features)
        val_features, mean, std = normalize_features(val_features, mean, std)
        test_features, mean, std = normalize_features(test_features, mean, std)

        # Convert to Torch DataSet objects
        train_data = create_dataset(train_features, train_classes)
        val_data = create_dataset(val_features, val_classes)
        # test_data = create_dataset(test_features, test_classes)

        # Train and evaluate multi-layer perceptron
        test_classes, test_names, pred_classes, pred_probs, valid_loss = run_mlp(
            train_data,
            val_data,
            test_features,
            test_classes,
            test_names,
            test_group_idxs,
            output_dim,
            neurons_per_layer,
            num_layers,
            num_epochs,
        )

        return pred_classes, pred_probs > 0.7, test_classes, test_names, valid_loss

    r = Parallel(n_jobs=-1)(delayed(run_single_fold)(x) for x in kfold)
    (
        predicted_classes_mlp,
        prob_above_07_mlp,
        true_classes_mlp,
        ztf_test_names,
        valid_loss_mlp,
    ) = zip(*r)

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
            fn = wc + ".png"
            fn_new = wc + "_" + wc_type + "_" + wrong_type + ".png"
            shutil.copy(
                os.path.join(FIT_PLOTS_FOLDER, fn),
                os.path.join(WRONGLY_CLASSIFIED_FOLDER, wc_type + "/" + fn_new),
            )

    with open(CLASSIFY_LOG_FILE, "a+") as the_file:
        the_file.write(str(goal_per_class) + " samples per class\n")
        the_file.write(str(neurons_per_layer) + " neurons per each of " + str(num_layers) + " layers\n")
        the_file.write(str(num_epochs) + " epochs\n")
        the_file.write(
            "HOW MANY CERTAIN "
            + str(len(true_classes_mlp))
            + " "
            + str(len(true_classes_mlp[prob_above_07_mlp]))
            + "\n"
        )
        the_file.write(
            "MLP class-averaged F1-score: %.04f\n"
            % f1_score(predicted_classes_mlp, true_classes_mlp, class_average=True)
        )
        the_file.write("Accuracy: %.04f\n" % calc_accuracy(predicted_classes_mlp, true_classes_mlp))
        the_file.write("Validation Loss: %.04f\n\n" % valid_loss_avg)

    # Plot full and p > 0.7 confusion matrices
    plot_confusion_matrix(true_classes_mlp, predicted_classes_mlp, fn_purity, True)
    plot_confusion_matrix(true_classes_mlp, predicted_classes_mlp, fn_completeness, False)

    plot_confusion_matrix(
        true_classes_mlp[prob_above_07_mlp],
        predicted_classes_mlp[prob_above_07_mlp],
        fn_purity_07,
        True,
    )
    plot_confusion_matrix(
        true_classes_mlp[prob_above_07_mlp],
        predicted_classes_mlp[prob_above_07_mlp],
        fn_completeness_07,
        False,
    )


def load_mlp(mlp_filename, mlp_params):
    """Load a trained MLP for subsequent classification of new objects.
    
    Parameters
    ----------
    mlp_filename : str
        Where the trained MLP is stored.
    mlp_params : tuple or array
        Includes (in order): input_size, output_size, n_neurons, n_hidden.
        
    Returns
    ----------
    torch.nn.Module
        The pre-trained MLP object.
    """
    model = MLP(*mlp_params)  # set up empty multi-layer perceptron
    model.load_state_dict(torch.load(mlp_filename))  # load trained state dict to the MLP
    return model
    
    
def classify_from_fit_params(fit_params):
    """Classify one or multiple light curves
    solely from the fit parameters used in the
    classifier. Excludes t0 and, for redshift-
    exclusive classifier, A. Includes chi-squared
    value.
    
    Parameters
    ----------
    fit_params : np.ndarray
        Set of model fit parameters.
        
    Returns
    ----------
    np.ndarray
        Probability of each light curve being each SN type. Sums to 1 along each row.
    """
    fit_params_2d = np.atleast_2d(fit_params_2d) # cast to 2D if only 1 light curve
        
    test_features, means, stds = normalize_features(  # pylint: disable=unused-variable
        fit_params_2d, MEANS_TRAINED_MODEL, STDDEVS_TRAINED_MODEL
    )
    test_data = torch.utils.data.TensorDataset(torch.Tensor(test_features))
    test_iterator = torch.utils.data.DataLoader(test_data, batch_size=32)
    images, probs = get_predictions_new(
        model, test_iterator, "cpu"
    )  # pylint: disable=unused-variable
    return probs
        

def classify_single_light_curve(obj_name, fits_dir, data_dirs):
    """Given an object name, return classification probabilities
    based on the model fit and data.
    
    Parameters
    ----------
    obj_name : str
        Name of the supernova.
    fits_dir : str
        Where model fit information is stored.
    data_dirs : np.ndarray
        Where the object's datafile could be stored.
        
    Returns
    ----------
    np.ndarray
        The average probability for each SN type across all equally-weighted sets of fit parameters.
    """
    try:
        post_features = get_posterior_samples(obj_name, fits_dir)
    except:
        print("no posts")
        return
    
    chisq = calculate_neg_chi_squareds(
        [
            obj_name,
        ]
        fits_dir,
        data_dirs,
    )[0]
    
    if np.abs(chisq) > 10: # probably not a SN
        print("OBJECT LIKELY NOT A SN")
        
    chisq_arr = np.array([chisq] * len(post_features))
    # concat chisq vals to end of input features
    post_features = np.append(post_features, chisq_arr.T, 1) 

    # normalize the log distributions
    test_features = adjust_log_dists(post_features)
    probs = classify_from_fit_params(post_features)
    probs_avg = np.mean(probs.numpy(), axis=0)
    return probs_avg
    

def return_new_classifications(test_csv, data_dirs, fit_dir, include_labels=False):
    """Return new classifications based on model and save probabilities
    to a CSV file.

    Parameters
    ----------
    test_csv : str
        Path to the CSV file containing the test data.
    data_dirs : list of str
        List of paths to directories containing data.
    fit_dir : str
        Path to the directory containing the fit data.
    include_labels : bool, optional
        If True, labels from the test data are included in the
        probability saving process. Defaults to False.
    """
    model = load_mlp(TRAINED_MODEL_FN, TRAINED_MODEL_PARAMS)
    with open(test_csv, "r") as tc:
        csv_reader = csv.reader(tc, delimiter=",")
        next(csv_reader)
        for _, row in enumerate(csv_reader):
            
            try:
                test_name = row[0]
            except:
                print(row, "skipped")
                continue
                
            if include_labels:
                label = row[1]

            probs_avg = classify_single_light_curve(test_name, fit_dir, data_dirs)

            if include_labels:
                save_test_probabilities(test_name, label, probs_avg)
            else:
                save_unclassified_test_probabilities(test_name, probs_avg)


def save_phase_versus_class_probs(probs_csv, data_dir):
    """Apply classifier to dataset over different phases. Plot overall
    trends of phase vs confidence, phase vs F1 score, phase vs each
    class accuracy.
    
    Note this was being manually altered for different desired plots.
    Future versions will move all that to function args.

    Parameters
    ----------
    probs_csv : str
        Path to the CSV file containing the test probabilities.
    data_dir : str
        Path to the directory containing the data.
    """
    model = load_mlp(TRAINED_MODEL_FN, TRAINED_MODEL_PARAMS)

    ct = 0

    t_cutoffs = np.arange(-18, 54, 4)
    with open(probs_csv, "r") as tc:
        csv_reader = csv.reader(tc, delimiter=",")
        next(csv_reader)
        for _, row in enumerate(csv_reader):
            if ct >= 60:
                break
            test_name = row[0]
            label = row[1]
            if int(label[-2]) != 4:
                continue

            ct += 1

            try:
                lc = Lightcurve.from_file(os.path.join(data_dir, test_name + ".npz"))
                lc.pad_bands(["g", "r"], PAD_SIZE)
                tarr = lc.times
                farr = lc.fluxes
            except:
                print("skipping import")
                continue

            mean_t0 = tarr[np.argmax(farr)]

            def single_loop(phase):
                t = phase + float(mean_t0)
                print(phase)
                if phase > 50.0:
                    return None

                try:
                    refit_posts = run_mcmc(os.path.join(data_dir, test_name + ".npz"), t)
                    test_chi = calculate_neg_chi_squareds(
                        [
                            test_name,
                        ],
                        FITS_DIR,
                        [
                            data_dir,
                        ],
                    )[0]
                    test_chis = np.array([[test_chi] * len(refit_posts)])
                except:
                    print("skipping fitting")
                    return None

                test_features = np.append(refit_posts, test_chis.T, 1)

                # normalize the log distributions
                test_features = adjust_log_dists(test_features)
                probs = classify_from_fit_params(test_features)
                probs_avg = np.mean(probs.numpy(), axis=0)
                # idx_random = np.random.choice(np.arange(len(probs)))
                save_test_probabilities(str(label), round(phase, 2), probs_avg)

            Parallel(n_jobs=-1)(delayed(single_loop)(float(x)) for x in t_cutoffs)
