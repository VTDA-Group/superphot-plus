"""This script provides functions for importing, preprocessing, and
manipulating data related to ZTF lightcurves."""

import csv

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

from superphot_plus.file_paths import FITS_DIR
from superphot_plus.file_utils import get_multiple_posterior_samples, has_posterior_samples
from superphot_plus.supernova_class import SupernovaClass as SnClass


def import_labels_only(input_csvs, allowed_types, fits_dir=None, needs_posteriors=True, sampler=None):
    """Filters CSVs for rows where label is in allowed_types and returns
    names, labels.

    Parameters
    ----------
    input_csvs : list of str
        List of input CSV file paths.
    allowed_types : list
        List of allowed types for labels.
    fits_dir : str, optional
        Directory path for FITS files. Defaults to None.
    needs_posteriors: boolean, optional
        Indicates whether to load posterior samples.
    sampler : str, optional
        The sampler to get posteriors from.

    Returns
    -------
    tuple of np.ndarray
        Tuple of names, labels and redshifts.

    Notes
    -----
    Maps groups of similar labels to a single representative label name
    (eg, "SN Ic", "SNIc-BL", and "21" all become "SN Ibc").
    """
    if fits_dir is None:
        fits_dir = FITS_DIR

    labels = []
    labels_orig = []
    repeat_ct = 0
    names = []
    redshifts = []

    print(input_csvs, fits_dir, sampler)
    for input_csv in input_csvs:
        with open(input_csv, newline="", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                name = row[0]
                if needs_posteriors and not has_posterior_samples(
                    lc_name=name, fits_dir=fits_dir, sampler=sampler
                ):
                    continue
                label_orig = row[1]
                row_label = SnClass.canonicalize(label_orig)

                if row_label not in allowed_types:
                    continue

                if name not in names:
                    names.append(name)
                    labels.append(row_label)
                    labels_orig.append(label_orig)
                    redshifts.append(float(row[2]))
                else:
                    repeat_ct += 1

    tally_each_class(labels_orig)
    print(repeat_ct)

    return np.array(names), np.array(labels), np.array(redshifts)


def generate_K_fold(features, classes, num_folds):
    """Generates set of K test sets and corresponding training sets.

    Parameters
    ----------
    features: list
        Input features.
    classes: list
        Input classes.
    num_folds : int
        Number of folds. If -1, sets num_folds=len(features).

    Returns
    -------
    generator
        Generator yielding the indices for training and test sets.
    """
    if num_folds == -1:
        kf = StratifiedKFold(n_splits=len(features), shuffle=True)  # cross-one out validation
    else:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    return kf.split(features, classes)


def tally_each_class(labels):
    """Prints the number of samples with each class label.

    Parameters
    ----------
    labels: list
        Input labels.
    """
    tally_dict = {}
    for label in labels:
        if label not in tally_dict:
            tally_dict[label] = 1
        else:
            tally_dict[label] += 1
    for tally_label, count in tally_dict.items():
        print(f"{tally_label}: {count}")
    print()


def oversample_using_posteriors(
    lc_names, labels, goal_per_class, fits_dir, sampler=None, redshifts=None, oversample_redshifts=False
):
    """Oversamples, drawing from posteriors of a certain fit.

    Parameters
    ----------
    lc_names : str
        Lightcurve names.
    labels : list
        List of labels.
    goal_per_class : int
        Number of samples per class.
    fits_dir : str
        Where fit parameters are stored.
    sampler : str, optional
        The name of the sampler to use.
    redshifts : list, optional
        List of redshift values.
    oversample_redshifts : boolean, optional
        Indicates whether to oversample redshifts.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing oversampled features, labels, and redshifts.
    """

    oversampled_redshifts = []
    oversampled_labels = []
    oversampled_features = []
    labels_unique = np.unique(labels)

    labels = np.array(labels)

    posterior_samples = get_multiple_posterior_samples(lc_names, fits_dir, sampler)

    for l in labels_unique:
        idxs_in_class = np.asarray(labels == l).nonzero()[0]
        num_in_class = len(idxs_in_class)
        samples_per_fit = max(1, np.round(goal_per_class / num_in_class).astype(int))
        for i in idxs_in_class:
            lc_name = lc_names[i]
            all_posts = posterior_samples[lc_name]
            sampled_idx = np.random.choice(np.arange(len(all_posts)), samples_per_fit)
            sampled_features = all_posts[sampled_idx]
            oversampled_features.extend(list(sampled_features))
            oversampled_labels.extend([l] * samples_per_fit)
            if oversample_redshifts:
                oversampled_redshifts.extend([redshifts[i]] * samples_per_fit)

    return np.array(oversampled_features), np.array(oversampled_labels), np.array(oversampled_redshifts)


def normalize_features(features, mean=None, std=None):
    """Normalizes the features for feeding into the neural network.

    Parameters
    ----------
    features : numpy array
        Input features. Must be a 2-d array where each row corresponds
        to a data point and each entry to a feature.
    mean : ndarray, optional
        Mean values for normalization. Defaults to None.
    std : ndarray, optional
        Standard deviation values for normalization. Defaults to None.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing normalized features, mean values, and standard
        deviation values.
    """
    if mean is None:
        mean = features.mean(axis=-2)
    if std is None:
        std = features.std(axis=-2)

    safe_std = np.copy(std)
    safe_std[std == 0.0] = 1.0
    return (features - mean) / safe_std, mean, std


def oversample_smote(features, labels):
    """
    Uses SMOTE to oversample data from rarer classes.
    """
    oversample = SMOTE()
    features_smote, labels_smote = oversample.fit_resample(features, labels)
    return features_smote, labels_smote
