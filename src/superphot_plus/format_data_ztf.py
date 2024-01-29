"""This script provides functions for importing, preprocessing, and
manipulating data related to ZTF lightcurves."""

import csv
import pandas as pd

import numpy as np

from superphot_plus.file_utils import get_multiple_posterior_samples, has_posterior_samples
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.posterior_samples import PosteriorSamples


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
    
    labels = []
    labels_orig = []
    repeat_ct = 0
    names = []
    redshifts = []
    
    for input_csv in input_csvs:
        df = pd.read_csv(input_csv)
        names_all = df.NAME.to_numpy()
        labels_all = df.CLASS.to_numpy()
        redshifts_all = df.Z.to_numpy()
        
        for i, name in enumerate(names_all):
            if needs_posteriors and (
                    fits_dir is None or not has_posterior_samples(
                    lc_name=name, fits_dir=fits_dir, sampler=sampler
                )
            ):
                continue
                
            label_orig = labels_all[i]
            row_label = SnClass.canonicalize(label_orig)

            if row_label not in allowed_types:
                continue

            if name not in names:
                names.append(name)
                labels.append(row_label)
                labels_orig.append(label_orig)
                redshifts.append(float(redshifts_all[i]))
            else:
                repeat_ct += 1

    tally_each_class(labels_orig)
    print(repeat_ct)

    return np.array(names), np.array(labels), np.array(redshifts)


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


def retrieve_posterior_set(
    lc_names, fits_dir, sampler=None,
    redshifts=None, labels=None,
    chisq_cutoff=np.inf,
):
    """Retrieve all sets of posterior samples, excluding
    poor median fits and invalid redshift values.
    
    Parameters
    ----------
    lc_names : str
        Lightcurve names.
    fits_dir : str
        Where fit parameters are stored.
    sampler : str, optional
        The name of the sampler to use.
    redshifts : list, optional
        List of redshift values.
    chisq_cutoff : float, optional
        Ignore all fit sets with median chisq above this value.
    """
    samples = []
    if redshifts is None:
        redshifts = np.ones(len(lc_names))

    for i, name in enumerate(lc_names):
        if np.isnan(redshifts[i]) or redshifts[i] <= 0:
            continue
        try:
            post_obj = PosteriorSamples.from_file(
                name=name,
                input_dir=fits_dir,
                sampling_method=sampler
            )
        except:
            continue
        # bandaid: add redshifts to PosteriorSamples object here
        post_obj.redshift = redshifts[i]
        if labels is not None:
            post_obj.sn_class = labels[i]
        all_posts = post_obj.samples
        
        if np.median(all_posts[:, -1]) > chisq_cutoff:
            continue
        
        samples.append(post_obj)

    return np.array(samples)


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
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)

    safe_std = np.copy(std)
    safe_std[std == 0.0] = 1.0
    return (features - mean) / safe_std, mean, std
