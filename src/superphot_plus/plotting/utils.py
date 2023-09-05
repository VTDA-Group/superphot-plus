"""This module contains helper functions to access/manipulate data for plotting more concisely."""

import colorsys
import os

import matplotlib.colors as mc
import numpy as np
import pandas as pd
from alerce.core import Alerce

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.supernova_class import SupernovaClass as SnClass


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    try:
        color_val = mc.cnames[color]
    except:
        color_val = color
    color_hls = colorsys.rgb_to_hls(*mc.to_rgb(color_val))
    return colorsys.hls_to_rgb(color_hls[0], max(0, min(1, amount * color_hls[1])), color_hls[2])


def get_survey_fracs():
    """Return catalog with supernova fractions from existing
    catalogue datasets. referenced in papers.
    """
    yse_counts = np.array([314, 107, 15, 2, 32])
    yse_fracs = yse_counts / np.sum(yse_counts)

    psmds_counts = np.array([404, 94, 24, 17, 19])
    psmds_fracs = psmds_counts / np.sum(psmds_counts)

    return {"YSE": yse_fracs, "PS-MDS": psmds_fracs}


def read_probs_csv(probs_fn):
    """Helper function to read in a probability csv file
    and return the columns as numpy arrays.
    """
    df = pd.read_csv(probs_fn)
    names = df.Name.to_numpy()
    labels = df.Label.to_numpy()
    probs = df.iloc[:, 2:7].astype(float).to_numpy()
    pred_classes = np.argmax(probs, axis=1)

    return names, labels, probs, pred_classes, df


def get_alerce_pred_class(ztf_name, alerce, superphot_style=False):
    """Get alerce probabilities corresponding to the four (no SN IIn)
    classes in our ZTF classifier.

    Parameters
    ----------
    ztf_name : str
        ZTF name of the object.
    superphot_style : bool, optional
        If True, change format of output labels. Default is False.

    Returns
    -------
    str
        Predicted class label.
    """
    try:
        query = alerce.query_probabilities(oid=ztf_name, format="pandas")
        query_transient = query[query["classifier_name"] == "lc_classifier_transient"]
        label = query_transient[query_transient["ranking"] == 1]["class_name"].iat[0]
        return SnClass.from_alerce_to_superphot_format(label) if superphot_style else label
    except:
        return "None"


def create_alerce_pred_csv(probs_fn, save_fn):
    """Generate csv with ALeRCE's predicted
    classes for SN names in probs_csv.
    """
    alerce_obj = Alerce()

    names = read_probs_csv(probs_fn)[0]

    alerce_labels = []
    for sn_name in names:
        alerce_labels.append(get_alerce_pred_class(sn_name, alerce_obj, superphot_style=True))

    df = pd.DataFrame({"name": names, "alerce_label": alerce_labels})

    df.to_csv(save_fn, index=False)


def retrieve_four_class_info(probs_csv, probs_alerce_csv, p07=False):
    """Extract Superphot+ and ALeRCE predictions and true class info."""
    _, classes_to_labels = SnClass.get_type_maps()

    (sn_names, true_classes, class_probs, pred_classes, _) = read_probs_csv(probs_csv)

    try:
        true_labels = np.array([classes_to_labels[x] for x in true_classes])
    except:
        true_labels = np.array([SnClass.canonicalize(x) for x in true_classes])
    pred_labels = np.array([classes_to_labels[x] for x in pred_classes])

    # read in ALeRCE classes
    df_alerce = pd.read_csv(probs_alerce_csv)
    pred_alerce = df_alerce.alerce_label.to_numpy().astype(str)

    ignore_mask = (pred_alerce == "None") | (pred_alerce == "nan") | (pred_alerce == "SKIP")
    # ignore true SNe IIn
    ignore_mask = ignore_mask | (true_labels == "SN IIn")

    (sn_names, true_labels, class_probs, pred_labels, pred_alerce) = (
        sn_names[~ignore_mask],
        true_labels[~ignore_mask],
        class_probs[~ignore_mask],
        pred_labels[~ignore_mask],
        pred_alerce[~ignore_mask],
    )

    # merge SN IIn predictions with SN II
    pred_labels[pred_labels == "SN IIn"] = "SN II"

    if p07:
        p07_mask = np.max(class_probs, axis=1) > 0.7
        (sn_names, true_labels, class_probs, pred_labels, pred_alerce) = (
            sn_names[p07_mask],
            true_labels[p07_mask],
            class_probs[p07_mask],
            pred_labels[p07_mask],
            pred_alerce[p07_mask],
        )

    return (sn_names, true_labels, class_probs, pred_labels, pred_alerce)


def gaussian(inputs, amp, mean, sigma):
    """Evaluate a gaussian with params A at the values in x.

    Parameters
    ----------
    inputs : array-like or float
        Value(s) to evaluate gaussian at
    amp : float
        Amplitude of the Gaussian.
    mean : float
        Mean of Gaussian
    sigma : float
        Standard deviation of Gaussian

    Returns
    ----------
    array-like or float:
        Gaussian values evaluated at x
    """
    if ~np.isscalar(inputs):
        inputs = np.array(inputs)
    return amp * np.exp(-((inputs - mean) ** 2) / sigma**2 / 2.0)


def histedges_equalN(vals, nbin):
    """Generate histogram bin edges, such that counts are equal in each bin.

    Parameters
    ----------
    vals : array-like or float
        Value(s) to bin in histogram
    nbin : integer
        number of bins
    """
    npt = len(vals)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(vals))


def add_snr_to_prob_csv(probs_csv, data_dir, new_csv):
    """
    Adds 10% SNR and num of SNR > 5 points columns
    to probability CSV. Useful for plots.
    """
    names, _, _, _, probs_df = read_probs_csv(probs_csv)

    extended_df = probs_df.copy()

    n_snr_3 = []
    n_snr_5 = []
    n_snr_10 = []
    snr_ten_percent = []
    max_flux = []

    for name in names:
        try:
            filename = os.path.join(data_dir, name + ".npz")
            lightcurve = Lightcurve.from_file(filename)
            snr = np.abs(lightcurve.fluxes / lightcurve.flux_errors)
            n_snr_3.append(len(snr[(snr > 3.0)]))
            n_snr_5.append(len(snr[(snr > 5.0)]))
            n_snr_10.append(len(snr[(snr > 10.0)]))
            snr_ten_percent.append(np.quantile(snr, 0.9))
            max_flux.append(lightcurve.find_max_flux(band="r")[0])
        except:
            n_snr_3.append(-1)
            n_snr_5.append(-1)
            n_snr_10.append(-1)
            snr_ten_percent.append(-1)
            max_flux.append(-1)

    extended_df["Fmax"] = np.array(max_flux)
    extended_df["SNR90"] = np.array(snr_ten_percent)
    extended_df["nSNR3"] = np.array(n_snr_3)
    extended_df["nSNR5"] = np.array(n_snr_5)
    extended_df["nSNR10"] = np.array(n_snr_10)

    extended_df.to_csv(new_csv, index=False)
