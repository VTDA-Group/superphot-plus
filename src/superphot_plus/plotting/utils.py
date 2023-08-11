import numpy as np
from alerce.core import Alerce
import pandas as pd
import os

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.lightcurve import Lightcurve


def get_survey_fracs():
    """Return catalog with supernova fractions from existing
    catalogue datasets. referenced in papers.
    """
    yse_counts = np.array([314, 107, 15, 2, 32])
    yse_fracs = yse_counts / np.sum(yse_counts)

    psmds_counts = np.array([404, 94, 24, 17, 19])
    psmds_fracs = psmds_counts / np.sum(psmds_counts)
    
    return {'YSE': yse_fracs, 'PS-MDS': psmds_fracs}


def read_probs_csv(probs_fn, return_dataframe=False):
    """Helper function to read in a probability csv file
    and return the columns as numpy arrays.
    """
    df = pd.read_csv(probs_fn)
    names = df.Name.to_numpy()
    labels = df.Label.to_numpy()
    probs = df.iloc[:, 2:7].astype(float).to_numpy()
    pred_classes = np.argmax(probs, axis=1)

    if return_dataframe:
        return names, labels, probs, pred_classes, df

    return names, labels, probs, pred_classes


def get_alerce_pred_class(ztf_name, superphot_style=False):
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
    alerce = Alerce()
    o = alerce.query_probabilities(oid=ztf_name, format="pandas")
    o_transient = o[o["classifier_name"] == "lc_classifier_transient"]
    label = o_transient[o_transient["ranking"] == 1]["class_name"].iat[0]
    return SnClass.from_alerce_to_superphot_format(label) if superphot_style else label


def gaussian(x, A, mu, sigma):
    """Evaluate a gaussian with params A at the values in x.

    Parameters
    ----------
    x : array-like or float
        Value(s) to evaluate gaussian at
    A : float
        Amplitude of the Gaussian.
    mu : float
        Mean of Gaussian
    sigma : float
        Standard deviation of Gaussian

    Returns
    ----------
    array-like or float:
        Gaussian values evaluated at x
    """
    if ~np.isscalar(x):
        x = np.array(x)
    return A * np.exp(-((x - mu) ** 2) / sigma**2 / 2.0)


def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))


def add_snr_to_prob_csv(probs_csv, data_dir, new_csv):
    """
    Adds 10% SNR and num of SNR > 5 points columns
    to probability CSV. Useful for plots.
    """
    names, labels, probs, pred_classes, df = read_probs_csv(probs_csv, return_dataframe=True)

    extended_df = df.copy()

    n_snr_3 = []
    n_snr_5 = []
    n_snr_10 = []
    snr_ten_percent = []
    max_flux = []

    for name in names:
        try:
            fn = os.path.join(data_dir, name + ".npz")
            lc = Lightcurve.from_file(fn)
            snr = np.abs(lc.fluxes / lc.flux_errors)
            n_snr_3.append(len(snr[(snr > 3.0)]))
            n_snr_5.append(len(snr[(snr > 5.0)]))
            n_snr_10.append(len(snr[(snr > 10.0)]))
            snr_ten_percent.append(np.quantile(snr, 0.9))
            max_flux.append(lc.find_max_flux(band="r")[0])
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
