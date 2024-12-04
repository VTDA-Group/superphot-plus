"""This module contains helper functions to access/manipulate data for plotting more concisely."""

import colorsys
import os
from functools import partial

import matplotlib.colors as mc
import numpy as np
import pandas as pd
from alerce.core import Alerce
from scipy.stats import binned_statistic
from snapi import Transient, TransientGroup

from superphot_plus.supernova_class import SupernovaClass as SnClass


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


def n_obs_snr(transient: Transient, min_snr: float):
    """Calculate number of observations with SNR >= min_snr"""
    phot = transient.photometry
    snr = np.abs(phot.fluxes / phot.flux_errors)
    return len(snr[snr >= min_snr])

def snr_90(transient: Transient):
    """Calculate 90th percentile SNR"""
    phot = transient.photometry
    snr = np.abs(phot.fluxes / phot.flux_errors)
    return np.nanquantile(snr, 0.9)
    
def add_snr_cols(transient_data: TransientGroup):
    """
    Adds 10% SNR and num of SNR > N points columns
    to transient_data. Useful for plots.
    """
    transient_data.add_col(
        "snr_90", snr_90
    )
    transient_data.add_col(
        "n_obs_snr3", partial(n_obs_snr, min_snr=3.0)
    )
    transient_data.add_col(
        "n_obs_snr5", partial(n_obs_snr, min_snr=5.0)
    )
    transient_data.add_col(
        "n_obs_snr10", partial(n_obs_snr, min_snr=10.0)
    )
    return transient_data

def get_alerce_pred_class(transient: Transient, alerce, superphot_style=False):
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
    query = alerce.query_probabilities(oid=ztf_name, format="pandas")
    query_transient = query[query["classifier_name"] == "lc_classifier_transient"]
    label = query_transient[query_transient["ranking"] == 1]["class_name"].iat[0]
    return SnClass.from_alerce_to_superphot_format(label) if superphot_style else label
    
def add_alerce_col(transient_data: TransientGroup):
    """Add ALeRCE lookup column.
    """
    transient_data.add_col(
        "alerce_class", partial(get_alerce_pred_class, alerce=Alerce(), superphot_style=True)
    )
    return transient_data
    
    