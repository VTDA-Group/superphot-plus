"""Methods for reading and writing input, intermediate, and output files."""

import os

import numpy as np

from superphot_plus.file_paths import FITS_DIR


def get_posterior_filename(lc_name, fits_dir=None, sampler=None):
    """Get the file name for equal weight posterior samples from a lightcurve fit.

    Parameters
    ----------
    lc_name : str
        Lightcurve name.
    fits_dir : str, optional
        Output directory path. Defaults to FITS_DIR.
    sampler : str, optional
        Variety of sampler. Can be included in the sample file name.

    Returns
    -------
    str
        File name for numpy array file containing the posterior samples.
    """
    if fits_dir is None:
        fits_dir = FITS_DIR
    if sampler is not None:
        posterior_filename = os.path.join(fits_dir, f"{lc_name}_eqwt_{sampler}.npz")
    else:
        posterior_filename = os.path.join(fits_dir, f"{lc_name}_eqwt.npz")
    return posterior_filename


def get_posterior_samples(lc_name, fits_dir=None, sampler=None):
    """Get all EQUAL WEIGHT posterior samples from a lightcurve fit.

    Parameters
    ----------
    lc_name : str
        Lightcurve name.
    fits_dir : str, optional
        Output directory path. Defaults to FITS_DIR.
    sampler : str, optional
        Variety of sampler. Can be included in the sample file name.

    Returns
    -------
    np.ndarray
        Numpy array containing the posterior samples.
    """
    posterior_filename = get_posterior_filename(lc_name, fits_dir, sampler)

    return np.load(posterior_filename)["arr_0"]


def get_multiple_posterior_samples(lc_names, fits_dir, sampler=None):
    """Reads all EQUAL WEIGHT posterior samples for a set of lightcurve fits.

    Parameters
    ----------
    lc_names : str
        Lightcurve names.
    fits_dir : str, optional
        Output directory path. Defaults to FITS_DIR.
    sampler : str, optional
        Variety of sampler. Can be included in the sample file name.

    Returns
    -------
    dict of np.ndarray
        Dictionary mapping the posterior samples to the light curves specified.
    """
    posterior_samples = {}
    for lc_name in np.unique(lc_names):
        posterior_samples[lc_name] = get_posterior_samples(lc_name, fits_dir, sampler)
    return posterior_samples


def has_posterior_samples(lc_name, fits_dir=None, sampler=None):
    """Determine if we already have some posterior sample data for the lightcurve.

    Parameters
    ----------
    lc_name : str
        Lightcurve name.
    fits_dir : str, optional
        Output directory path. Defaults to FITS_DIR.
    sampler : str, optional
        Variety of sampler. Can be included in the sample file name.

    Returns
    -------
    boolean
        Does a file already exist for the lightcurve fit
    """
    posterior_filename = get_posterior_filename(lc_name, fits_dir, sampler)

    return os.path.isfile(posterior_filename)
