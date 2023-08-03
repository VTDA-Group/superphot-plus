"""Methods for reading and writing input, intermediate, and output files."""

import os

import numpy as np

from superphot_plus.file_paths import FITS_DIR


def read_single_lightcurve(filename, time_ceiling=None):
    """
    Import a compressed lightcurve data file.

    Parameters
    ----------
    filename : str
        Name of the data file.
    time_ceiling : float, optional
        Upper limit for time, and any points in the light curve after this ceiling
        will be dropped. Defaults to None and all points are returned.

    Returns
    -------
    tuple
        Tuple containing the imported data (t, f, ferr, b).
    """
    npy_array = np.load(filename)
    arr = npy_array["arr_0"]

    ferr = arr[2]
    t = arr[0][ferr != "nan"].astype(float)
    f = arr[1][ferr != "nan"].astype(float)
    b = arr[3][ferr != "nan"]
    ferr = ferr[ferr != "nan"].astype(float)

    if time_ceiling is not None:
        f = f[t <= time_ceiling]
        b = b[t <= time_ceiling]
        ferr = ferr[t <= time_ceiling]
        t = t[t <= time_ceiling]

    if len(t) > 0:
        max_flux_loc = t[b == "r"][np.argmax(f[b == "r"] - np.abs(ferr[b == "r"]))]

        t -= max_flux_loc  # make relative

    return t, f, ferr, b


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


def get_multiple_posterior_samples(lc_names, fits_dir):
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
        posterior_samples[lc_name] = get_posterior_samples(lc_name, fits_dir)
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
