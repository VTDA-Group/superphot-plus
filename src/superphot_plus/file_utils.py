"""Methods for reading and writing input, intermediate, and output files."""

import numpy as np


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
