import numpy as np


def read_single_lightcurve(filename, t0_lim=None):
    """
    Import a compressed lightcurve data file.

    Parameters
    ----------
    filename : str
        Name of the data file.
    t0_lim : float, optional
        Upper limit for t0. Defaults to None.
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

    if t0_lim is not None:
        f = f[t <= t0_lim]
        b = b[t <= t0_lim]
        ferr = ferr[t <= t0_lim]
        t = t[t <= t0_lim]

    max_flux_loc = t[b == "r"][np.argmax(f[b == "r"] - np.abs(ferr[b == "r"]))]

    t -= max_flux_loc  # make relative

    return t, f, ferr, b
