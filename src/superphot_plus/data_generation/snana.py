import pandas as pd
import numpy as np
from superphot_plus.surveys.surveys import Survey
from superphot_plus.import_utils import clip_lightcurve_end


def import_snana(snana_fn, survey=Survey.ZTF(), clip_lightcurve=True):
    """Import SNANA formatted ASCII file. Returns
    output in same format as import_lc.
    """
    header_rows = []
    header = {}

    with open(snana_fn, "r") as sf:
        for row_num, row in enumerate(sf):
            if row[0] == "#":
                header_rows.append(row_num)
                header_key, header_val = row.split(":", 1)
                header_key = header_key.strip("#").strip()
                header_val = header_val.strip()
                header[header_key] = header_val

    df = pd.read_table(snana_fn, sep=r"\s+", skiprows=header_rows)

    # remove first column
    mjd, bands, flux, flux_err = df[["MJD", "FLT", "FLUXCAL", "FLUXCALERR"]].to_numpy().T

    try:
        # find RA and DEC
        ra = header["RA"]
        dec = header["DEC"]

        # correct for extinction
        ext_dict = survey.get_extinctions(ra, dec)
    except:
        return [
            None,
        ] * 6

    sort_idx = np.argsort(np.array(mjd))
    t = np.array(mjd)[sort_idx].astype(float)
    f = np.array(flux)[sort_idx].astype(float)
    ferr = np.array(flux_err)[sort_idx].astype(float)
    b = np.array(bands)[sort_idx]

    no_nan_mask = ferr != np.nan
    t = t[no_nan_mask]
    f = f[no_nan_mask]
    b = b[no_nan_mask]
    ferr = ferr[no_nan_mask]

    unique_b = np.unique(b)

    for ub in unique_b:
        if ub in survey.wavelengths:
            f[b == ub] *= 10 ** (0.4 * ext_dict[ub])
        else:
            t = t[b != ub]
            f = f[b != ub]
            ferr = ferr[b != ub]
            b = b[b != ub]

    if clip_lightcurve:
        t, f, ferr, b = clip_lightcurve_end(t, f, ferr, b)

    snr = np.abs(f / ferr)

    for band in survey.wavelengths:
        if len(snr[(snr > 3.0) & (b == band)]) < 5:  # pragma: no cover
            return [None] * 6
        if (np.max(f[b == band]) - np.min(f[b == band])) < 3.0 * np.mean(ferr[b == band]):  # pragma: no cover
            return [None] * 6

    # look for some keywords used in LightCurve object, move rest to kwargs

    return t, f, ferr, b, ra, dec
