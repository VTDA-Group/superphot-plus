"""This script provides functions for importing and manipulating ZTF 
data from the Alerce API."""

import csv
import os

import numpy as np

from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import convert_mags_to_flux


def import_lc(filename, survey=Survey.ZTF(), clip_lightcurve=True):
    """Imports a single file, but only the points from a single
    survey.

    Parameters
    ----------
    filename : str
        Path to the input CSV file.
    survey : Survey, optional
        Assumes light curve data was taken by this survey. Defaults to ZTF.

    Returns
    -------
    tuple
        Tuple containing the imported light curve data.
    """
    ra = None
    dec = None
    if not os.path.exists(filename):  # pragma: no cover
        return [None] * 6
    with open(filename, "r", encoding="utf-8") as csv_f:
        csvreader = csv.reader(csv_f, delimiter=",")
        row_intro = next(csvreader)

        ra_idx = row_intro.index("ra")
        dec_idx = row_intro.index("dec")
        b_idx = row_intro.index("fid")
        f_idx = row_intro.index("magpsf")
        ferr_idx = row_intro.index("sigmapsf")

        flux = []
        flux_err = []
        mjd = []
        bands = []

        for row in csvreader:
            if ra is None:
                ra = float(row[ra_idx])
                dec = float(row[dec_idx])
                try:
                    ext_dict = survey.get_extinctions(ra, dec)
                except:  # pragma: no cover
                    return [None] * 6
            if int(row[b_idx]) == 2:
                flux.append(float(row[f_idx]) - ext_dict["r"])
                bands.append("r")
            elif int(row[b_idx]) == 1:
                flux.append(float(row[f_idx]) - ext_dict["g"])
                bands.append("g")
            else:  # pragma: no cover
                continue
            mjd.append(float(row[1]))
            flux_err.append(float(row[ferr_idx]))

    sort_idx = np.argsort(np.array(mjd))
    t = np.array(mjd)[sort_idx].astype(float)
    m = np.array(flux)[sort_idx].astype(float)
    merr = np.array(flux_err)[sort_idx].astype(float)
    b = np.array(bands)[sort_idx]

    t = t[merr != np.nan]
    m = m[merr != np.nan]
    b = b[merr != np.nan]
    merr = merr[merr != np.nan]

    f, ferr = convert_mags_to_flux(m, merr, 26.3)

    if clip_lightcurve:
        t, f, ferr, b = clip_lightcurve_end(t, f, ferr, b)
    snr = np.abs(f / ferr)

    for band in survey.wavelengths:
        if len(snr[(snr > 3.0) & (b == band)]) < 5:  # pragma: no cover
            return [None] * 6
        if (np.max(f[b == band]) - np.min(f[b == band])) < 3.0 * np.mean(ferr[b == band]):  # pragma: no cover
            print("SKIPPED BECAUSE AMP")
            return [None] * 6
    return t, f, ferr, b, ra, dec


def clip_lightcurve_end(times, fluxes, fluxerrs, bands):
    """Clips end of lightcurve with approximately 0 slope. Checks from
    back to max of lightcurve.

    Parameters
    ----------
    times : np.ndarray
        Time values of the light curve.
    fluxes : np.ndarray
        Flux values of the light curve.
    fluxerrs : np.ndarray
        Flux error values of the light curve.
    bands : np.ndarray
        Band information of the light curve.

    Returns
    -------
    tuple
        Tuple containing the clipped light curve data.
    """
    t_clip, flux_clip, ferr_clip, b_clip = [], [], [], []
    for b in np.unique(bands):
        idx_b = bands == b
        t_b, f_b, ferr_b = times[idx_b], fluxes[idx_b], fluxerrs[idx_b]
        end_i = len(t_b) - np.argmax(f_b)
        num_to_cut = 0

        if np.argmax(f_b) == len(f_b) - 1:
            t_clip.extend(t_b)
            flux_clip.extend(f_b)
            ferr_clip.extend(ferr_b)
            b_clip.extend([b] * len(f_b))
            continue

        m_cutoff = 0.2 * np.abs((f_b[-1] - np.amax(f_b)) / (t_b[-1] - t_b[np.argmax(f_b)]))

        for i in range(2, end_i):
            cut_idx = -1 * i
            m = (f_b[cut_idx] - f_b[-1]) / (t_b[cut_idx] - t_b[-1])

            if np.abs(m) < m_cutoff:
                num_to_cut = i

        if num_to_cut > 0:
            t_clip.extend(t_b[:-num_to_cut])
            flux_clip.extend(f_b[:-num_to_cut])
            ferr_clip.extend(ferr_b[:-num_to_cut])
            b_clip.extend([b] * len(f_b[:-num_to_cut]))
        else:
            t_clip.extend(t_b)
            flux_clip.extend(f_b)
            ferr_clip.extend(ferr_b)
            b_clip.extend([b] * len(f_b))

    return np.array(t_clip), np.array(flux_clip), np.array(ferr_clip), np.array(b_clip)


def add_to_new_csv(name, label, redshift, output_csv):
    """Add row to CSV of included files for training.

    Parameters
    ----------
    name : str
        Name in the new row.
    label : str
        Label in the new row.
    redshift : float
        Redshift value  in the new row.
    output_csv : str
        The output CSV file path.
    """
    with open(output_csv, "a", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow([name, label, redshift])
