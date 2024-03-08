"""This script provides functions for importing and manipulating ZTF 
data from the Alerce API."""

import csv
import os

import numpy as np
import pandas as pd

from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import convert_mags_to_flux

LOW_SNR_FILE="low_snr_classes.dat"
LOW_VAR_FILE="low_var_classes.dat"

def import_lc(
    filename,
    tpe=None,
    survey=Survey.ZTF(),
    clip_lightcurve=True,
    low_snr_file=LOW_SNR_FILE,
    low_var_file=LOW_VAR_FILE
):
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
    if not os.path.exists(filename):  # pragma: no cover
        print(filename, "BAD FILE")
        return [None] * 6
    
    single_df = pd.read_csv(filename)
    sub_df = single_df[["mjd", "ra", "dec", "fid", "magpsf", "sigmapsf"]]
    pruned_df = sub_df.dropna(subset=["mjd", "fid", "magpsf", "sigmapsf"])
    pruned_df2 = pruned_df.drop(
        pruned_df[pruned_df['fid'] > 2].index
    ) # remove i band
    sorted_df = pruned_df2.sort_values(by=['mjd'])
    sorted_df['bandpass'] = np.where(sorted_df.fid.to_numpy() == 1, 'g', 'r')
    sorted_df = sorted_df.drop(columns=['fid',])
    
    ra = np.nanmean(sorted_df.ra.to_numpy())
    dec = np.nanmean(sorted_df.dec.to_numpy())
    
    if np.isnan(ra) or np.isnan(dec):
        print(filename, "BAD LOC")
        return [None] * 6
    
    try:
        ext_dict = survey.get_extinctions(ra, dec)
    except:
        print(filename, "BAD LOC")
        return [None] * 6

    m = sorted_df.magpsf.to_numpy()
    merr = sorted_df.sigmapsf.to_numpy()
    b = sorted_df.bandpass.to_numpy()
    t = sorted_df.mjd.to_numpy()
    
    m[b == "r"] -= ext_dict['r']
    m[b == "g"] -= ext_dict['g']
    
    f, ferr = convert_mags_to_flux(m, merr, 26.3)

    if clip_lightcurve:
        t, f, ferr, b = clip_lightcurve_end(
            t, f, ferr, b
        )

    snr = np.abs(f / ferr)

    for band in survey.wavelengths:
        if len(snr[(snr > 3.0) & (b == band)]) < 5:
            with open(low_snr_file, "a+") as f:
                f.write(f"{tpe}\n")
            return [None] * 6
        if np.std(f[b == band]) < np.mean(ferr[b == band]):
            with open(low_var_file, "a+") as f:
                f.write(f"{tpe}\n")
            return [None] * 6
        if np.max(f[b == band]) - np.min(f[b == band]) < 3. * np.mean(ferr[b == band]):  # pragma: no cover
            with open(low_var_file, "a+") as f:
                f.write(f"{tpe}\n")
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
