"""This script provides functions for importing and manipulating ZTF 
data from the Antares API."""

import csv
import os

import numpy as np
from antares_client.search import get_by_ztf_object_id

from superphot_plus.format_data_ztf import tally_each_class
from superphot_plus.import_utils import add_to_new_csv, clip_lightcurve_end
from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import convert_mags_to_flux


def generate_files_from_antares(
    input_csv, output_folder, output_csv
):  # pylint: disable=too-many-statements; # pragma: no cover
    """Generates flux files for all ZTF samples in the master CSV file,
    using ANTARES' API.

    Includes correct zeropoints.

    input_csv : str
        The path to the input CSV file.
    output_folder : str
        Path to the output folder.
    output_csv : str
        The output CSV file path.
    """
    with open(output_csv, "w+", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["Name", "Label", "Redshift"])

    label_dict = {}
    with open(input_csv, "r", encoding="utf-8") as mc:
        csvreader = csv.reader(mc, delimiter=",", skipinitialspace=True)
        next(csvreader)
        for row in csvreader:
            try:
                ztf_name = row[0]
                if os.path.exists(f"{output_folder}/{str(ztf_name)}.npz"):
                    continue
                print(ztf_name)
                # Getting detections for an object
                locus = get_by_ztf_object_id(ztf_name)
                ts = locus.timeseries[
                    [
                        "ant_mjd",
                        "ztf_magpsf",
                        "ztf_sigmapsf",
                        "ztf_fid",
                        "ant_ra",
                        "ant_dec",
                        "ztf_magzpsci",
                    ]
                ]

            except:
                continue

            label = row[3]
            print(label)
            try:
                redshift = float(row[4].strip())
            except:
                redshift = -1

            t, m, merr, b_int, ra, dec, zp = ts.to_pandas().to_numpy().T
            b = np.where(b_int.astype(int) == 1, "g", "r")
            try:
                ra = np.mean(ra[~np.isnan(ra)])
                dec = np.mean(dec[~np.isnan(dec)])
                extinctions = Survey.ZTF().get_extinctions(ra, dec)
            except:
                continue
            m[b == "r"] -= extinctions["r"]
            m[b == "g"] -= extinctions["g"]

            valid_idx = ~np.isnan(merr) & ~np.isnan(zp)
            t = t[valid_idx]
            m = m[valid_idx]
            b = b[valid_idx]
            zp = zp[valid_idx]
            merr = merr[valid_idx]

            f, ferr = convert_mags_to_flux(m, merr, zp)
            t, f, ferr, b = clip_lightcurve_end(t, f, ferr, b)
            snr = np.abs(f / ferr)

            if len(snr[(snr > 3.0) & (b == "g")]) < 5:  # not enough good datapoints
                print("snr too low")
                continue
            if (np.max(f[b == "g"]) - np.min(f[b == "g"])) < 3.0 * np.mean(ferr[b == "g"]):
                continue

            if len(snr[(snr > 3.0) & (b == "r")]) < 5:  # not enough good datapoints
                print("snr too low")
                continue
            if (np.max(f[b == "r"]) - np.min(f[b == "r"])) < 3.0 * np.mean(ferr[b == "r"]):
                continue

            lc = Lightcurve(
                name=ztf_name,
                times=t,
                fluxes=f,
                flux_errors=ferr,
                bands=b,
            )

            lc.save_to_file(os.path.join(output_folder, ztf_name + ".npz"))
            add_to_new_csv(ztf_name, label, redshift, output_csv)

    tally_each_class(label_dict)
