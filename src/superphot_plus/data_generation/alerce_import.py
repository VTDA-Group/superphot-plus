"""This script provides functions for importing and manipulating ZTF 
data from the Alerce API."""

import csv
import os

from alerce.core import Alerce

alerce = Alerce()
MIN_PER_FILTER = 5

# pylint: disable=global-variable-not-assigned


def add_stamp_column(input_filename, output_filename):
    """Checks whether stamp classifier categorizes each lightcurve in
    spreadsheet as a supernova-like transient, and adds as additional
    column.

    Parameters
    ----------
    input_filename : str
        Path to the input CSV file.
    output_filename : str
        Path to the output CSV file.
    """
    csv_rows = []
    with open(input_filename, "r", encoding="utf-8") as fn_csv:
        csv_reader = csv.reader(fn_csv, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            csv_rows.append(row)

    print("done reading in rows")
    with open(output_filename, "w+", encoding="utf-8") as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=",")
        csv_writer.writerow(["NAME", "PROB", "CLASS", "STAMP"])
        for row in csv_rows:
            try:
                name = row[0]
                print(name)

                p = alerce.query_probabilities(oid=name, format="pandas")

                p_class = p[p["classifier_name"] == "stamp_classifier"]
                prob = p_class[p_class["ranking"] == 1]["probability"].iat[0]
                best_label = p_class[p_class["ranking"] == 1]["class_name"].iat[0]

                stamp = (best_label == "SN") and (prob >= 0.5)
                csv_writer.writerow([*row, stamp])

            except:
                csv_writer.writerow([*row, "None"])


def get_all_unclassified_samples(save_csv):
    """Get all unclassified samples and save them to a CSV file.

    Parameters
    ----------
    save_csv : str
        Path to the output CSV file.
    """
    global alerce

    classifiers = alerce.query_classifiers()
    print(classifiers)
    i = 40
    repeat_names = set()

    with open(save_csv, "r", encoding="utf-8") as sc:
        csv_reader = csv.reader(sc, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            repeat_names.add(row[0])

    while True:
        print(i)

        while True:
            try:
                objs = alerce.query_objects(
                    classifier="stamp_classifier",
                    classifier_version="stamp_classifier_1.0.4",
                    class_name="SN",
                    format="pandas",
                    page_size=500,
                    probability=0.5,
                    page=i,
                )
                break
            except:
                pass

        if len(objs) == 0:  # finished
            return None

        with open(save_csv, "a+", encoding="utf-8") as sc:
            csv_writer = csv.writer(sc, delimiter=",")

            for row_idx in range(len(objs)):
                try:
                    row = objs.iloc[row_idx]
                    name = row.iat[0]
                    if name in repeat_names:
                        print("REPEAT")
                        continue
                    p = alerce.query_probabilities(oid=name, format="pandas")

                    p_class = p[p["classifier_name"] == "stamp_classifier"]
                    prob = p_class[p_class["ranking"] == 1]["probability"].iat[0]
                    best_label = p_class[p_class["ranking"] == 1]["class_name"].iat[0]
                    if best_label != "SN":
                        print("wrong type")
                        continue
                    csv_writer.writerow([name, prob, best_label])
                    repeat_names.add(name)

                except:
                    print("skipped")
                    continue
        i += 1


def generate_flux_files(master_csv, save_folder):
    """Generates flux files for all ZTF samples in the master CSV file,
    using ALeRCE's API.

    Parameters
    ----------
    master_csv : str
        Path to the master CSV file.
    save_folder : str
        Path to the folder where the flux files will be saved.
    """
    global alerce
    os.makedirs(save_folder, exist_ok=True)
    with open(master_csv, "r", encoding="utf-8") as mc:
        csvreader = csv.reader(mc, delimiter=",", skipinitialspace=True)
        for row in csvreader:
            try:
                ztf_name = row[0]
                if os.path.exists(os.path.join(save_folder, ztf_name + ".csv")):
                    continue
                # print(ztf_name)
                # Getting detections for an object
                detections = alerce.query_detections(ztf_name, format="pandas")
                detections.to_csv(os.path.join(save_folder, ztf_name + ".csv"), index=False)
            except:
                continue


def generate_single_flux_file(ztf_name, save_folder):
    """Generates a flux file for a single ZTF sample in the master CSV
    file, using ALeRCE's API.

    Parameters
    ----------
    ztf_name : str
        Name of the ZTF sample.
    save_folder : str
        Path to the folder where the flux file will be saved.
    """
    global alerce
    os.makedirs(save_folder, exist_ok=True)

    # Getting detections for an object
    detections = alerce.query_detections(ztf_name, format="pandas")
    print(os.path.join(save_folder, ztf_name + ".csv"))
    detections.to_csv(os.path.join(save_folder, ztf_name + ".csv"), index=False)
