"""This script provides functions for importing and manipulating ZTF 
data from the Alerce API."""

import csv
import os
import ast

import pandas as pd

from alerce.core import Alerce

alerce = Alerce()
MIN_PER_FILTER = 5

# pylint: disable=global-variable-not-assigned


def add_stamp_column(input_filename, output_filename):  # pragma: no cover
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
    input_df = pd.read_csv(input_filename)
    
    names = input_df.NAME.to_numpy()
    stamp = []
    
    for i,name in enumerate(names):
        if i % 100 == 0:
            print(i)
        try:
            p = alerce.query_probabilities(oid=name, format="pandas")

            p_class = p[p["classifier_name"] == "stamp_classifier"]
            prob = p_class[p_class["ranking"] == 1]["probability"].iat[0]
            best_label = p_class[p_class["ranking"] == 1]["class_name"].iat[0]

            stamp.append( (best_label == "SN") and (prob >= 0.5) )
        except:
            stamp.append( False )
    
    input_df['STAMP'] = stamp
    input_df.to_csv(output_filename, index=False)


def get_all_unclassified_samples(save_csv, start_i=0):  # pragma: no cover
    """Get all unclassified samples and save them to a CSV file.

    Parameters
    ----------
    save_csv : str
        Path to the output CSV file.
    """
    global alerce

    classifiers = alerce.query_classifiers()
    print(classifiers)
    repeat_names = set()
    i = start_i
    
    if os.path.exists(save_csv):
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
                    classifier="lc_classifier_top",
                    #classifier_version="hierarchical_random_forest_1.0.0",
                    class_name="Transient",
                    format="pandas",
                    page_size=1000,
                    probability=0.5,
                    page=i,
                )
                break
            except:
                print("trying again")
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
                        #print("REPEAT")
                        continue
                    p = alerce.query_probabilities(oid=name, format="pandas")

                    p_class = p[p["classifier_name"] == "lc_classifier_transient"]
                    prob = p_class[p_class["ranking"] == 1]["probability"].iat[0]
                    best_label = p_class[p_class["ranking"] == 1]["class_name"].iat[0]
                    
                    csv_writer.writerow([name, prob, best_label])
                    repeat_names.add(name)

                except:
                    print("skipped")
                    continue
        i += 1
