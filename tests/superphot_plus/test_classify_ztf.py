import csv
import os

import numpy as np

from superphot_plus.classify_ztf import classify_single_light_curve, return_new_classifications
from superphot_plus.file_paths import PROBS_FILE, PROBS_FILE2
from superphot_plus.supernova_class import SupernovaClass as SnClass


def test_classify_single_light_curve(classifier, test_data_dir):
    """Classify light curve based on a pretrained model and fit data."""
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    _, classes_to_labels = SnClass.get_type_maps(allowed_types)

    expected_classes = {
        "ZTF22abvdwik": SnClass.SUPERNOVA_IA,
        "ZTF23aacrvqj": SnClass.SUPERNOVA_II,
        "ZTF22abcesfo": SnClass.SUPERNOVA_IIN,
        "ZTF22aarqrxf": SnClass.SUPERLUMINOUS_SUPERNOVA_I,
        "ZTF22abytwjj": SnClass.SUPERNOVA_IBC,
    }

    for ztf_name, label in expected_classes.items():
        lc_probs = classify_single_light_curve(classifier, ztf_name, test_data_dir)
        assert classes_to_labels[np.argmax(lc_probs)] == label


def test_return_new_classifications(classifier, test_data_dir, tmp_path):
    """Classify light curves from a CSV test file."""
    csv_file = os.path.join(tmp_path, "labels.csv")

    with open(csv_file, "w+", encoding="utf-8") as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=",")
        csv_writer.writerow(["Name", "Label"])
        csv_writer.writerow(["ZTF22abvdwik", SnClass.SUPERNOVA_IA])
        csv_writer.writerow(["ZTF23aacrvqj", SnClass.SUPERNOVA_II])
        csv_writer.writerow(["ZTF22abcesfo", SnClass.SUPERNOVA_IIN])
        csv_writer.writerow(["ZTF22aarqrxf", SnClass.SUPERLUMINOUS_SUPERNOVA_I])
        csv_writer.writerow(["ZTF22abytwjj", SnClass.SUPERNOVA_IBC])

    # Save test file with labels
    return_new_classifications(classifier, csv_file, test_data_dir, include_labels=True, output_dir=tmp_path)
    assert os.path.exists(os.path.join(tmp_path, PROBS_FILE))

    # Save test file without labels
    return_new_classifications(classifier, csv_file, test_data_dir, include_labels=False, output_dir=tmp_path)
    assert os.path.exists(os.path.join(tmp_path, PROBS_FILE2))
