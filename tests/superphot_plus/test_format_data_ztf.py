import csv
import os
from pathlib import Path

import numpy as np

from superphot_plus.format_data_ztf import (
    import_labels_only,
    normalize_features
)
from superphot_plus.supernova_class import SupernovaClass


def test_import_labels_only(tmp_path):
    """Test loading a file of labels and applying filters"""
    csv_file = os.path.join(tmp_path, "labels.csv")
    with open(csv_file, "w+", encoding="utf-8") as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=",")
        csv_writer.writerow(["NAME", "CLASS", "Z"])
        csv_writer.writerow(["ZTF_SN_1234", "SN Ic", 4.5])
        csv_writer.writerow(["ZTF_SN_1234", "SN IIn", 4.5])
        csv_writer.writerow(["ZTF_SN_4567", "SN Ib-Ca-rich", 5.6])

    ## With no weighted fits, we skip all of the inputs
    names, labels, redshifts = import_labels_only(
        [csv_file],
        SupernovaClass.all_classes(),
    )

    assert len(names) == 0
    assert len(labels) == 0
    assert len(redshifts) == 0

    ## Add one weighted fit file and we should pick up that label.
    fits_dir = os.path.join(tmp_path, "fits")
    os.makedirs(fits_dir, exist_ok=True)
    Path(os.path.join(fits_dir, "ZTF_SN_1234_eqwt.npz")).touch()

    names, labels, redshifts = import_labels_only(
        [csv_file],
        SupernovaClass.all_classes(),
        fits_dir=fits_dir
    )

    ## Should not include duplicate label.
    assert names == ["ZTF_SN_1234"]
    assert labels == ["SN Ibc"]
    assert redshifts == [4.5]

    ## Remove that class from the allowed types and we're back to nothing.
    names, labels, redshifts = import_labels_only(
        [csv_file], [SupernovaClass.SUPERLUMINOUS_SUPERNOVA_I], fits_dir=fits_dir
    )

    assert len(names) == 0
    assert len(labels) == 0
    assert len(redshifts) == 0

def test_normalize_features():
    # Feature #1: mean = 1.0, std ~= 0.81649658
    # Feature #2: mean = 0.75, std ~= 0.54006172
    # Feature #3: mean = 1.0, std == 0.0
    features = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [2.0, 1.25, 1.0]])
    expected = np.array(
        [
            [0.0, 0.25 / 0.54006172, 0.0],
            [-1.0 / 0.81649658, -0.75 / 0.54006172, 0.0],
            [1.0 / 0.81649658, 0.5 / 0.54006172, 0.0],
        ]
    )
    computed, mean, std = normalize_features(features)

    assert np.allclose(mean, [1.0, 0.75, 1.0])
    assert np.allclose(std, [0.81649658, 0.54006172, 0.0])
    assert np.allclose(computed, expected)
