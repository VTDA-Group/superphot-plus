import csv
import os
from pathlib import Path

import numpy as np

from superphot_plus.format_data_ztf import (
    get_lightcurve_posterior_samples,
    import_labels_only,
    oversample_using_posteriors,
)
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.supernova_class import SupernovaClass


def test_import_labels_only(tmp_path):
    """Test loading a file of labels and applying filters"""
    csv_file = os.path.join(tmp_path, "labels.csv")
    with open(csv_file, "w+", encoding="utf-8") as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=",")
        csv_writer.writerow(["ZTF_SN_1234", "SN Ic", 4.5])
        csv_writer.writerow(["ZTF_SN_1234", "SN IIn", 4.5])
        csv_writer.writerow(["ZTF_SN_4567", "SN Ib-Ca-rich", 5.6])

    ## With no weighted fits, we skip all of the inputs
    names, labels = import_labels_only([csv_file], SupernovaClass.all_classes())

    assert len(names) == 0
    assert len(labels) == 0

    ## Add one weighted fit file and we should pick up that label.
    fits_dir = os.path.join(tmp_path, "fits")
    os.makedirs(fits_dir, exist_ok=True)
    Path(os.path.join(fits_dir, "ZTF_SN_1234_eqwt.npz")).touch()

    names, labels = import_labels_only([csv_file], SupernovaClass.all_classes(), fits_dir=fits_dir)

    ## Should not include duplicate label.
    assert names == ["ZTF_SN_1234"]
    assert labels == ["SN Ibc"]

    ## Remove that class from the allowed types and we're back to nothing.
    names, labels = import_labels_only(
        [csv_file], [SupernovaClass.SUPERLUMINOUS_SUPERNOVA_I], fits_dir=fits_dir
    )

    assert len(names) == 0
    assert len(labels) == 0


def test_get_lightcurve_posterior_samples(single_ztf_sn_id, single_ztf_eqwt_compressed, test_data_dir):
    """Test loading the posterior samples from an EQWT fits file"""

    # Check existence of EQWT fits file.
    assert os.path.exists(single_ztf_eqwt_compressed)

    # Read posterior samples from file.
    post_arr = get_lightcurve_posterior_samples(single_ztf_sn_id, fits_dir=test_data_dir)

    # Check output length.
    assert len(post_arr) == 300

    # Check output values.
    expected = [
        9.85589522e02,
        5.19716954e-03,
        1.61198756e01,
        -5.75673236e00,
        3.26708896e00,
        2.38970410e01,
        3.64242112e-02,
        1.04759061e00,
        1.04258722e00,
        1.00856218e00,
        9.99988091e-01,
        9.66154117e-01,
        5.76787619e-01,
        8.59146651e-01,
    ]
    sample_mean = np.mean(post_arr, axis=0)
    assert np.all(np.isclose(sample_mean, expected, rtol=0.5))


def test_oversample_using_posteriors(test_data_dir, single_ztf_sn_id):
    """Test oversampling using posteriors"""

    names = [single_ztf_sn_id] * 3
    chis = np.ones(len(names))
    goal_per_class = 10

    # Oversampling from a set of unique supernova classes.
    classes = [4, 1, 2]  # Classes for "Sn Ibc", "SN II" and "SN IIn"
    features, labels, chis = oversample_using_posteriors(names, classes, chis, goal_per_class, test_data_dir)

    # We should have 30 samples in total, 10 for each class.
    assert len(features) == len(labels) == len(chis) == 30
    assert len(labels[labels == 4]) == len(labels[labels == 1]) == len(labels[labels == 2]) == 10

    # Oversampling from a set with repeated supernova classes.
    classes = [4, 1, 1]  # Classes for "Sn Ibc" and "SN II"
    features, labels, chis = oversample_using_posteriors(names, classes, chis, goal_per_class, test_data_dir)

    # We should have less samples due to repeated class.
    assert len(features) == len(labels) == len(chis) == 20
    assert len(labels[labels == 4]) == len(labels[labels == 1]) == 10
