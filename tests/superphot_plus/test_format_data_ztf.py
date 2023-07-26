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


def test_get_lightcurve_posterior_samples(tmp_path):
    """Test loading the posterior samples from an EQWT fits file"""

    # Create fake lightcurve data.
    times = np.array(range(10))
    fluxes = np.array([100.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1])
    bands = np.array(["r"] * 10)
    errors = np.array([0.1] * 10)
    lc = Lightcurve(times, fluxes, errors, bands)

    filename = os.path.join(tmp_path, "my_test_file")

    # Create simulated file for EQWT fit samples.
    lc.save_to_file(f"{filename}_eqwt", overwrite=True)

    # Read posterior samples from file.
    post_arr = get_lightcurve_posterior_samples(filename, fits_dir=tmp_path)

    assert np.allclose(post_arr[0].astype(float), times)
    assert np.allclose(post_arr[1].astype(float), fluxes)
    assert np.allclose(post_arr[2].astype(float), errors)
    assert np.array_equal(post_arr[3], bands)


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
