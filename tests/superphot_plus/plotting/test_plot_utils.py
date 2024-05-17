import os

import numpy as np
import pandas as pd
from alerce.core import Alerce

from superphot_plus.plotting.utils import (
    add_snr_to_prob_csv,
    gaussian,
    get_alerce_pred_class,
    histedges_equalN,
    read_probs_csv,
)


def test_gaussian(tmp_path):
    """Test the gaussian helper function."""
    assert gaussian(0.0, 1.0, 0.0, 1.0) == 1.0
    assert np.isclose(gaussian([0.0, 1.0], 1.0, 0.0, 1.0), [1.0, 0.606531], rtol=1e-3).all()


def test_histedges_equalN():
    """Test histogram bin edges generation, such that counts
    are equal in each bin.
    """
    nbin = 2
    x = [1, 1, 1, 2, 3, 3, 5, 5]
    assert np.all(histedges_equalN(x, nbin) == [1, 3, 5])

    # out of order
    x_shuffled = [5, 1, 3, 2, 3, 1, 5, 1]
    assert np.all(histedges_equalN(x_shuffled, nbin) == [1, 3, 5])


def test_read_probs_csv(class_probs_csv):
    """Test reading in a probability CSV."""
    names, labels, probs, pred_classes, _, df = read_probs_csv(class_probs_csv)

    assert len(names) == len(labels) == len(probs) == len(pred_classes) == 499
    assert probs.shape[1] == 5
    assert np.array_equal(df.Name.to_numpy(), names)


def test_get_alerce_pred_class(single_ztf_sn_id):
    """Test we can obtain ALeRCE predictions in both its native
    and Superphot+'s label formatting.
    """
    alerce = Alerce()
    label = get_alerce_pred_class(single_ztf_sn_id, alerce)
    assert label in ["SNIa", "SNII", "SLSN", "SNIbc"]

    label_superphot = get_alerce_pred_class(single_ztf_sn_id, alerce, superphot_style=True)
    assert label_superphot in ["SN Ia", "SN II", "SLSN-I", "SN Ibc"]


def test_add_snr_to_prob_csv(class_probs_csv, class_probs_snr_csv, test_data_dir, tmp_path):
    """Test whether SNR values are being properly added to an existing
    probability CSV.
    """
    new_csv_path = os.path.join(tmp_path, "probs_snr.csv")
    add_snr_to_prob_csv(class_probs_csv, test_data_dir, new_csv_path)
    df = pd.read_csv(new_csv_path)
    df_orig = pd.read_csv(class_probs_snr_csv)
    assert np.array_equal(df.columns, df_orig.columns)
