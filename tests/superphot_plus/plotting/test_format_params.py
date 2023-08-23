import os

import numpy as np
import pytest

from superphot_plus.plotting.format_params import *


def test_param_labels():
    """Test parameter labels are correctly being generated for
    different numbers of auxiliary bands.
    """
    plot_labels, save_labels = param_labels()  # no aux bands

    assert len(plot_labels) == len(save_labels) == 8

    plot_labels_with_g, save_labels_with_g = param_labels(["g"])
    assert len(plot_labels_with_g) == len(save_labels_with_g) == 15

    # large number of bands
    n_aux = 10
    plot_labels_many, save_labels_many = param_labels(np.arange(1, n_aux + 1))
    assert len(plot_labels_many) == len(save_labels_many) == 7 * n_aux + 8
    assert plot_labels_many[-1] == r"$\chi^2$"
    assert plot_labels_many[-8] == r"$A_{10}$"
