import os

import numpy as np

from superphot_plus.data_generation.make_fake_spp_data import create_clean_models


def test_generate_clean_data():
    # Generate 10 light curves with 50 time steps each.
    params, lcs = create_clean_models(10, 50)
    assert len(params) == 10
    assert len(lcs) == 10
    for i in range(10):
        assert lcs[i].shape == (4, 50)
        assert len(params[i]) == 14
