import os

import numpy as np

from superphot_plus.data_generation.make_fake_spp_data import create_clean_models, create_ztf_model


def test_generate_clean_data():
    # Generate 10 light curves with 50 time steps each.
    params, phots = create_clean_models(10, 50)
    assert len(params) == 10
    assert len(phots) == 10
    for i in range(10):
        #assert len(phots[i]) == 50
        #assert lcs[i].obs_count("r") > 0
        #assert lcs[i].obs_count("g") > 0
        assert len(params[i]) == 14


def test_generate_ztf_data():
    ## Basic change detection.
    cube, phot = create_ztf_model()
    print(cube)
    print(phot)
    # assert np.isclose(beta, 0.0039, rtol=0.5)
    # assert np.isclose(es, 0.0099, rtol=0.5, atol=0.1)
