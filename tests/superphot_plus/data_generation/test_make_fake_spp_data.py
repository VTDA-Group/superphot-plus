import os

import numpy as np

from superphot_plus.data_generation.make_fake_spp_data import create_clean_models, create_ztf_model


def test_generate_clean_data():
    # Generate 10 light curves with 50 time steps each.
    params, lcs = create_clean_models(10, 50)
    assert len(params) == 10
    assert len(lcs) == 10
    for i in range(10):
        assert lcs[i].obs_count() == 50
        assert lcs[i].obs_count("r") > 0
        assert lcs[i].obs_count("g") > 0
        assert len(params[i]) == 14

    # Generate 5 light curves with 10 time steps each and 3 bands
    params, lcs = create_clean_models(5, 10)
    assert len(params) == 5
    assert len(lcs) == 5
    for i in range(5):
        assert lcs[i].obs_count() == 10
        assert lcs[i].obs_count("r") > 0
        assert lcs[i].obs_count("g") > 0
        #assert lcs[i].obs_count("i") > 0
        assert len(params[i]) == 14


def test_generate_ztf_data():
    ## Basic change detection.
    print(create_ztf_model())
    (A, beta, gamma, t0, tau_rise, tau_fall, es), tdata, filter_data, dirty_model, sigmas = create_ztf_model()
    print((A, beta, gamma, t0, tau_rise, tau_fall, es))
    print(tdata, filter_data, dirty_model, sigmas)
    # assert np.isclose(beta, 0.0039, rtol=0.5)
    # assert np.isclose(es, 0.0099, rtol=0.5, atol=0.1)
