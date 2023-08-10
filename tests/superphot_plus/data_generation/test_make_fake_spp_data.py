import os

import numpy as np

from superphot_plus.data_generation.make_fake_spp_data import create_clean_models, create_ztf_model


def test_generate_clean_data():
    # Generate 10 light curves with 50 time steps each.
    params, lcs = create_clean_models(10, 50)
    assert len(params) == 10
    assert len(lcs) == 10
    for i in range(10):
        assert lcs[i].shape == (4, 50)
        assert len(params[i]) == 14


def test_generate_ztf_data():
    ## Basic change detection.
    (A, beta, gamma, t0, tau_rise, tau_fall, es), tdata, filter_data, dirty_model, sigmas = create_ztf_model()
    print((A, beta, gamma, t0, tau_rise, tau_fall, es))
    print(tdata, filter_data, dirty_model, sigmas)
    assert np.isclose(beta, 0.0039, rtol=0.1)
    assert np.isclose(es, 0.0099, rtol=0.1)
