import numpy as np

def test_to_numpy(ztf_priors):
    assert np.isclose(ztf_priors.to_numpy().sum(), -88.595, atol=0.2)
