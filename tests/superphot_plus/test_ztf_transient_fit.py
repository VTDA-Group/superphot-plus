import os

import numpy as np

from superphot_plus.file_utils import save_lightcurves
from superphot_plus.ztf_transient_fit import dynesty_single_file
from superphot_plus.data_generation.make_fake_spp_data.py import create_clean_models

def test_dynesty_single_file(tmp_path, single_ztf_lightcurve_compressed):
    """Just test that we generated a new file with fits"""
    dynesty_single_file(
        single_ztf_lightcurve_compressed,
        tmp_path,
        skip_if_exists=False,
        rstate=np.random.default_rng(9876),
    )

    output_file = os.path.join(tmp_path, "ZTF22abvdwik_eqwt_dynesty.npz")
    assert os.path.exists(output_file)

    params = np.load(output_file)["arr_0"]
    ## could be between ~600 and ~800, and can vary based on hardware.
    assert 600 <= len(params) <= 800



def test_dynesty_known_fit(tmp_path):
    # Create fake data and save it to a temporary file.
    params, lcs = create_clean_models(1)
    output_file = os.path.join(tmp_path, "my_test_file.npz")
    save_lightcurves(output_file, lcs)

    # Reload the file and use dynesty sampler to estimate the parameters.
    sample_mean = dynesty_single_file(
        output_file,
        tmp_path,
        skip_if_exists=False,
        rstate=np.random.default_rng(9876),
    )

    # Check that we got within 10% of the true parameters.
    assert np.all(np.isclose(sample_mean, params, rtol=.1))
