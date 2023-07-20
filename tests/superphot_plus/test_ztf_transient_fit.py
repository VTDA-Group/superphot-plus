import os
import numpy as np

from superphot_plus.ztf_transient_fit import dynesty_single_file


def test_dynesty_single_file(tmp_path, single_ztf_lightcurve_compressed):
    """Just test that we generated a new file with fits"""

    # Set a random seed to avoid flaky tests. WARNING: Do not do this
    # for scientific runs as it will not produce correct results.
    assert "DYNESTY_RANDOM_SEED" not in os.environ
    os.environ["DYNESTY_RANDOM_SEED"] = "2023"

    sample_mean = dynesty_single_file(
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

    # Check that the same means are reasonably close (within 10% relative value).
    # Despite setting the the random seed, we still need to account for floating
    # point differences, etc.
    expected = [
        1035.0,
        0.005,
        13.5,
        -4.8,
        4.0,
        23.4,
        0.03,
        1.1,
        1.0,
        1.0,
        1.0,
        0.96,
        0.56,
        0.87,
    ]
    assert np.all(np.isclose(sample_mean, expected, rtol=.1))

    # Unset the random seed for consistency.
    del os.environ["DYNESTY_RANDOM_SEED"]
    assert "DYNESTY_RANDOM_SEED" not in os.environ
