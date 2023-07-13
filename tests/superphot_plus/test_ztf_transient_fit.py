import os

from superphot_plus.ztf_transient_fit import dynesty_single_file


def test_dynesty_single_file(tmp_path, single_ztf_lightcurve_compressed):
    """Just test that we generated a new file with fits"""
    dynesty_single_file(
        single_ztf_lightcurve_compressed, tmp_path, skip_if_exists=False
    )

    assert os.path.exists(os.path.join(tmp_path, "ZTF22abvdwik_eqwt_dynesty.npz"))
