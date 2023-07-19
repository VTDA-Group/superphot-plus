from superphot_plus.file_utils import read_single_lightcurve


def test_read_single_lightcurve(single_ztf_lightcurve_compressed):
    """Test that we can load a single light curve from pickled file."""

    time, flux, flux_err, band = read_single_lightcurve(single_ztf_lightcurve_compressed)

    assert len(time) == 19
    assert len(flux) == 19
    assert len(flux_err) == 19
    assert len(band) == 19
