from superphot_plus.file_utils import read_single_lightcurve


def test_read_single_lightcurve(single_ztf_lightcurve_compressed):
    """Test that we can load a single light curve from pickled file."""

    time, flux, flux_err, band = read_single_lightcurve(single_ztf_lightcurve_compressed)

    assert len(time) == 19
    assert len(flux) == 19
    assert len(flux_err) == 19
    assert len(band) == 19


def test_read_single_lightcurve_with_time_celing(single_ztf_lightcurve_compressed):
    """Test that we can load a single light curve from pickled file,
    restricting the time window for events."""

    ## nonsense time.
    time, flux, flux_err, band = read_single_lightcurve(single_ztf_lightcurve_compressed, time_ceiling=45.0)

    expected_length = 0
    assert len(time) == expected_length
    assert len(flux) == expected_length
    assert len(flux_err) == expected_length
    assert len(band) == expected_length

    ## time somewhere within the light curve.
    time, flux, flux_err, band = read_single_lightcurve(
        single_ztf_lightcurve_compressed, time_ceiling=59932.0
    )

    expected_length = 15
    assert len(time) == expected_length
    assert len(flux) == expected_length
    assert len(flux_err) == expected_length
    assert len(band) == expected_length
