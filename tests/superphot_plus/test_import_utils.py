import pytest

from superphot_plus.import_utils import import_lc


def test_import_lc(single_ztf_lightcurve):
    """Test that we can load a single light curve from CSV"""

    t, f, ferr, b, ra, dec = import_lc(single_ztf_lightcurve)

    assert len(t) == 19
    assert len(f) == 19
    assert len(ferr) == 19
    assert len(b) == 19
    assert ra == pytest.approx(16.93, 0.01)
    assert dec == pytest.approx(32.44, 0.001)
