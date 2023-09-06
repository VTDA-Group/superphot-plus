from superphot_plus.data_generation.snana import import_snana
import numpy as np


def test_import_snana(snana_filename):
    """Test importing a SNANA file."""
    t, f, ferr, b, ra, dec = import_snana(snana_filename)
    assert (t is not None) and (len(t) > 0)
    assert len(np.unique(b)) == 2  # exclude undefined bands
