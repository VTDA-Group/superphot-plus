import os

import numpy as np
import pytest

from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.surveys.surveys import Survey


def test_create(ztf_priors):
    """Create some valid and invalid Survey objects."""
    # generate default Survey
    default_survey = Survey()
    assert default_survey.name == ""
    assert len(default_survey.wavelengths) == 0
    assert len(default_survey.priors.bands) == 0

    # generate pre-filled Survey
    default_priors = MultibandPriors()
    # test extra wavelengths defined = OK
    test_valid_survey = Survey("Test", default_priors, {"r": 5000.0})
    assert test_valid_survey.name == "Test"
    assert len(test_valid_survey.wavelengths) == 1
    assert len(test_valid_survey.priors.bands) == 0

    # test not enough wavelengths defined = NOT OK
    with pytest.raises(AssertionError):
        _ = Survey("Test2", ztf_priors, {"r": 5000.0})


def test_ztf():
    """Test ZTF Survey generation."""
    ztf_survey = Survey.ZTF()
    assert ztf_survey.name == "ZTF"
    assert len(ztf_survey.priors.bands) == 2
    assert ("r" in ztf_survey.wavelengths) and ("g" in ztf_survey.wavelengths)
    assert len(ztf_survey.wavelengths) == 2

    # confirm band order self-consistent
    if np.array_equal(ztf_survey.priors.ordered_bands, ["r", "g"]):
        assert np.array_equal(ztf_survey.get_ordered_wavelengths(), [6173.23, 4741.64])
    elif np.array_equal(ztf_survey.priors.ordered_bands, ["g", "r"]):
        assert np.array_equal(ztf_survey.get_ordered_wavelengths(), [4741.64, 6173.23])
    else:
        assert False

    ## Confirm a single prior value (to ensure chained deserialization)
    assert ztf_survey.priors.bands["r"].gamma.mean == 1.4258

    
def test_lsst():
    """Test LSST/Rubin Survey generation."""
    lsst_survey = Survey.LSST()
    assert lsst_survey.name == "LSST"
    assert len(lsst_survey.priors.bands) == 6
    for b in "ugrizY":
        assert b in lsst_survey.wavelengths
        
    # assert band order + wavelengths
    assert np.array_equal(
        lsst_survey.priors.ordered_bands,
        ["u","g","r","i","z","Y"]
    )
    assert np.array_equal(
        lsst_survey.get_ordered_wavelengths(),
        [3751.20, 4741.64, 6173.23, 7500.97, 8678.90, 9711.82],
    )

    ## Confirm a single prior value (to ensure chained deserialization)
    assert lsst_survey.priors.bands["u"].gamma.mean == 1.0075

    
def test_write_to_file(tmp_path):
    """Test that we can save survey data to a file, and fetch it later."""
    default_priors = MultibandPriors()
    default_survey = Survey("default", default_priors, {"r": 5000.0})

    default_survey.write_to_file(os.path.join(tmp_path, "default_survey.yaml"))
    read_values = Survey.from_file(os.path.join(tmp_path, "default_survey.yaml"))
    assert read_values == default_survey
