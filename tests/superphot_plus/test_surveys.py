import os

import numpy as np
import pytest

from superphot_plus.priors.fitting_priors import MultibandPriors
from superphot_plus.surveys import Survey


def test_create():
    """Create some valid and invalid Survey objects."""
    # generate default Survey
    default_survey = Survey()
    assert default_survey.name == ""
    assert len(default_survey.wavelengths) == 0
    assert len(default_survey.priors.bands) == 0

    # generate pre-filled Survey
    default_priors = MultibandPriors()
    ztf_priors = MultibandPriors.load_ztf_priors()
    # test extra wavelengths defined = OK
    test_valid_survey = Survey("Test", default_priors, {"r": 5000.0})
    assert test_valid_survey.name == "Test"
    assert len(test_valid_survey.wavelengths) == 1
    assert len(test_valid_survey.priors.bands) == 0

    # test not enough wavelengths defined = NOT OK
    with pytest.raises(AssertionError):
        test_invalid_survey = Survey("Test2", ztf_priors, {"r": 5000.0})


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
