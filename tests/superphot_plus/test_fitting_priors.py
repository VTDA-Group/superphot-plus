import os

import numpy as np

from superphot_plus.priors.fitting_priors import CurvePriors, MultibandPriors, PriorFields


def test_write_to_file(tmp_path) -> None:
    default = MultibandPriors(
        {
            "r": CurvePriors(
                PriorFields(),
                PriorFields(),
                PriorFields(),
                PriorFields(),
                PriorFields(),
                PriorFields(),
                PriorFields(),
            ),
            "g": CurvePriors(
                PriorFields(),
                PriorFields(),
                PriorFields(),
                PriorFields(),
                PriorFields(),
                PriorFields(),
                PriorFields(),
            ),
        }
    )

    default.write_to_file(os.path.join(tmp_path, "empty_priors.yaml"))
    read_values = MultibandPriors.from_file(os.path.join(tmp_path, "empty_priors.yaml"))
    assert read_values == default


def test_read_survey_priors():
    ztf_priors = MultibandPriors.load_ztf_priors()

    assert len(ztf_priors.bands) == 2


def test_to_numpy():
    ztf_priors = MultibandPriors.load_ztf_priors()
    assert np.isclose(ztf_priors.to_numpy().sum(), 181.5, atol=0.2)
