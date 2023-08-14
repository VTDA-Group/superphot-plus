import os
import os.path

import pytest
from jax import random
import numpy as np

from superphot_plus.constants import TRAINED_MODEL_PARAMS
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.mlp import ModelConfig, MLP
from superphot_plus.surveys.surveys import Survey

TEST_DIR = os.path.dirname(__file__)

DATA_DIR_NAME = "data"

# pylint: disable=missing-function-docstring, redefined-outer-name


@pytest.fixture
def test_data_dir():
    return os.path.join(TEST_DIR, DATA_DIR_NAME)


@pytest.fixture
def single_ztf_lightcurve(test_data_dir):
    return os.path.join(test_data_dir, "ZTF22abvdwik.csv")


@pytest.fixture
def single_ztf_lightcurve_compressed(test_data_dir):
    return os.path.join(test_data_dir, "ZTF22abvdwik.npz")


@pytest.fixture
def single_ztf_lightcurve_object(single_ztf_lightcurve_compressed):
    return Lightcurve.from_file(single_ztf_lightcurve_compressed)


@pytest.fixture
def single_ztf_eqwt_compressed(test_data_dir):
    return os.path.join(test_data_dir, "ZTF22abvdwik_eqwt.npz")


@pytest.fixture
def single_ztf_sn_id():
    return "ZTF22abvdwik"


@pytest.fixture
def jax_key():
    return random.PRNGKey(4)


@pytest.fixture
def class_probs_csv(test_data_dir):
    return os.path.join(test_data_dir, "probs.csv")


@pytest.fixture
def class_probs_snr_csv(test_data_dir):
    return os.path.join(test_data_dir, "probs_snr.csv")


@pytest.fixture
def ztf_priors():
    return Survey.ZTF().priors


@pytest.fixture
def classifier(test_data_dir):
    filename = os.path.join(test_data_dir, "superphot-model-ZTF23aagkgnz.pt")
    config = ModelConfig(*TRAINED_MODEL_PARAMS)
    return MLP.load(filename, config)


@pytest.fixture
def dummy_posterior_sample_dict():
    """Create a posterior sample dictionary for r and g bands with random values"""
    param_list = [
        "logA",
        "beta",
        "log_gamma",
        "t0",
        "log_tau_rise",
        "log_tau_fall",
        "log_extra_sigma",
        "A_g",
        "beta_g",
        "gamma_g",
        "t0_g",
        "tau_rise_g",
        "tau_fall_g",
        "extra_sigma_g",
    ]
    return {param: np.random.rand(1, 20).flatten() for param in param_list}


@pytest.fixture
def dummy_posterior_sample_dict_batch():
    """Create a batched posterior sample dictionary for r and g bands with random values"""
    param_list = [
        "logA",
        "beta",
        "log_gamma",
        "t0",
        "log_tau_rise",
        "log_tau_fall",
        "log_extra_sigma",
        "A_g",
        "beta_g",
        "gamma_g",
        "t0_g",
        "tau_rise_g",
        "tau_fall_g",
        "extra_sigma_g",
    ]
    return {param: np.random.rand(3, 20) for param in param_list}
