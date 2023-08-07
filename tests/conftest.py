import os
import os.path

from jax import random

import pytest
import torch

from superphot_plus.classify_ztf import load_mlp

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.priors.fitting_priors import MultibandPriors

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
def ztf_priors():
    return MultibandPriors.load_ztf_priors()


@pytest.fixture
def classifier(test_data_dir):
    mlp_filename = os.path.join(test_data_dir, "superphot-model-ZTF23aagkgnz.pt")
    mlp_params = (13, 5, 128, 3)
    return load_mlp(mlp_filename, mlp_params)