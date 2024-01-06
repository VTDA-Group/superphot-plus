import os

import numpy as np
import pandas as pd
import pytest
from jax import random

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.model.mlp import SuperphotMLP
from superphot_plus.trainer import SuperphotTrainer
#from superphot_plus.model.lightgbm import SuperphotLightGBM

from superphot_plus.surveys.surveys import Survey

TEST_DIR = os.path.dirname(__file__)

DATA_DIR_NAME = "data"

# pylint: disable=missing-function-docstring, redefined-outer-name


@pytest.fixture
def test_data_dir():
    return os.path.join(TEST_DIR, DATA_DIR_NAME)


@pytest.fixture
def training_csv(test_data_dir):
    return os.path.join(test_data_dir, "training_set.csv")


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
def single_ztf_lightcurve_fit():
    """
    return [
        0.10358849,
        0.00973718,
        1.25686431,
        -5.22735547,
        0.48882998,
        1.31443557,
        -1.76931969,
        -0.03194404,
        -0.18223563,
        0.02049174,
        -1.64989513,
        -0.09223167,
        -0.14907296,
        -0.15285657,
        0.74649619
    ]
    """
    return [
        0.10701995,
        0.00953622,
        1.23953683,
        -5.10189513,
        0.50944826,
        1.32378597,
        -1.7695045,
        -0.03006368,
        -0.25154109,
        0.02070548,
        -1.65427578,
        -0.08849417,
        -0.16305463,
        -0.19222715,
        0.49539004
    ]

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
def test_class_frac_csv(test_data_dir):
    return os.path.join(test_data_dir, "test_cf.csv")

@pytest.fixture
def config_filename(test_data_dir):
    return os.path.join(test_data_dir, "superphot-config-test.yaml")

@pytest.fixture
def mlp(test_data_dir, config_filename):
    filename = os.path.join(test_data_dir, "superphot-model-ZTF23aagkgnz.pt")
    return SuperphotMLP.load(filename, config_filename)[0]

@pytest.fixture
def trainer_mlp(test_data_dir, config_filename, mlp):
    trainer = SuperphotTrainer(
        config_filename,
        test_data_dir,
        sampler="dynesty",
        model_type='MLP',
        n_folds=1,
    )
    trainer.setup_model(load_checkpoint=False)
    # manual override
    trainer.models[0] = mlp
    return trainer

@pytest.fixture
def dummy_alerce_preds(class_probs_csv, test_data_dir):
    names = pd.read_csv(class_probs_csv).Name
    dummy_labels = ["SN Ia"] * len(names)
    alerce_df = pd.DataFrame({"name": names, "alerce_label": dummy_labels})
    fn = os.path.join(test_data_dir, "test_alerce_preds.csv")
    alerce_df.to_csv(fn)
    return fn


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


@pytest.fixture
def snana_filename(test_data_dir):
    """Filename to SNANA ASCII File."""
    return os.path.join(test_data_dir, "sample.snana.txt")
