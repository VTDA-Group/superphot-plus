import os

import numpy as np
import pandas as pd
import pytest
from jax import random
from snapi import Transient
from snapi.analysis import SamplerResult

from superphot_plus.model.mlp import SuperphotMLP
from superphot_plus.trainer import SuperphotTrainer

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
def single_ztf_sn_id():
    return "ZTF22abvdwik"

@pytest.fixture
def test_ztf_transient(test_data_dir):
    return Transient.load(
        os.path.join(test_data_dir, "2022abfi.h5")
    )

@pytest.fixture
def test_ztf_photometry(test_ztf_transient):
    return test_ztf_transient.photometry

@pytest.fixture
def ztf_priors():
    return Survey.ZTF().priors

@pytest.fixture
def test_sampler_result(test_data_dir):
    return SamplerResult.load(
       load_prefix="2022abfi_result",
       load_folder=test_data_dir
    )

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
def test_class_frac_csv(test_data_dir):
    return os.path.join(test_data_dir, "test_cf.csv")

@pytest.fixture
def config_filename(test_data_dir):
    return os.path.join(test_data_dir, "superphot-config-test.yaml")


@pytest.fixture
def mlp(test_data_dir, config_filename):
    filename = os.path.join(test_data_dir, "training_paper_mlp-0.pt")
    return SuperphotMLP.load(filename, config_filename)[0]


@pytest.fixture
def trainer_mlp(test_data_dir, config_filename, mlp):
    trainer = SuperphotTrainer(
        config_filename,
        test_data_dir,
        sampler="dynesty",
        model_type='MLP',
        include_redshift=False,
        n_folds=1,
    )
    trainer.setup_model(load_checkpoint=False)
    # manual override
    trainer.models[0] = mlp
    return trainer

@pytest.fixture
def trainer_lightgbm(test_data_dir, config_filename, gbm):
    trainer = SuperphotTrainer(
        config_filename,
        test_data_dir,
        sampler="dynesty",
        model_type='LightGBM',
        include_redshift=False,
        n_folds=1,
    )
    trainer.setup_model(load_checkpoint=False)
    # manual override
    #trainer.models[0] = mlp
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
