import os

import pytest

from superphot_plus.model.classifier import SuperphotClassifier

BENCHMARKS_DIR = os.path.dirname(__file__)

DATA_DIR_NAME = "data"

# pylint: disable=missing-function-docstring, redefined-outer-name


@pytest.fixture
def benchmarks_data_dir():
    return os.path.join(BENCHMARKS_DIR, DATA_DIR_NAME)


@pytest.fixture
def classifier(benchmarks_data_dir):
    filename = os.path.join(benchmarks_data_dir, "superphot-model-ZTF23aagkgnz.pt")
    config_filename = os.path.join(benchmarks_data_dir, "superphot-config-test.yaml")
    return SuperphotClassifier.load(filename, config_filename)[0]


@pytest.fixture
def single_ztf_lightcurve_compressed(benchmarks_data_dir):
    return os.path.join(benchmarks_data_dir, "ZTF22abvdwik.npz")


@pytest.fixture
def single_ztf_id():
    return "ZTF22abvdwik"
