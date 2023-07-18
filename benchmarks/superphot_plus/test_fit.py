"""Benchmarks the available fitting methods."""

import os
import pytest

from superphot_plus.ztf_transient_fit import dynesty_single_file
from superphot_plus.fit_numpyro import numpyro_single_file
from superphot_plus.import_ztf_from_alerce import (
    generate_single_flux_file,
    save_datafile,
    import_lc,
)

from time import sleep

OUTPUT_DIR = "benchmarks/data/"

test_sn = "ZTF22abvdwik"  # can change to any ZTF supernova
lc_fn = os.path.join(OUTPUT_DIR, test_sn + ".csv")
fn_to_fit = os.path.join(OUTPUT_DIR, test_sn + ".npz")


@pytest.fixture(scope="session", autouse=True)
def setup():
    """Generates sample data file for benchmarking purposes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_single_flux_file(test_sn, OUTPUT_DIR)
    # Save the preprocessed lightcurves
    t, f, ferr, b, _, _ = import_lc(lc_fn)
    save_datafile(test_sn, t, f, ferr, b, OUTPUT_DIR)


def test_dynesty_single_file():
    """Uses the dynesty optimizer with nested sampling"""
    sleep(60)
    dynesty_single_file(fn_to_fit, OUTPUT_DIR, skip_if_exists=False)


def test_nuts_single_file():
    """Uses the NUTS sampler"""
    sleep(60)
    numpyro_single_file(fn_to_fit, OUTPUT_DIR, sampler="NUTS")


def test_svi_single_file():
    """Uses the svi sampler"""
    numpyro_single_file(fn_to_fit, OUTPUT_DIR, sampler="svi")
