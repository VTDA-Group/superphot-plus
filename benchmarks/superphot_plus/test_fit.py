"""Benchmarks the available fitting methods."""

import os
import pytest

from superphot_plus.ztf_transient_fit import dynesty_single_file
from superphot_plus.fit_numpyro import numpyro_single_file
from superphot_plus.import_ztf_from_alerce import save_datafile, import_lc

OUTPUT_DIR = "benchmarks/data/"

fn_to_fit = os.path.join(OUTPUT_DIR, "ZTF22abvdwik.npz")


def test_dynesty_single_file():
    """Benchmarks the dynesty optimizer with nested sampling"""
    dynesty_single_file(fn_to_fit, OUTPUT_DIR, skip_if_exists=False)


def test_nuts_single_file():
    """Benchmarks the NUTS sampler"""
    numpyro_single_file(fn_to_fit, OUTPUT_DIR, sampler="NUTS")


def test_svi_single_file():
    """Benchmarks the svi sampler"""
    numpyro_single_file(fn_to_fit, OUTPUT_DIR, sampler="svi")
