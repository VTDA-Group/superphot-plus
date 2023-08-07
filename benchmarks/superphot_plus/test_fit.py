"""Benchmarks the available fitting methods."""

import os

import numpy as np

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.priors.fitting_priors import MultibandPriors
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler

OUTPUT_DIR = "benchmarks/data/"

fn_to_fit = os.path.join(OUTPUT_DIR, "ZTF22abvdwik.npz")


def test_dynesty_single_file():
    """Benchmarks the dynesty optimizer with nested sampling"""
    sampler = DynestySampler()
    lightcurve = Lightcurve.from_file(fn_to_fit)
    posteriors = sampler.run_single_curve(
        lightcurve, priors=MultibandPriors.load_ztf_priors(), rstate=np.random.default_rng(9876)
    )
    posteriors.save_to_file(OUTPUT_DIR)


def test_nuts_single_file():
    """Benchmarks the NUTS sampler"""
    sampler = NumpyroSampler()
    lightcurve = Lightcurve.from_file(fn_to_fit)
    posteriors = sampler.run_single_curve(
        lightcurve, priors=MultibandPriors.load_ztf_priors(), sampler="NUTS"
    )
    posteriors.save_to_file(OUTPUT_DIR)


def test_svi_single_file():
    """Benchmarks the svi sampler"""
    sampler = NumpyroSampler()
    lightcurve = Lightcurve.from_file(fn_to_fit)
    posteriors = sampler.run_single_curve(lightcurve, priors=MultibandPriors.load_ztf_priors(), sampler="svi")
    posteriors.save_to_file(OUTPUT_DIR)
