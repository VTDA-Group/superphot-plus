"""Benchmarks the available fitting methods."""

import os

import numpy as np

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.surveys.surveys import Survey


def test_dynesty_single_file(single_ztf_lightcurve_compressed, tmp_path):
    """Benchmarks the dynesty optimizer with nested sampling"""
    sampler = DynestySampler()
    lightcurve = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    posteriors = sampler.run_single_curve(
        lightcurve, priors=Survey.ZTF().priors, rstate=np.random.default_rng(9876)
    )
    posteriors.save_to_file(tmp_path)


def test_nuts_single_file(single_ztf_lightcurve_compressed, tmp_path):
    """Benchmarks the NUTS sampler"""
    sampler = NumpyroSampler()
    lightcurve = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    posteriors = sampler.run_single_curve(lightcurve, priors=Survey.ZTF().priors, sampler="NUTS")
    posteriors.save_to_file(tmp_path)


def test_svi_single_file(single_ztf_lightcurve_compressed, tmp_path):
    """Benchmarks the svi sampler"""
    sampler = NumpyroSampler()
    lightcurve = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    posteriors = sampler.run_single_curve(lightcurve, priors=Survey.ZTF().priors, sampler="svi")
    posteriors.save_to_file(tmp_path)
