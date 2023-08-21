"""Benchmarks the end-to-end inference task (sampling + classification)."""

from superphot_plus.classify_ztf import classify_single_light_curve
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.surveys.surveys import Survey


def test_dynesty_inference(classifier, single_ztf_id, single_ztf_lightcurve_compressed, tmp_path):
    """Benchmarks the inference task using the dynesty optimizer"""
    sampler = DynestySampler()
    lightcurve = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    posteriors = sampler.run_single_curve(lightcurve, priors=Survey.ZTF().priors)
    posteriors.save_to_file(tmp_path)

    classify_single_light_curve(classifier, single_ztf_id, tmp_path, "dynesty")


def test_numpyro_nuts_inference(classifier, single_ztf_id, single_ztf_lightcurve_compressed, tmp_path):
    """Benchmarks the inference task using the NUTS sampler"""
    sampler = NumpyroSampler()
    lightcurve = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    posteriors = sampler.run_single_curve(lightcurve, priors=Survey.ZTF().priors, sampler="NUTS")
    posteriors.save_to_file(tmp_path)

    classify_single_light_curve(classifier, single_ztf_id, tmp_path, "NUTS")


def test_numpyro_svi_inference(classifier, single_ztf_id, single_ztf_lightcurve_compressed, tmp_path):
    """Benchmarks the inference task using the svi sampler"""
    sampler = NumpyroSampler()
    lightcurve = Lightcurve.from_file(single_ztf_lightcurve_compressed)
    posteriors = sampler.run_single_curve(lightcurve, priors=Survey.ZTF().priors, sampler="svi")
    posteriors.save_to_file(tmp_path)

    classify_single_light_curve(classifier, single_ztf_id, tmp_path, "svi")
