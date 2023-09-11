import tempfile

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.surveys.surveys import Survey

from .constants import CLASSIFIER_CONF, CLASSIFIER_FILE, SINGLE_ZTF_ID, SINGLE_ZTF_LIGHTCURVE_COMPRESSED


class InferenceSuite:
    """Benchmarks the end-to-end inference task (sampling + classification)."""

    def setup_cache(self):
        """Warms up the suite, loading the classifier."""
        return SuperphotClassifier.load(CLASSIFIER_FILE, CLASSIFIER_CONF)[0]

    def time_dynesty_inference(self, classifier):
        """Benchmarks the inference task using the dynesty optimizer"""
        sampler = DynestySampler()
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(lightcurve, priors=Survey.ZTF().priors)

        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)
            classifier.classify_single_light_curve(SINGLE_ZTF_ID, tmp_dir, "dynesty")

    def time_numpyro_nuts_inference(self, classifier):
        """Benchmarks the inference task using the NUTS sampler"""
        sampler = NumpyroSampler()
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(
            lightcurve, priors=Survey.ZTF().priors, rng_seed=None, sampler="NUTS"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)
            classifier.classify_single_light_curve(SINGLE_ZTF_ID, tmp_dir, "NUTS")

    def time_numpyro_svi_inference(self, classifier):
        """Benchmarks the inference task using the svi sampler"""
        sampler = NumpyroSampler()
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(
            lightcurve, priors=Survey.ZTF().priors, rng_seed=None, sampler="svi"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)
            classifier.classify_single_light_curve(SINGLE_ZTF_ID, tmp_dir, "svi")
