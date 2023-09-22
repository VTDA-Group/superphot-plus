import tempfile

import numpy as np

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.iminuit_sampler import IminuitSampler
from superphot_plus.samplers.licu_sampler import LiCuSampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.surveys.surveys import Survey

from .constants import SINGLE_ZTF_LIGHTCURVE_COMPRESSED


class SamplingSuite:
    """Benchmarks the available fitting methods."""

    def time_dynesty_single_file(self):
        """Benchmarks the dynesty optimizer with nested sampling"""
        sampler = DynestySampler()
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(
            lightcurve, priors=Survey.ZTF().priors, rstate=np.random.default_rng(9876)
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)

    def time_iminuit_single_file(self):
        """Benchmarks the iminuit optimizer"""
        sampler = IminuitSampler()
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(
            lightcurve, priors=Survey.ZTF().priors, rstate=np.random.default_rng(9876)
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)

    def time_licu_ceres_single_file(self):
        """Benchmarks the iminuit optimizer"""
        sampler = LiCuSampler(algorithm="ceres")
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(
            lightcurve, priors=Survey.ZTF().priors
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)

    def time_licu_mcmc_ceres_single_file(self):
        """Benchmarks the iminuit optimizer"""
        sampler = LiCuSampler(algorithm="mcmc-ceres", mcmc_niter=10_000)
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(
            lightcurve, priors=Survey.ZTF().priors
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)

    def time_nuts_single_file(self):
        """Benchmarks the NUTS sampler"""
        sampler = NumpyroSampler()
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(
            lightcurve, priors=Survey.ZTF().priors, rng_seed=None, sampler="NUTS"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)

    def time_svi_single_file(self):
        """Benchmarks the svi sampler"""
        sampler = NumpyroSampler()
        lightcurve = Lightcurve.from_file(SINGLE_ZTF_LIGHTCURVE_COMPRESSED)
        posteriors = sampler.run_single_curve(
            lightcurve, priors=Survey.ZTF().priors, rng_seed=None, sampler="svi"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            posteriors.save_to_file(tmp_dir)