import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from os import urandom

import numpy as np
from tqdm import tqdm

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.iminuit_sampler import IminuitSampler
from superphot_plus.samplers.licu_sampler import LiCuSampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.surveys.surveys import Survey


class PosteriorsGenerator:
    """Generates posterior samples using multi-core parallelization."""

    def __init__(self, sampler_name, lightcurves_dir, survey, num_workers, output_dir):
        """Generates posterior samples using multi-core parallelization.

        Parameters
        ----------
        sampler_name : str
            The method used for fitting.
        lightcurves_dir : str
            Directory where light curve CSV data is stored.
        survey : Survey
            The survey to which data belongs to.
        num_workers : int
            Number of workers to run in parallel.
        output_dir : str
            Base directory for classification outputs.
        """
        self.sampler_name = sampler_name
        self.lightcurves_dir = lightcurves_dir
        self.survey = Survey.ZTF() if survey == "ZTF" else Survey.LSST()
        self.num_workers = num_workers

        # Initialize posteriors directory
        self.posteriors_dir = os.path.join(output_dir, f"{sampler_name}_fits")
        os.makedirs(self.posteriors_dir, exist_ok=True)

    def generate_data(self, seed):
        """Distributes data generation between available workers.

        Parameters
        ----------
        seed : int
            Random seed value for deterministic data generation.
        """
        # Determine which posterior files to generate
        posteriors = self.get_posteriors_to_generate()

        # Split realizations evenly between workers
        splits = np.array_split(posteriors, self.num_workers)

        # Initialize sampler
        sampler, kwargs = self.setup_sampler(self.sampler_name, seed)

        with ProcessPoolExecutor(self.num_workers) as executor:
            for i, split in enumerate(splits):
                executor.submit(
                    self.run_sampler,
                    sampler=sampler,
                    kwargs=kwargs,
                    lightcurves=split,
                    worker_id=i,
                )

    def get_posteriors_to_generate(self):
        """Determines which fit files to generate.

        Returns
        -------
        list of str
            The file names of the missing posterior samples.
        """
        lightcurve_files = os.listdir(self.lightcurves_dir)
        generated_posteriors = os.listdir(self.posteriors_dir)

        missing_posteriors = [
            f for f in lightcurve_files if self.get_posteriors_fn(f) not in generated_posteriors
        ]
        print(f"Skipping {len(generated_posteriors)} realizations...")
        print(f"Generating {len(missing_posteriors)} posterior samples...")

        return missing_posteriors

    def setup_sampler(self, sampler_name, seed):
        """Creates a sampler and its kwargs from its name.

        Parameter
        ---------
        sampler_name : str
            The name of the sampler to use. One of "dynesty", "svi",
            "NUTS", "iminuit", "licu-ceres" or "licu-mcmc-ceres".
        seed : int
            Random seed value used for deterministic data generation.

        Returns
        -------
        sampler : Sampler
            The sampler object.
        kwargs : dict
            The sampler specific arguments.
        """
        kwargs = {}

        kwargs["priors"] = self.survey.priors

        if sampler_name == "dynesty":
            sampler_obj = DynestySampler()
        elif sampler_name == "svi":
            sampler_obj = NumpyroSampler()
            kwargs["sampler"] = "svi"
        elif sampler_name == "NUTS":
            sampler_obj = NumpyroSampler()
            kwargs["sampler"] = "NUTS"
        elif sampler_name == "iminuit":
            sampler_obj = IminuitSampler()
        elif sampler_name == "licu-ceres":
            sampler_obj = LiCuSampler(algorithm="ceres")
        elif sampler_name == "licu-mcmc-ceres":
            sampler_obj = LiCuSampler(algorithm="mcmc-ceres", mcmc_niter=10_000)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")

        kwargs["rng_seed"] = seed

        return sampler_obj, kwargs

    def run_sampler(self, sampler, kwargs, lightcurves, worker_id):
        """Runs fitting for a set of light curves.

        Parameters
        ----------
        sampler : Sampler
            The sampler object.
        kwargs : dict
            The sampler specific arguments.
        lightcurves : list
            The list of light curve file names.
        worker_id : int
            The worker identifier.
        """
        print(f"Worker {worker_id} has started")

        pbar = tqdm(lightcurves)
        pbar.set_description(f"Worker {worker_id}")

        for lc_name in pbar:
            file = os.path.join(self.lightcurves_dir, lc_name)
            lightcurve = Lightcurve.from_file(file)
            posteriors = sampler.run_single_curve(lightcurve, **kwargs)
            posteriors.save_to_file(self.posteriors_dir)

    def get_posteriors_fn(self, filename):
        """Returns the posteriors filename for a light curve and sampler.

        Parameters
        ----------
        filename : str
            The name of the light curve file.

        Returns
        -------
        str
            The name of the posterior samples file.
        """
        return f"{os.path.splitext(filename)[0]}_eqwt_{self.sampler_name}.npz"


def extract_cmd_args():
    """Extracts the script command-line arguments."""
    parser = ArgumentParser(
        description="Parses mosfit data and generates posteriors and physical property files",
    )
    parser.add_argument(
        "--output_dir",
        help="Base directory for classification outputs",
    )
    parser.add_argument(
        "--lightcurves_dir",
        help="Directory where light curve CSV data is stored",
    )
    parser.add_argument(
        "--sampler",
        help="The sampler to use for fitting",
        choices=[
            "dynesty",
            "svi",
            "NUTS",
            "iminuit",
            "licu-ceres",
            "licu-mcmc-ceres",
        ],
        default="dynesty",
    )
    
    parser.add_argument(
        "--survey",
        help="Survey to which data belongs to",
        choices=["ZTF", "LSST"],
        default="ZTF",
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers for multi-core processing",
        default=3,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = extract_cmd_args()

    PosteriorsGenerator(
        sampler_name=args.sampler,
        lightcurves_dir=args.lightcurves_dir,
        survey=args.survey,
        num_workers=int(args.num_workers),
        output_dir=args.output_dir,
    ).generate_data(seed=int.from_bytes(urandom(4), "big"))
