from concurrent.futures import ProcessPoolExecutor
import json
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from superphot_plus.file_paths import MOSFIT_DIR, MOSFIT_INPUT_JSON
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.supernova_class import SupernovaClass
from superphot_plus.supernova_properties import SupernovaProperties
from superphot_plus.surveys.surveys import Survey
from superphot_plus.utils import convert_mags_to_flux


class MosfitDataGenerator:
    """Generates mosfit data using multi-core paralellization."""

    def __init__(self, mosfit_file, survey, num_realizations, num_workers, mosfit_dir):
        """Generates mosfit data using multi-core paralellization.

        Parameters
        ----------
        mosfit_file : str
            The path to the mosfit input JSON.
        survey : Survey
            The survey to which data belongs to.
        num_realizations : int
            Number of realizations to generate.
        num_workers : int
            Number of workers to run in parallel.
        mosfit_dir : str
            Directory where mosfit data is stored.
        """
        self.mosfit_file = mosfit_file
        self.survey = survey
        self.num_realizations = num_realizations
        self.num_workers = num_workers

        # Initialize output directories
        self.posteriors_dir = os.path.join(mosfit_dir, "posts")
        self.properties_dir = os.path.join(mosfit_dir, "props")
        os.makedirs(self.posteriors_dir, exist_ok=True)
        os.makedirs(self.properties_dir, exist_ok=True)

    def generate_data(self):
        """Generates data using multi-core processing."""

        # Read mosfit content
        data = self.read_mosfit_file()

        # Determine which realizations to generate
        realizations, skipped_realizations = self.get_realizations()
        print(f"Skipping {len(skipped_realizations)} realizations...")
        print(f"Generating {len(realizations)} missing realizations...")

        # Split realizations evenly between workers
        splits = np.array_split(realizations, self.num_workers)

        with ProcessPoolExecutor(int(self.num_workers)) as executor:
            for i, split in enumerate(splits):
                executor.submit(
                    self.generate_mosfit_data,
                    data=data,
                    realizations=split,
                    worker_id=i,
                )

    def read_mosfit_file(self):
        """Reads the mosfit data content from disk.

        Returns
        -------
        Dict
            The file content as a dictionary.
        """
        data = []
        with open(self.mosfit_file, encoding="utf-8") as slsn_file:
            data = json.load(slsn_file)
        top_key = [*data][0]
        return data[top_key]

    def get_realizations(self):
        """Determines which realizations should be generated.

        Parameters
        ----------
        num_realizations: int
            The maximum number of realizations to have.

        Returns
        -------
        tuple of np.array
            A tuple containing an array of missing realization names
            and an array of the realizations that have previously
            been generated and will be skipped.
        """
        missing_realizations = []
        skipped_realizations = []

        for realization in np.arange(1, self.num_realizations + 1):
            lc_name = self.format_realization_name(realization)
            posteriors_fn = f"{lc_name}_eqwt_svi.npz"

            properties_file = os.path.join(self.properties_dir, lc_name)
            posteriors_file = os.path.join(self.posteriors_dir, posteriors_fn)

            if os.path.exists(properties_file) and os.path.exists(posteriors_file):
                skipped_realizations.append(realization)
            else:
                missing_realizations.append(realization)

        return np.array(missing_realizations), np.array(skipped_realizations)

    def generate_mosfit_data(self, data, realizations, worker_id):
        """Generates the light curve posteriors and stores the mosfit dictionaries
        with the physical parameters for each.

        Parameters
        ----------
        data : Dict
            Dictionary containing the mosfit file content.
        realizations : np.array
            The array of realizations to generate.
        worker_id : int
            The worker identifier.
        """
        print(f"Worker {worker_id} has started")

        pbar = tqdm(realizations)

        for i, realization in enumerate(pbar):
            pbar.set_description(f"Worker {worker_id}")

            lc, properties = self.import_slsn_realization(data, realization=int(i + 1))

            sampler = NumpyroSampler()
            posterior_samples = sampler.run_single_curve(
                lc,
                rng_seed=4,
                priors=self.survey.priors,
                sampler="svi",
            )

            # Save posteriors
            posterior_samples.save_to_file(self.posteriors_dir)

            # Save properties
            realization_name = self.format_realization_name(realization)
            properties_file = os.path.join(self.properties_dir, realization_name)
            properties.write_to_file(properties_file)

    def import_slsn_realization(self, data, realization):
        """Imports SLSN specific data from mosfit.

        Parameters
        ----------
        data : Dict
            Dictionary containing the mosfit file content.
        realization : int
            The realization / light curve to import.

        Returns
        -------
        tuple
            The light curve and the respective properties.
        """
        t = []
        mag = []
        merr = []
        b = []

        for datum in data["photometry"]:
            if realization == int(datum["realization"]):
                # Ignore upper limits
                if "upperlimit" in datum:
                    continue
                t.append(float(datum["time"]))
                mag.append(float(datum["magnitude"]))
                merr.append(mag[-1] * float(datum["e_magnitude"]))
                b.append(datum["band"])

        # Also grab parameters...
        # Note that this will ALWAYS be the 0th model, so it is fine to hard code
        for my_realization in data["models"][0]["realizations"]:
            if int(my_realization["alias"]) == realization:
                bfield = my_realization["parameters"]["Bfield"]["value"]
                pspin = my_realization["parameters"]["Pspin"]["value"]
                mejecta = my_realization["parameters"]["mejecta"]["value"]
                vejecta = my_realization["parameters"]["vejecta"]["value"]

        t = np.asarray(t, dtype=float)
        mag = np.asarray(mag, dtype=float)
        merr = np.asarray(merr, dtype=float)
        b = np.asarray(b, dtype=str)

        flux, err = convert_mags_to_flux(mag, merr, self.survey.zero_point)
        err = err / np.max(flux)
        flux = flux / np.max(flux)

        lc = Lightcurve(
            t,
            flux,
            err,
            b,
            name=self.format_realization_name(realization),
            sn_class=SupernovaClass.SUPERLUMINOUS_SUPERNOVA_I,
        )

        properties = SupernovaProperties(
            bfield,
            pspin,
            mejecta,
            vejecta,
        )

        return lc, properties

    def format_realization_name(self, realization):
        """Determines realization name from its number.

        Parameters
        ----------
        realization : int
            The realization number.

        Returns
        -------
        str
            The name of the light curve realization.
        """
        return "lc_" + str(realization).zfill(6)  # extra zero for good measure


def extract_cmd_args():
    """Extracts the script command-line arguments."""
    parser = ArgumentParser(
        description="Parses mosfit data and generates posteriors and physical property files",
    )
    parser.add_argument(
        "--mosfit_file",
        help="The input JSON file containing mosfit data",
        default=MOSFIT_INPUT_JSON,
    )
    parser.add_argument(
        "--survey",
        help="Survey to which data belongs to",
        default=Survey.ZTF(),
    )
    parser.add_argument(
        "--num_realizations",
        help="The number of realizations to generate",
        default=10000,
    )
    parser.add_argument(
        "--mosfit_dir",
        help="Directory where mosfit data is stored",
        default=MOSFIT_DIR,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers for multi-core processing",
        default=3,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = extract_cmd_args()

    MosfitDataGenerator(
        mosfit_file=args.mosfit_file,
        survey=args.survey,
        num_realizations=int(args.num_realizations),
        num_workers=int(args.num_workers),
        mosfit_dir=args.mosfit_dir,
    ).generate_data()
