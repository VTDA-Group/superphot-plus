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


def read_mosfit_file(mosfit_file):
    """Reads the mosfit data content from disk.

    Parameters
    ----------
    mosfit_file :  str
        The mosfit file path.

    Returns
    -------
    Dict
        The file content as a dictionary.
    """
    data = []
    with open(mosfit_file, encoding="utf-8") as slsn_file:
        data = json.load(slsn_file)
    top_key = [*data][0]
    return data[top_key]


def format_realization_name(realization):
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


def import_slsn_realization(data, realization, survey):
    """Imports SLSN specific data from mosfit.

    Parameters
    ----------
    data : Dict
        Dictionary containing the mosfit file contents.
    realization : int
        The realization / light curve to import.
    survey : Survey
        Survey to which data belongs to.

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

    flux, err = convert_mags_to_flux(mag, merr, survey.zero_point)
    err = err / np.max(flux)
    flux = flux / np.max(flux)

    lc = Lightcurve(
        t,
        flux,
        err,
        b,
        name=format_realization_name(realization),
        sn_class=SupernovaClass.SUPERLUMINOUS_SUPERNOVA_I,
    )

    properties = SupernovaProperties(
        bfield,
        pspin,
        mejecta,
        vejecta,
    )

    return lc, properties


def generate_mosfit_data(mosfit_file, survey):
    """Generates the light curve posteriors and stores the mosfit dictionaries
    with the physical parameters for each.

    Parameters
    ----------
    mosfit_file : str
        The mosfit file path.
    survey : Survey
        The survey to which data belongs to.
    """
    # Read mosfit content
    data = read_mosfit_file(mosfit_file)

    for i, realization in enumerate(tqdm(realizations)):
        lc, properties = import_slsn_realization(data, realization=int(i + 1), survey=survey)

        sampler = NumpyroSampler()
        posterior_samples = sampler.run_single_curve(
            lc,
            rng_seed=4,
            priors=survey.priors,
            sampler="svi",
        )

        # Save posteriors
        posterior_samples.save_to_file(posteriors_dir)

        # Save properties
        realization_name = format_realization_name(realization)
        properties_file = os.path.join(properties_dir, realization_name)
        properties.write_to_file(properties_file)


def get_realizations(num_realizations):
    """Determines which realizations should be generated.

    Parameters
    ----------
    num_realizations: int
        The maximum number of realizations to have.

    Returns
    -------
    tuple
        A tuple containing a list of missing realization names
        and a list of the realizations that have previously
        been generated and will be skipped.
    """
    missing_realizations = []
    skipped_realizations = []

    for realization in np.arange(1, num_realizations + 1):
        lc_name = format_realization_name(realization)
        posteriors_fn = f"{lc_name}_eqwt_svi.npz"

        properties_file = os.path.join(properties_dir, lc_name)
        posteriors_file = os.path.join(posteriors_dir, posteriors_fn)

        if os.path.exists(properties_file) and os.path.exists(posteriors_file):
            skipped_realizations.append(realization)
        else:
            missing_realizations.append(realization)

    return missing_realizations, skipped_realizations


def extract_cmd_args():
    """Extracts the script command-line arguments."""
    parser = ArgumentParser(
        description="Parses mosfit data and generates posteriors and physical property files",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = extract_cmd_args()

    # Initialize data directories
    posteriors_dir = os.path.join(args.mosfit_dir, "posteriors")
    properties_dir = os.path.join(args.mosfit_dir, "properties")

    os.makedirs(posteriors_dir, exist_ok=True)
    os.makedirs(properties_dir, exist_ok=True)

    # Realizations to generate
    realizations, skipped = get_realizations(int(args.num_realizations))
    print(f"Skipping {len(skipped)} realizations...")
    print(f"Generating {len(realizations)} missing realizations...")

    generate_mosfit_data(mosfit_file=MOSFIT_INPUT_JSON, survey=args.survey)
