import json
import os
import numpy as np

from superphot_plus.file_paths import MOSFIT_DIR, MOSFIT_INPUT_JSON
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.surveys.surveys import Survey

from superphot_plus.supernova_properties import SupernovaProperties


def import_slsn_mosfit(mosfit_fn, realization):
    """This is specifically for SLSN code, since we happen to know
    which parameters actually matter."""
    data = []

    with open(mosfit_fn, encoding="utf-8") as slsn_file:
        data = json.load(slsn_file)

    top_key = [*data][0]

    t = []
    flux = []
    err = []
    b = []

    for datum in data[top_key]["photometry"]:
        if realization == int(datum["realization"]):
            # Ignore upper limits
            if "upperlimit" in datum:
                continue
            t.append(float(datum["time"]))
            flux.append(10.0 ** ((float(datum["magnitude"]) + 48.6) / -2.5))
            err.append(flux[-1] * float(datum["e_magnitude"]))
            b.append(datum["band"])

    # Also grab parameters...
    # Note that this will ALWAYS be the 0th model, so it is fine to hard code
    for my_realization in data[top_key]["models"][0]["realizations"]:
        if int(my_realization["alias"]) == realization:
            bfield = my_realization["parameters"]["Bfield"]["value"]
            pspin = my_realization["parameters"]["Pspin"]["value"]
            mejecta = my_realization["parameters"]["mejecta"]["value"]
            vejecta = my_realization["parameters"]["vejecta"]["value"]

    t = np.asarray(t, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    b = np.asarray(b, dtype=str)

    return t, flux, err, b, bfield, pspin, mejecta, vejecta


def generate_mosfit_data(data_file, out_dir, num_realizations):
    """Generates the light curve posteriors and stores the mosfit dictionaries
    with the physical parameters for each.

    There are 10000 lightcurves:
        -> ~40MB mosfit files
        -> ~120MB posterior files (SVI)
    """
    posteriors_out_dir = os.path.join(out_dir, "posteriors")
    properties_out_dir = os.path.join(out_dir, "params")

    os.makedirs(posteriors_out_dir, exist_ok=True)
    os.makedirs(properties_out_dir, exist_ok=True)

    for realization in np.arange(1, num_realizations + 1):
        lc_name = f"lc_{realization}"
        mosfit_file = os.path.join(properties_out_dir, lc_name)

        # Skip existent files
        if os.path.exists(mosfit_file):
            print(f"Skipping realization {realization}")
            continue

        times, flux, err, bands, bfield, pspin, mejecta, vejecta = import_slsn_mosfit(
            data_file, realization=int(realization)
        )
        err = err / np.max(flux)
        flux = flux / np.max(flux)

        lc = Lightcurve(
            times,
            flux,
            err,
            bands,
            name=lc_name,
        )

        sampler = NumpyroSampler()
        posterior_samples = sampler.run_single_curve(
            lc,
            rng_seed=4,
            priors=Survey.ZTF().priors,
            sampler="svi",
        )
        posterior_samples.save_to_file(posteriors_out_dir)

        mosfit = SupernovaProperties(
            bfield,
            pspin,
            mejecta,
            vejecta,
        )
        mosfit.write_to_file(mosfit_file)

        print(f"Progress {(realization/num_realizations*100):.02f}%")


if __name__ == "__main__":
    generate_mosfit_data(
        data_file=MOSFIT_INPUT_JSON,
        out_dir=MOSFIT_DIR,
        num_realizations=1000,
    )
