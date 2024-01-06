"""Runs diffing to make sure results have not been changed.

This includes test_diffs, which will automatically check new results 
against pre-generated goldens file, and interactive_test_diffs, which 
can be run from the command line to interactively walk user through
prompts to:
1. force regeneration of golden files
2. generate new results files and compare to goldens
3. remove new results files

Borrows from KBMOD's diff test:
https://github.com/dirac-institute/kbmod/blob/main/tests/diff_test.py
"""

import glob
import math
import os
import time
from pathlib import Path

import numpy as np

from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.samplers.dynesty_sampler import DynestySampler
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler
from superphot_plus.surveys.surveys import Survey


def check_goldens_exist(goldens_dir):
    """Checks that goldens directory exists and contains the expected files.

    Parameters
    ----------
    goldens_dir : pathlib Path
        The path of the goldens directory.

    Returns
    -------
    boolean
        True if directory and expected files exist; False otherwise."""
    if not goldens_dir.is_dir():
        return False
    has_dynesty = False
    has_nuts = False
    has_svi = False
    for file_or_dir in os.listdir(goldens_dir):
        if "dynesty" in file_or_dir:
            has_dynesty = True
        if "NUTS" in file_or_dir:
            has_nuts = True
        if "svi" in file_or_dir:
            has_svi = True
    return has_dynesty and has_nuts and has_svi


def generate_files(lightcurve, output_dir):
    """Generates new files in the given directory for dynesty, NUTS, and svi.

    Parameters
    ----------
    lightcurve : Lightcurve
        The lightcurve data we are using in file generation.
    output_dir : pathlib Path
        The directory where we will save the files.
    """
    print("Generating files", end="...")

    # Make sure our target directory exists
    if not output_dir.is_dir():
        os.makedirs(output_dir)

    # Set up
    priors = Survey.ZTF().priors

    # Generate dynesty
    sampler = DynestySampler()
    posteriors = sampler.run_single_curve(lightcurve, priors=priors, rstate=np.random.default_rng(9876))
    posteriors.save_to_file(output_dir)

    # Generate NUTS
    sampler = NumpyroSampler(sampler="NUTS")
    posteriors = sampler.run_single_curve(lightcurve, rng_seed=4, priors=priors)
    posteriors.save_to_file(output_dir)

    # Generate svi
    sampler = NumpyroSampler(sampler="svi")
    posteriors = sampler.run_single_curve(lightcurve, rng_seed=1, priors=priors)
    posteriors.save_to_file(output_dir)

    print(f"saved files to {output_dir}.")


def compare_directories(goldens_dir, temp_results_dir):
    """Compares files in goldens directory to files in temp results directory.

    Parameters
    ----------
    goldens_dir : pathlib Path
        The directory containing our goldens.
    temp_results_dir : pathlib Path
        The directory containing the temporary results files we've just generated.

    Returns
    -------
    boolean
        True if files in directories are sufficiently similar; False otherwise.
    """
    no_differences_found = True

    for file_name in os.listdir(goldens_dir):
        # Make sure the file is in both directories
        if file_name not in os.listdir(temp_results_dir):
            raise ValueError(
                f"Files improperly generated in new results directory; could not find {file_name}"
            )
        # Compare
        if not compare_two_files(file_name, goldens_dir, temp_results_dir):
            no_differences_found = False

    return no_differences_found


def compare_two_files(file_name, goldens_dir, temp_results_dir):
    """Compares similarity of sample means.

    Notes
    -----
        This function specifies different deltas (relative tolerance) for each
        sampling method.

        Also, we choose not to compare file sizes here, as dynesty produces
        different result sizes on different machines.

    Parameters
    ----------
    file_name : str
        The file to compare (a file of this name should exist in each dir).
    goldens_dir : pathlib Path
        The directory containing our goldens.
    temp_results_dir : pathlib Path
        The directory containing the temporary results files we've just generated.

    Returns
    -------
    boolean
        True if files are sufficiently similar; False otherwise."""

    # Set up
    deltas = {"dynesty": 0.25, "svi": 0.25, "NUTS": 0.25}
    no_differences_found = True

    # Compare sample means
    lightcurve_name = file_name.split("_")[0]
    sampling_method = file_name.split("_")[-1].split(".")[0]

    goldens_samples = PosteriorSamples.from_file(
        input_dir=goldens_dir, name=lightcurve_name, sampling_method=sampling_method
    )
    temp_results_samples = PosteriorSamples.from_file(
        input_dir=temp_results_dir,
        name=lightcurve_name,
        sampling_method=sampling_method,
    )

    print(sampling_method, end=": ")
    for index, golden_val in enumerate(goldens_samples.sample_mean()):
        temp_val = temp_results_samples.sample_mean()[index]
        print(temp_val, golden_val)

        print(f"{(abs( 1 - (golden_val / temp_val))):.2f}", end=", ")
        if not math.isclose(golden_val, temp_val, rel_tol=deltas[sampling_method], abs_tol=0.1):
            no_differences_found = False
            print(
                f"\nSignificantly distant values in sample means of {file_name} at index {index} "
                f"(and possibly later)."
                f"\nDelta (relative tolerance) of {sampling_method} = {deltas[sampling_method]}."
                f"\nGoldens sample mean: {goldens_samples.sample_mean()}"
                f"\nNew results sample mean: {temp_results_samples.sample_mean()}"
                f"\n"
            )
            break

    print(f"Checked {file_name} (delta={deltas[sampling_method]}).")

    return no_differences_found


def delete_temp_files(temp_dir):
    """Deletes all files in the given directory, then deletes the directory itself.

    Parameters
    ----------
    temp_dir : pathlib Path
        The temporary directory where we've just generated our new results.
    """
    if "temp" not in temp_dir.as_posix() and "tmp" not in temp_dir.as_posix():
        raise ValueError("Attempted to delete directory that is not designated with 'temp' or 'tmp'.")
    for file in os.listdir(temp_dir):
        os.remove(Path(temp_dir, file))
    os.rmdir(temp_dir)


def test_diffs(test_data_dir, tmp_path):
    """Check new results against pre-generated goldens file.

    This is the function that is automatically run by pytest.
    """
    # data_dir = Path("tests", "data")
    goldens_dir = Path(test_data_dir, "goldens")

    lightcurve_name = "ZTF22abvdwik"
    lightcurve = Lightcurve.from_file(Path(test_data_dir, lightcurve_name + ".npz").as_posix())

    assert check_goldens_exist(goldens_dir)
    generate_files(lightcurve, tmp_path)
    assert compare_directories(goldens_dir, tmp_path)


def interactive_test_diffs():
    """Tests diffs interactively, including a prompt to force regeneration of goldens."""
    data_dir = Path("..", "data")
    goldens_dir = Path(data_dir, "goldens")
    temp_dir = Path(data_dir, "temp_diff")

    lightcurve_name = "ZTF22abvdwik"
    lightcurve = Lightcurve.from_file(Path(data_dir, lightcurve_name + ".npz").as_posix())

    # Check and print goldens info
    if check_goldens_exist(goldens_dir):
        list_of_goldens = glob.glob(f"{goldens_dir}/*")
        last_modified_golden = max(list_of_goldens, key=os.path.getmtime)
        last_modified_time = time.ctime(os.path.getmtime(last_modified_golden))
        print(f"Goldens exist and were most recently modified {last_modified_time}.")
    else:
        print("Goldens are missing or incomplete.")

    # Prompt: regen goldens?
    do_regen = input("Force regenerate goldens? [y/N] ")
    if do_regen == "y" or do_regen == "Y":
        print("Regeneratng goldens...")
        generate_files(lightcurve, goldens_dir)

    # Prompt: run comparison?
    do_comparison = input("Generate new results and compare to goldens? [Y/n] ")
    if not do_comparison == "n" and not do_comparison == "N":
        print("Generating new results and comparing...")
        generate_files(lightcurve, temp_dir)
        compare_directories(goldens_dir, temp_dir)

    # Prompt: delete temp files?
    do_delete_generated_files = input("Delete just-generated results files? [Y/n] ")
    if not do_delete_generated_files == "n" and not do_delete_generated_files == "N":
        print("Deleting just-generated files...")
        delete_temp_files(temp_dir)


if __name__ == "__main__":
    interactive_test_diffs()
