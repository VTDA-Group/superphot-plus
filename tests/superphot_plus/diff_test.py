import os
from pathlib import Path

import numpy as np

from superphot_plus.fit_numpyro import numpyro_single_file


def compare_result_files(goldens_file, new_results_file, delta=0.001):
    """Compare a golden file with a new result file.

    Parameters
    ----------
    goldens_file : str
        The path and filename of the golden results.
    new_results_file : str
        The path and filename of the new results.
    delta : float 
        The maximum absolute difference in numerical values.

    Returns
    -------
    boolean
        True if files are the same (within delta); False otherwise.
    """
    files_equal = True

    res_new = np.load(new_results_file)["arr_0"]
    print(f'Loaded {len(res_new)} rows from new file {new_results_file}.')
    res_old = np.load(goldens_file)["arr_0"]
    print(f'Loaded {len(res_old)} rows from new file {goldens_file}.')

    # Check that the number of results matches up
    if len(res_new) != len(res_old):
        print(f'Mismatched number of results ({len(res_new)} vs {len(res_old)} rows).')
        files_equal = False
    if res_new.size != res_old.size:
        print(f'Mismatched number of results ({res_new.size} vs {res_old.size} values).')
        files_equal = False

    # Check that the values are close to each other
    # This makes less sense for svi (we'd rather check to some mean than
    # row by row), but we're just using this as a placeholder for now)
    if not np.all(np.isclose(res_old, res_new, atol=delta)):
        print(
            f"{np.isclose(res_old, res_new, atol=delta).sum()} of {res_old.size} values mismatch. (max delta={delta})"
        )
        files_equal = False

    return files_equal


def test_diffs():
    """ Make golden files if they do not yet exist; otherwise, make new 
    results and compare with goldens.
    """
    lc_name = 'ZTF22abvdwik'
    suffix = '_eqwt_svi.npz'
    lc_data_path = Path('tests', 'data', f'{lc_name}.npz')
    goldens_dir = Path('tests', 'data', 'goldens')
    goldens_file = Path(goldens_dir, f'{lc_name}{suffix}')
    temp_results_dir = Path('tests', 'data', 'temp_results')
    temp_results_file = Path(temp_results_dir, f'{lc_name}{suffix}')

    # Check that goldens dir exists and that the files we'll use are there:
    if not goldens_dir.is_dir() or not goldens_file.is_file():
        print("Creating goldens...")
        if not goldens_dir.is_dir():
            os.makedirs(goldens_dir)
        numpyro_single_file(str(lc_data_path), goldens_dir, sampler='svi')
        print(f'Created goldens: {goldens_file}')

    # If we're comparing to the goldens this time:
    else:
        print("Comparing...")

        if not temp_results_dir.is_dir():
            os.makedirs(temp_results_dir)
        numpyro_single_file(str(lc_data_path), temp_results_dir, sampler='svi')

        compare_result_files(goldens_file, temp_results_file, 5.0)

        os.remove(temp_results_file)
        os.rmdir(temp_results_dir)

if __name__ == "__main__":
    test_diffs()
