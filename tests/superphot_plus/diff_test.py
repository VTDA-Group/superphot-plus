import os
from pathlib import Path

import numpy as np

from superphot_plus.fit_numpyro import numpyro_single_file


def compare_result_files(goldens_file, new_results_file, delta=0.001):
    """
    Compare two result files.

    Arguments:
         goldens_file - The path and filename of the golden results.
         new_results_file - The path and filename of the new results.
         delta - The maximum difference in numerical values.

    Returns:
         A Boolean indicating whether the files are the same.
    """
    files_equal = True

    res_new = np.load(new_results_file)["arr_0"]
    print("Loaded %i new results from %s." % (len(res_new), new_results_file))
    res_old = np.load(goldens_file)["arr_0"]
    print("Loaded %i old results from %s." % (len(res_old), goldens_file))

    # Check that the number of results matches up.
    if len(res_new) != len(res_old):
        print("Mismatched number of results (%i vs %i)." % (len(res_old), len(res_new)))
        files_equal = False
    
    # Check that the values are close to each other.
    # This makes less sense for svi (we'd rather check to some mean than
    # row by row), but we're just using this as a placeholder for now)
    tolerance = 7.5
    if not np.all(np.isclose(res_old, res_new, rtol=tolerance)):
        print(f'{np.isclose(res_old, res_new, rtol=tolerance).sum()} of {res_old.size} rows mismatch.')
        files_equal = False

    return files_equal

#if __name__ == "__main__":
def test_diffs():
    """ Make golden files and compare"""
    print('\nRunning diff...')

    lc_name = 'ZTF22abvdwik'
    suffix = '_eqwt_svi.npz'
    lc_data_path = Path('tests', 'data', f'{lc_name}.npz')
    goldens_dir = Path('tests', 'data', 'goldens')
    temp_diffs_dir = Path('tests', 'data', 'temp_diffs')

    # Check that goldens dir exists and that the files we'll use are there:
    if not goldens_dir.is_dir(): # TODO check files exist
        print("Making goldens...")
        os.makedirs(goldens_dir)
        numpyro_single_file(str(lc_data_path), goldens_dir, sampler="svi")

    # If we're comparing to the goldens this time:
    else:
        print("Comparing...")
        
        # Make new file to diff against goldens
        if not temp_diffs_dir.is_dir():
            os.makedirs(temp_diffs_dir)
        
        numpyro_single_file(str(lc_data_path), temp_diffs_dir, sampler="svi")

        # Run comparison
        golden_file = Path(goldens_dir, f'{lc_name}{suffix}')
        temp_diff_file = Path(temp_diffs_dir, f'{lc_name}{suffix}')
        compare_result_files(golden_file, temp_diff_file)

    print("Done.")

if __name__ == "__main__":
    test_diffs()
