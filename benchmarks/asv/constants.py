# pylint: disable=missing-function-docstring

import os

BENCH_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SINGLE_ZTF_ID = "ZTF22abvdwik"
FIT_FILE = f"{SINGLE_ZTF_ID}.npz"
SINGLE_ZTF_LIGHTCURVE_COMPRESSED = os.path.join(BENCH_DATA_DIR, FIT_FILE)

CLASSIFIER_FILE = os.path.join(BENCH_DATA_DIR, "superphot-model-ZTF23aagkgnz.pt")
CLASSIFIER_CONF = os.path.join(BENCH_DATA_DIR, "superphot-config-test.yaml")
