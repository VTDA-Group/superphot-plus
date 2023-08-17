"""A collection of paths corresponding to input and output files and 
directories.
"""

DATA_DIR = "data"

FITS_DIR = "/storage/group/vav5084/default/superphot+/dynesty_fits_unclassified_5_9_2023/"
DATA_DIRS = [
    "/storage/group/vav5084/default/superphot+/data_reformatted_05_09_2023",
]
input_csvs = ["../data/training_set_combined_05_09_2023.csv"]

MODELS_DIR = f"{DATA_DIR}/models"
METRICS_DIR = f"{DATA_DIR}/figs/metrics"
PROBS_FILE = f"{DATA_DIR}/probs_new.csv"
PROBS_FILE2 = f"{DATA_DIR}/probs_phased.csv"

CM_FOLDER = f"{DATA_DIR}/figs/hp_cm"
WRONGLY_CLASSIFIED_FOLDER = f"{DATA_DIR}/figs/wrongly_classified"
FIT_PLOTS_FOLDER = f"{DATA_DIR}/figs/fits_good"
CLASSIFY_LOG_FILE = f"{DATA_DIR}/temp_output.txt"

TRAINED_MODEL_FN = (
    "../../tests/data/superphot-model-ZTF23aagkgnz.pt"  # glob.glob("models_saved/*ZTF23aagkgnz.pt")[0]
)
TRAINED_MODEL_CONF_FILE = "../../tests/data/superphot-config-test.json"
