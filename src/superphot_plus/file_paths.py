"""A collection of paths corresponding to input and output files and directories."""

DATA_DIR = "data"

FITS_DIR = "/storage/group/vav5084/default/superphot+/dynesty_fits_unclassified_5_9_2023/"
DATA_DIRS = [
    "/storage/group/vav5084/default/superphot+/data_reformatted_05_09_2023",
]
INPUT_CSVS = [f"{DATA_DIR}/training_set.csv"]  # ["../data/training_set_combined_05_09_2023.csv"]
MOSFIT_INPUT_JSON = f"{DATA_DIR}/slsn.json"

# Base directories
CLASSIFICATION_DIR = f"{DATA_DIR}/classification"
MOSFIT_DIR = f"{DATA_DIR}/mosfit"

# Directories to store classification metrics
METRICS_DIR = f"{CLASSIFICATION_DIR}/figs/metrics"
FIT_PLOTS_FOLDER = f"{CLASSIFICATION_DIR}/figs/fits_good"
CM_FOLDER = f"{CLASSIFICATION_DIR}/figs/hp_cm"
WRONGLY_CLASSIFIED_FOLDER = f"{CLASSIFICATION_DIR}/figs/wrongly_classified"

# Directory to store classification models
MODELS_DIR = f"{CLASSIFICATION_DIR}/models"

# Classification output files
CLASSIFY_LOG_FILE = f"{CLASSIFICATION_DIR}/classification_log.txt"
PROBS_FILE = f"{CLASSIFICATION_DIR}/probs_new.csv"
PROBS_FILE2 = f"{CLASSIFICATION_DIR}/probs_phased.csv"

# Pretrained model
# glob.glob("models_saved/*ZTF23aagkgnz.pt")[0]
TRAINED_MODEL_FN = "../../tests/data/superphot-model-ZTF23aagkgnz.pt"
TRAINED_CONFIG_FN = "../../tests/data/superphot-config-test.yaml"
