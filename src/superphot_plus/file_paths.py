"""A collection of paths corresponding to input and output files and 
directories.
"""

FITS_DIR = "/storage/group/vav5084/default/superphot+/dynesty_fits_unclassified_5_9_2023/"
DATA_DIRS = [
    "/storage/group/vav5084/default/superphot+/data_reformatted_05_09_2023",
]
input_csvs = ["../data/training_set_combined_05_09_2023.csv"]
CM_FOLDER = "./figs/hp_cm"
WRONGLY_CLASSIFIED_FOLDER = "./figs/wrongly_classified"
FIT_PLOTS_FOLDER = "./figs/fits_good"
CLASSIFY_LOG_FILE = "./temp_output.txt"

MODELS_DIR = "./models"
METRICS_DIR = "./figs/metrics"
PROBS_FILE = "probs_new.csv"
PROBS_FILE2 = "probs_phased.csv"

TRAINED_MODEL_FN = (  # glob.glob("models_saved/*ZTF23aagkgnz.pt")[0]
    "../../tests/data/superphot-model-ZTF23aagkgnz.pt"
)
TRAINED_CONFIG_FN = "../../tests/data/superphot-config-test.json"
