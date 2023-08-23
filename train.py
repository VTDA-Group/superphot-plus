import os

from superphot_plus.classify_ztf import classify
from superphot_plus.file_paths import CM_FOLDER, MODELS_DIR, METRICS_DIR, FIT_PLOTS_FOLDER

if __name__ == "__main__":
    training_csv = "data/training_set.csv"

    num_epochs = 100
    num_folds = 5
    goal_per_class = 500
    neurons_per_layer = 128
    num_layers = 3
    log_file = "classification_log.txt"
    dynesty_fit_folder = "data/dynesty_fits"

    os.makedirs(CM_FOLDER, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIT_PLOTS_FOLDER, exist_ok=True)

    classify(
        [training_csv],
        dynesty_fit_folder,
        goal_per_class,
        num_epochs,
        neurons_per_layer,
        num_layers,
        log_file,
        num_folds=num_folds,
    )
