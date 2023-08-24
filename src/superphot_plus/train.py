"""Entrypoint to train and evaluate models using K-Fold cross validation."""
import os

from argparse import ArgumentParser
from superphot_plus.classify_ztf import classify
from superphot_plus.file_paths import (
    CLASSIFY_LOG_FILE,
    CM_FOLDER,
    DATA_DIR,
    INPUT_CSVS,
    MODELS_DIR,
    METRICS_DIR,
    FIT_PLOTS_FOLDER,
)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Entrypoint to train and evaluate models using K-Fold cross validation",
    )

    parser.add_argument(
        "--input_csvs",
        help="List of CSVs containing light curve data (comma separated)",
        default=",".join(INPUT_CSVS),
    )
    parser.add_argument(
        "--sampler", help="Name of the sampler to use", choices=["dynesty", "nuts", "svi"], default="dynesty"
    )
    parser.add_argument("--num_layers", help="Number of network hidden layers", default=3)
    parser.add_argument("--neurons_per_layer", help="Number of neurons per hidden layer", default=128)
    parser.add_argument("--num_epochs", help="Number of epochs for training", default=100)
    parser.add_argument("--num_folds", help="Number of folds for K-Fold cross validation", default=5)
    parser.add_argument("--goal_per_class", help="Number of samples per supernova class", default=500)
    parser.add_argument(
        "--log_file",
        help="File to log classification results",
        default=CLASSIFY_LOG_FILE,
    )

    args = parser.parse_args()

    os.makedirs(CM_FOLDER, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIT_PLOTS_FOLDER, exist_ok=True)

    classify(
        input_csvs=args.input_csvs.split(","),
        fit_dir=f"{DATA_DIR}/{args.sampler}_fits",
        goal_per_class=args.goal_per_class,
        num_epochs=args.num_epochs,
        num_layers=args.num_layers,
        neurons_per_layer=args.neurons_per_layer,
        classify_log_file=args.log_file,
        num_folds=args.num_folds,
    )
