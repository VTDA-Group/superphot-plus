"""Entry point to train and evaluate models using K-Fold cross validation."""
from argparse import ArgumentParser, BooleanOptionalAction

from superphot_plus.classify_ztf import CrossValidationTrainer
from superphot_plus.file_paths import (
    CLASSIFY_LOG_FILE,
    INPUT_CSVS,
)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Entry point to train and evaluate models using K-Fold cross validation",
    )
    parser.add_argument(
        "--input_csvs",
        help="List of CSVs containing light curve data (comma separated)",
        default=",".join(INPUT_CSVS),
    )
    parser.add_argument(
        "--sampler",
        help="Name of the sampler to load fits from",
        choices=["dynesty", "nuts", "svi"],
        default="dynesty",
    )
    parser.add_argument(
        "--include_redshifts",
        help="If flag is set, include redshift data for training",
        default=False,
        action=BooleanOptionalAction,
    )
    parser.add_argument("--num_layers", help="Number of network hidden layers", default=3)
    parser.add_argument("--neurons_per_layer", help="Number of neurons per hidden layer", default=128)
    parser.add_argument("--num_epochs", help="Number of epochs for training", default=100)
    parser.add_argument("--num_folds", help="Number of folds for K-Fold cross validation", default=5)
    parser.add_argument("--goal_per_class", help="Number of samples per supernova class", default=500)
    parser.add_argument(
        "--classify_log_file",
        help="File to log classification results",
        default=CLASSIFY_LOG_FILE,
    )

    args = parser.parse_args()

    trainer = CrossValidationTrainer(
        num_layers=int(args.num_layers),
        neurons_per_layer=int(args.neurons_per_layer),
        goal_per_class=int(args.goal_per_class),
        classify_log_file=args.classify_log_file,
        sampler=args.sampler,
        include_redshift=args.include_redshifts,
    )

    trainer.run(
        input_csvs=args.input_csvs.split(","), num_epochs=int(args.num_epochs), num_folds=int(args.num_folds)
    )
