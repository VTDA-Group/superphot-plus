"""
Entry point to tune, train and evaluate models. Tuning uses K-Fold
cross validation to estimate model performance.
"""
from argparse import ArgumentParser, BooleanOptionalAction

from superphot_plus.trainer import CrossValidationTrainer
from superphot_plus.file_paths import INPUT_CSVS, PROBS_FILE

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
        "--model_name",
        help="If set, the trainer loads this model and skips tuning",
        default=None,
    )
    parser.add_argument(
        "--sampler",
        help="Name of the sampler to load fits from",
        choices=["dynesty", "nuts", "svi"],
        default="dynesty",
    )
    parser.add_argument(
        "--include_redshift",
        help="If flag is set, include redshift data for training",
        default=False,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--num_hp_samples",
        help="Name of parameter combinations to try",
        default=10,
    )
    parser.add_argument(
        "--extract_wc",
        help="If flag is set, extract wrongly classified samples",
        default=False,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--probs_file",
        help="File to log test probability results",
        default=PROBS_FILE,
    )

    args = parser.parse_args()

    trainer = CrossValidationTrainer(
        model_name=args.model_name,
        sampler=args.sampler,
        include_redshift=args.include_redshift,
        probs_file=args.probs_file,
    )

    trainer.run(
        input_csvs=args.input_csvs.split(","),
        num_hp_samples=args.num_hp_samples,
        extract_wc=args.extract_wc,
    )
