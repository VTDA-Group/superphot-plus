"""Entry point to model tuning using K-Fold cross validation."""
from argparse import ArgumentParser, BooleanOptionalAction

from superphot_plus.file_paths import INPUT_CSVS
from superphot_plus.samplers.sampler import Sampler
from superphot_plus.tuners.classifier_tuner import ClassifierTuner


def extract_cmd_args():
    """Extracts the script command-line arguments."""
    parser = ArgumentParser(
        description="Model tuning using K-Fold cross validation",
    )
    parser.add_argument(
        "--sampler",
        help="Name of the sampler to load fits from",
        choices=Sampler.CHOICES,
        default="dynesty",
    )
    parser.add_argument(
        "--include_redshift",
        help="If flag is set, include redshift data for training",
        default=True,
        action=BooleanOptionalAction,
    )
    parser.add_argument(
        "--num_cpu",
        help="Number of CPUs to use in each parallel experiment",
        default=2,
    )
    parser.add_argument(
        "--num_gpu",
        help="Number of GPUs to use in each parallel experiment",
        default=0,
    )
    parser.add_argument(
        "--input_csvs",
        help="List of CSVs containing light curve data (comma separated)",
        default=",".join(INPUT_CSVS),
    )
    parser.add_argument(
        "--num_hp_samples",
        help="Number of parameter combinations to try",
        default=10,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = extract_cmd_args()

    tuner = ClassifierTuner(
        sampler=args.sampler,
        include_redshift=args.include_redshift,
        num_cpu=args.num_cpu,
        num_gpu=args.num_gpu,
    )

    tuner.run(
        input_csvs=args.input_csvs.split(","),
        num_hp_samples=int(args.num_hp_samples),
    )
