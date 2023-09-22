"""Entry point to model tuning using K-Fold cross validation."""
from argparse import ArgumentParser

from superphot_plus.tuners.mosfit_tuner import MosfitTuner


def extract_cmd_args():
    """Extracts the script command-line arguments."""
    parser = ArgumentParser(
        description="Model tuning using K-Fold cross validation",
    )
    parser.add_argument(
        "--parameter",
        help="Name of the physical property to tune model on",
        required=True,
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
        "--num_hp_samples",
        help="Name of parameter combinations to try",
        default=10,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = extract_cmd_args()

    tuner = MosfitTuner(
        parameter=args.parameter,
        num_cpu=args.num_cpu,
        num_gpu=args.num_gpu,
    )

    tuner.run(num_hp_samples=int(args.num_hp_samples))
