"""Entry point to regressor tuning using K-Fold cross validation."""
from argparse import ArgumentParser

from superphot_plus.file_paths import MOSFIT_DIR
from superphot_plus.load_data import read_mosfit_data
from superphot_plus.samplers.sampler import Sampler
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
        "--sampler",
        help="Name of the sampler to load fits from",
        choices=Sampler.CHOICES,
        default="dynesty",
    )
    parser.add_argument(
        "--mosfit_dir",
        help="Directory where mosfit data is stored",
        default=MOSFIT_DIR,
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
        help="Number of parameter combinations to try",
        default=10,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = extract_cmd_args()

    tuner = MosfitTuner(
        parameter=args.parameter,
        sampler=args.sampler,
        mosfit_dir=args.mosfit_dir,
        num_cpu=args.num_cpu,
        num_gpu=args.num_gpu,
    )

    data = read_mosfit_data(
        sampler=tuner.sampler,
        params_dir=tuner.params_dir,
        fits_dir=tuner.fits_dir,
    )

    tuner.run(
        data=data,
        num_hp_samples=int(args.num_hp_samples),
    )
