"""Entry point to regressor training and evaluation."""
from argparse import ArgumentParser, BooleanOptionalAction

from superphot_plus.file_paths import MOSFIT_DIR
from superphot_plus.load_data import read_mosfit_data
from superphot_plus.samplers.sampler import Sampler
from superphot_plus.trainers.mosfit_trainer import MosfitTrainer


def extract_cmd_args():
    """Extracts the script command-line arguments."""
    parser = ArgumentParser(
        description="Entry point to train and evaluate models using K-Fold cross validation",
    )
    parser.add_argument(
        "--parameter",
        help="Name of the physical property to train model on",
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
        "--load_checkpoint",
        help="If set, load pretrained model for the respective configuration",
        default=False,
        action=BooleanOptionalAction,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = extract_cmd_args()

    trainer = MosfitTrainer(
        parameter=args.parameter,
        sampler=args.sampler,
        mosfit_dir=args.mosfit_dir,
    )
    data = read_mosfit_data(
        sampler=trainer.sampler,
        params_dir=trainer.params_dir,
        fits_dir=trainer.fits_dir,
    )
    trainer.run(
        data=data,
        load_checkpoint=args.load_checkpoint,
    )
