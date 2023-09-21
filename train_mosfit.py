from argparse import ArgumentParser, BooleanOptionalAction

from superphot_plus.file_paths import MOSFIT_DIR
from superphot_plus.predictor import MosfitTrainer


def extract_cmd_args():
    """Extracts the script command-line arguments."""
    parser = ArgumentParser(
        description="Entry point to train and evaluate models using K-Fold cross validation",
    )
    parser.add_argument(
        "--parameter",
        help="Name of the physical parameter to train model on",
        required=True,
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
        mosfit_dir=args.mosfit_dir,
    )

    trainer.run(load_checkpoint=args.load_checkpoint)
