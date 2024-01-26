"""Entry point to model training and evaluation."""
from argparse import ArgumentParser, BooleanOptionalAction

from superphot_plus.config import SuperphotConfig
from superphot_plus.trainer import SuperphotTrainer

import numpy as np

def extract_cmd_args():
    """Extracts the script command-line arguments."""
    default_config = SuperphotConfig(create_dirs=False)
    parser = ArgumentParser(
        description="Entry point to train and evaluate models using K-Fold cross validation",
    )
    parser.add_argument(
        "--input_csvs",
        help="List of CSVs containing light curve data (comma separated)",
        default=",".join(default_config.input_csvs),
    )
    parser.add_argument(
        "--sampler",
        help="Name of the sampler to load fits from",
        choices=["dynesty", "nuts", "svi"],
        default="dynesty",
    )
    parser.add_argument(
        "--model_type",
        help="Name of the model type to train",
        choices=["LightGBM", "MLP"],
        default="LightGBM",
    )
    parser.add_argument(
        "--include_redshift",
        help="If flag is set, include redshift data for training",
        default=False,
        action=BooleanOptionalAction,
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
        default=default_config.probs_fn,
    )
    parser.add_argument(
        "--fits_dir",
        help="Directory holding fit parameters",
        default=default_config.fits_dir,
    )
    parser.add_argument(
        "--config_name",
        help="The name of the file containing the model configuration",
        required=True,
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

    trainer = SuperphotTrainer(
        config_name=args.config_name,
        fits_dir=args.fits_dir,
        sampler=args.sampler,
        model_type=args.model_type,
        include_redshift=args.include_redshift,
        probs_file=args.probs_file,
    )

    trainer.run(
        input_csvs=args.input_csvs.split(","),
        extract_wc=args.extract_wc,
        load_checkpoint=args.load_checkpoint,
    )
