"""Entry point to model training and evaluation."""
from argparse import ArgumentParser, BooleanOptionalAction

from superphot_plus.file_paths import CLASSIFICATION_DIR, INPUT_CSVS
from superphot_plus.samplers.sampler import Sampler
from superphot_plus.trainers.classifier_trainer import ClassifierTrainer


def extract_cmd_args():
    """Extracts the script command-line arguments."""
    parser = ArgumentParser(
        description="Entry point to train and evaluate models using K-Fold cross validation",
    )
    parser.add_argument(
        "--config_name",
        help="The name of the file containing the model configuration",
        required=True,
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
        "--classification_dir",
        help="Directory where classification data is stored",
        default=CLASSIFICATION_DIR,
    )
    parser.add_argument(
        "--input_csvs",
        help="List of CSVs containing light curve data (comma separated)",
        default=",".join(INPUT_CSVS),
    )
    parser.add_argument(
        "--extract_wc",
        help="If flag is set, extract wrongly classified samples",
        default=False,
        action=BooleanOptionalAction,
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

    trainer = ClassifierTrainer(
        config_name=args.config_name,
        sampler=args.sampler,
        include_redshift=args.include_redshift,
        classification_dir=args.classification_dir,
    )

    trainer.run(
        input_csvs=args.input_csvs.split(","),
        extract_wc=args.extract_wc,
        load_checkpoint=args.load_checkpoint,
    )
