import os
import shutil

from superphot_plus.model.config import ModelConfig
from superphot_plus.model.mlp import SuperphotMlp


class TrainerBase:
    """Trainer base class."""

    def __init__(
        self,
        sampler,
        fits_dir,
        models_dir,
        metrics_dir,
        output_file,
        log_file,
    ):
        # Sampler and respective posterior folder
        self.sampler = sampler
        self.fits_dir = fits_dir
        # Common output folders
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        # Common output files
        self.output_file = output_file
        self.log_file = log_file
        # Model information
        self.model = None
        self.config = None

    def clean_outputs(
        self,
        additional_dirs=[],
        additional_files=[],
        delete_prev=False,
    ):
        # Removes previous output folders and recreates them
        for folder in [self.metrics_dir, self.models_dir] + additional_dirs:
            if delete_prev and os.path.isdir(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

        # Remove previous output files
        for file in [self.output_file, self.log_file] + additional_files:
            if delete_prev and os.path.isfile(file):
                os.remove(file)

    def setup_model(
        self,
        cls: SuperphotMlp,
        config_name,
        load_checkpoint=False,
    ):
        """Reads model configuration from disk and loads the
        saved checkpoint if load_checkpoint flag was enabled.

        Parameters
        ----------
        load_checkpoint : bool
            If true, load pretrained model checkpoint.
        """
        path = os.path.join(self.models_dir, config_name)

        model_file = f"{path}.pt"
        config_file = f"{path}.yaml"

        self.config = ModelConfig.from_file(config_file)

        if load_checkpoint:
            self.model, _ = cls.load(model_file, config_file)
