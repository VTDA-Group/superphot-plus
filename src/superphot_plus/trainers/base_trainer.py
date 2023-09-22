import os
import shutil

from superphot_plus.model.config import ModelConfig


class BaseTrainer:
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
        """Trainer base class.

        Parameters
        ----------
        sampler : str
            Method used for fitting light curves.
        fits_dir : str
            The light curve posterior samples.
        models_dir : str
            Directory to store model checkpoints and configurations.
        metrics_dir : str
            Directory to store model metrics.
        output_file : str
            File where model predictions are written to.
        log_file : str
            File where model information is logged.
        """
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
        additional_dirs=None,
        additional_files=None,
        delete_prev=False,
    ):
        """Creates directories to store model checkpoints
        and metrics, and additional directories if desired.

        Parameters
        ----------
        additional_dirs : list of str
            List of additional directories to create.
        additional_files : list of str
            List of additional directories to create.
        delete_prev : bool
            If true, erases previously created directories and files.
        """
        if additional_dirs is None:
            additional_dirs = []
        if additional_files is None:
            additional_files = []

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
        cls,
        config_name,
        load_checkpoint=False,
    ):
        """Reads model configuration from disk and loads the
        saved checkpoint if load_checkpoint flag was enabled.

        Parameters
        ----------
        cls: SuperphotMlp
            The type of network to load. May currently be
            SuperphotClassifier or SuperphotRegressor.
        config_name : str
            The model configuration file name. This file should be located
            under the trainer's models directory.
        load_checkpoint : bool, optional
            If true, load pretrained model checkpoint. Defaults to False.
        """
        if config_name is None:
            raise ValueError("Model configuration not specified.")

        path = os.path.join(self.models_dir, config_name)
        model_file, config_file = f"{path}.pt", f"{path}.yaml"

        self.config = ModelConfig.from_file(config_file)

        if load_checkpoint:
            self.model, _ = cls.load(model_file, config_file)
