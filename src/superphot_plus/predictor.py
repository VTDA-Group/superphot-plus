import os
import dataclasses
import numpy as np

from sklearn.model_selection import train_test_split

from superphot_plus.file_paths import DATA_DIR

from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.supernova_properties import SupernovaProperties

from superphot_plus.model.data import TrainData
from superphot_plus.utils import create_dataset, log_regressor_metrics_to_tensorboard

from superphot_plus.format_data_ztf import normalize_features

from superphot_plus.model.regressor import SuperphotRegressor
from superphot_plus.model.config import ModelConfig

from superphot_plus.utils import adjust_log_dists

from superphot_plus.plotting.regressor_results import plot_model_metrics

from superphot_plus.trainer_base import TrainerBase


class MosfitsPredictor(TrainerBase):
    """Infers supernovae physical parameters."""

    def __init__(
        self,
        config_name,
        sampler="svi",
        include_redshift=True,
    ):
        super().__init__(sampler, include_redshift)

        self.config_name = config_name

        self.fits_dir = f"{DATA_DIR}/mosfits/posteriors"
        self.params_dir = f"{DATA_DIR}/mosfits/params"
        self.models_dir = f"{DATA_DIR}/mosfits/models"

        self.model, self.config = None, None

        # self.create_output_dirs(delete_prev=False)

    def run(self):
        """Runs the machine learning workflow."""
        self.setup_model(load_checkpoint=False)

        posteriors, properties = self.read_data()

        # Train regression models for all parameters
        for field in dataclasses.fields(SupernovaProperties):
            curr_prop = SupernovaProperties.filter_property(properties, field.name)
            train_dataset, val_dataset, test_dataset = self.generate_data(posteriors, curr_prop)
            self.train(train_dataset, val_dataset, field.name)

    def read_data(self):
        """Reads posteriors and supernova physical properties
        for all the light curves stored on disk.

        Returns
        -------
        tuple of np.array
            The posterior samples and the physical parameters
            for all the light curves.
        """
        all_posteriors = []
        all_params = []

        for file in os.listdir(self.params_dir):
            filename = file.split(".")[0]

            properties = SupernovaProperties.from_file(
                input_dir=self.params_dir,
                name=filename,
            )
            all_params.append(properties)

            posteriors = PosteriorSamples.from_file(
                input_dir=self.fits_dir,
                name=filename,
                sampling_method=self.sampler,
            )
            all_posteriors.append(posteriors.sample_mean())

        return np.array(all_posteriors), np.array(all_params)

    def setup_model(self, load_checkpoint=False):
        """Reads model configuration from disk and loads the
        saved checkpoint if load_checkpoint flag was enabled.

        Parameters
        ----------
        load_checkpoint : bool
            If true, load pretrained model checkpoint.
        """
        path = os.path.join(self.models_dir, self.config_name)
        model_file, config_file = f"{path}.pt", f"{path}.yaml"

        config = ModelConfig.from_file(config_file)

        if load_checkpoint:
            self.model, _ = SuperphotRegressor.load(model_file, config_file)

        self.config = config

    def generate_data(self, posteriors, curr_props):
        """Splits data to create training, validation and testing datasets.

        Parameters
        ----------
        posteriors : np.ndarray
            The array of posterior samples for the light curve.
        curr_props : str
            The current property values.

        Returns
        -------
        tuple of TorchDataset
            The training, validation and testing datasets.
        """
        # Split into train and test data
        post, test_post, prop, test_prop = train_test_split(
            posteriors,
            curr_props,
            shuffle=True,
            test_size=0.1,
        )

        # Split into train and val data
        train_post, val_posts, train_props, val_props = train_test_split(
            post,
            prop,
            shuffle=True,
            test_size=0.1,
        )

        # TODO: Add redshift data here

        train_post = adjust_log_dists(train_post)
        val_posts = adjust_log_dists(val_posts)

        train_post, mean, std = normalize_features(train_post)
        val_posts, mean, std = normalize_features(val_posts, mean, std)

        train_dataset = create_dataset(train_post, train_props)
        val_dataset = create_dataset(val_posts, val_props)
        test_dataset = create_dataset(test_post, test_prop)

        self.config.set_non_tunable_params(
            input_dim=train_post.shape[1],
            output_dim=1,
            norm_means=mean.tolist(),
            norm_stddevs=std.tolist(),
        )

        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset, param):
        """Trains the network to predict a physical parameter.

        Parameters
        ----------
        train_dataset : TorchDataset
            The training dataset.
        val_dataset : TorchDataset
            The validation dataset.
        param : str
            The target parameter the model will predict.
        """
        run_id = f"regression_{param}"

        # Create regression model
        self.model = SuperphotRegressor.create(self.config)

        # Train and validate multi-layer perceptron
        metrics = self.model.train_and_validate(
            train_data=TrainData(train_dataset, val_dataset),
            num_epochs=self.config.num_epochs,
        )

        # Save model checkpoint
        self.model.save(self.models_dir, name=param)

        # Plot training and validation metrics
        plot_model_metrics(
            metrics=metrics,
            num_epochs=self.config.num_epochs,
            plot_name=run_id,
            metrics_dir=self.metrics_dir,
        )

        # Log average metrics per epoch to plot on Tensorboard.
        log_regressor_metrics_to_tensorboard(
            metrics=[metrics],
            config=self.config,
            trial_id=run_id,
        )


if __name__ == "__main__":
    MosfitsPredictor(config_name="best-config").run()
