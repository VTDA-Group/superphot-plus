import dataclasses
import os

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

from superphot_plus.file_paths import DATA_DIR
from superphot_plus.format_data_ztf import normalize_features
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.data import TrainData
from superphot_plus.model.regressor import SuperphotRegressor
from superphot_plus.plotting.regressor_results import plot_model_metrics
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.supernova_properties import SupernovaProperties
from superphot_plus.trainer_base import TrainerBase
from superphot_plus.utils import (
    adjust_log_dists,
    create_dataset,
    log_regressor_metrics_to_tensorboard,
    save_regression_outputs,
    write_regression_metrics_to_file,
)


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

    def run(self):
        """Runs the machine learning workflow, training and
        evaluating models for each physical parameter."""
        self.setup_model()

        names, posteriors, properties = self.read_data()

        # Splitting data using indices for datasets to coincide
        # in the training of each model
        train_index, test_index = train_test_split(np.arange(0, len(names)), test_size=0.1)

        # Train regression models for all parameters
        for field in dataclasses.fields(SupernovaProperties):
            curr_prop = SupernovaProperties.get_property_by_name(properties, field.name)

            print(f"Shape names: {np.asarray(names).shape}")
            print(f"Shape posteriors: {np.asarray(posteriors).shape}")
            print(f"Shape Curr_prop: {np.asarray(curr_prop).shape}")

            train_dataset, val_dataset = self.generate_train_data(
                posteriors,
                curr_prop,
                train_index,
            )
            self.train(train_dataset, val_dataset, field.name)

            test_dataset, test_names, ground_truths, scaler = self.generate_test_data(
                names,
                posteriors,
                curr_prop,
                test_index,
            )
            self.evaluate(test_dataset, test_names, ground_truths, field.name, scaler)

    def setup_model(self):
        """Reads model configuration from disk."""
        path = os.path.join(self.models_dir, self.config_name)
        self.config = ModelConfig.from_file(f"{path}.yaml")

    def read_data(self):
        """Reads posteriors and supernova physical properties
        for all the light curves stored on disk.

        Returns
        -------
        tuple of np.array
            The posterior samples and the physical parameters
            for all the light curves.
        """
        all_names = []
        all_posteriors = []
        all_params = []

        for file in os.listdir(self.params_dir):
            filename = file.split(".")[0]

            all_names.append(filename)

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

        return np.array(all_names), np.array(all_posteriors), np.array(all_params)

    def generate_train_data(self, posteriors, curr_props, train_index):
        """Splits data to create training and validation datasets
        and applies all the pre-processing transformations.

        Parameters
        ----------
        posteriors : np.ndarray
            The array of posterior samples for the light curve.
        curr_props : str
            The current property values.
        train_index : list of int
            The indices for the training samples.

        Returns
        -------
        tuple of TorchDataset
            The training and validation datasets.
        """
        train_index, val_index = train_test_split(train_index, shuffle=True, test_size=0.1)

        train_posts = posteriors[train_index]
        val_posts = posteriors[val_index]
        train_props = curr_props[train_index]
        val_props = curr_props[val_index]

        train_posts = adjust_log_dists(train_posts)
        val_posts = adjust_log_dists(val_posts)
        # train_props = adjust_log_dists(train_props)
        # val_props = adjust_log_dists(val_props)

        train_posts, mean, std = normalize_features(train_posts)
        val_posts, mean, std = normalize_features(val_posts, mean, std)

        scaler = StandardScaler()
        train_props = scaler.fit_transform(train_props.reshape(-1, 1))
        val_props = scaler.fit_transform(val_props.reshape(-1, 1))
        # train_props, mean, std = normalize_features(train_props)
        # val_props, mean, std = normalize_features(val_props, mean, std)

        train_dataset = create_dataset(train_posts, train_props)
        val_dataset = create_dataset(val_posts, val_props)

        self.config.set_non_tunable_params(
            input_dim=train_posts.shape[1],
            output_dim=1,
            norm_means=mean.tolist(),
            norm_stddevs=std.tolist(),
        )

        return train_dataset, val_dataset

    def generate_test_data(self, names, posteriors, curr_props, test_index):
        """Applies pre-processing transformations and creates
        the dataset for model evaluation.

        Parameters
        ----------
        names : np.ndarray
            The array of light curve names.
        posteriors : np.ndarray
            The array of posterior samples for the light curves.
        curr_props : str
            The current property values for the light curves.
        test_index : list of int
            The indices for the test samples.

        Returns
        -------
        tuple
            The test dataset and the respective light curve test names.
        """
        test_names = names[test_index]
        test_posts = posteriors[test_index]
        raw_test_props = curr_props[test_index]

        test_posts = adjust_log_dists(test_posts)
        test_posts, _, _ = normalize_features(test_posts)

        scaler = StandardScaler()
        test_props = scaler.fit_transform(raw_test_props.reshape(-1, 1))

        test_dataset = create_dataset(test_posts, test_props)

        return test_dataset, test_names, raw_test_props, scaler

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

        # Log average metrics per epoch to plot on Tensorboard
        log_regressor_metrics_to_tensorboard(
            metrics=[metrics],
            config=self.config,
            trial_id=run_id,
        )

    def evaluate(self, test_dataset, test_names, ground_truths, param, scaler):
        """Evaluates model on the test dataset.

        Parameters
        ----------
        test_dataset : TorchDataset
            The test dataset.
        param : str
            The name of the physical parameter.
        """
        if self.model is None:
            raise ValueError("Cannot evaluate uninitialized model.")

        if not SupernovaProperties.check_property_exists(param):
            raise ValueError(f"Invalid physical parameter {param}.")

        predictions = self.model.evaluate(test_dataset=test_dataset)

        predictions = scaler.inverse_transform(predictions).flatten()

        save_regression_outputs(
            test_names,
            param,
            ground_truths,
            predictions,
            save_file=os.path.join(DATA_DIR, f"{param}.csv"),
        )

        write_regression_metrics_to_file(
            config=self.config,
            true_outputs=ground_truths,
            pred_outputs=predictions,
            log_file=os.path.join(DATA_DIR, f"{param}_log.txt"),
        )


if __name__ == "__main__":
    MosfitsPredictor(config_name="best-config").run()
