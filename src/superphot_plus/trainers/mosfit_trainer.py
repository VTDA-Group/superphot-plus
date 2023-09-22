import os

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from superphot_plus.file_paths import MOSFIT_DIR
from superphot_plus.format_data_ztf import normalize_features
from superphot_plus.model.data import TrainData
from superphot_plus.model.regressor import SuperphotRegressor
from superphot_plus.plotting.regressor_results import plot_model_metrics
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.supernova_properties import SupernovaProperties
from superphot_plus.trainers.base_trainer import BaseTrainer
from superphot_plus.utils import (
    adjust_log_dists,
    create_dataset,
    log_regressor_metrics_to_tensorboard,
    save_regression_outputs,
    write_regression_metrics_to_file,
)


class MosfitTrainer(BaseTrainer):
    """Infers supernovae physical parameters."""

    def __init__(
        self,
        parameter,
        sampler="svi",
        mosfit_dir=MOSFIT_DIR,
    ):
        """Trains and evaluates regressor on a specific supernova property.

        The model may be trained from scratch using a specified configuration
        or be loaded from a previous checkpoint stored on disk. In both scenarios
        the model is evaluated on a test holdout set and metrics are generated.

        Parameters
        ----------
        parameter : str
            The name of the supernova property to train the model on.
        sampler : str, optional
            The type of sampler used for the lightcurve fits. Defaults to "svi".
        mosfits_dir : str, optional
            The base directory where mosfit outputs should be logged.
        """
        super().__init__(
            sampler=sampler,
            fits_dir=os.path.join(mosfit_dir, "posteriors"),
            models_dir=os.path.join(mosfit_dir, "models"),
            metrics_dir=os.path.join(mosfit_dir, "metrics"),
            output_file=os.path.join(mosfit_dir, f"{parameter}_preds.csv"),
            log_file=os.path.join(mosfit_dir, f"{parameter}_logs.txt"),
        )

        # Regression specific
        self.parameter = parameter
        self.params_dir = os.path.join(mosfit_dir, "params")
        self.scaler = StandardScaler()

        # Restart output files
        self.clean_outputs()

    def run(self, load_checkpoint=False):
        """Runs the machine learning workflow.

        Trains the model on the whole training set and evaluates it on a
        test holdout set. Metrics are plotted and logged to files.

        Parameters
        ----------
        load_checkpoint : bool
            If true, load pretrained model checkpoint.
        """
        # Load parameter specific model
        self.setup_model(
            SuperphotRegressor,
            config_name=self.parameter,
            load_checkpoint=load_checkpoint,
        )

        # Read posteriors and light curve samples
        names, posteriors, properties = self.read_data()

        # Extract desired supernova property
        curr_prop = SupernovaProperties.get_property_by_name(properties, self.parameter)

        # Split data into training and testing sets
        train_index, test_index = train_test_split(np.arange(0, len(names)), test_size=0.1)

        if not load_checkpoint:
            self.train(
                posteriors=posteriors,
                curr_prop=curr_prop,
                train_index=train_index,
            )

        self.evaluate(names, posteriors, curr_prop, test_index)

    def read_data(self):
        """Reads posteriors and supernova physical properties
        for all the light curves stored on disk.

        Returns
        -------
        tuple of np.array
            The names, the posterior samples and the physical
            parameters for each of the light curves.
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

    def generate_train_data(self, posteriors, curr_props, train_index, val_index):
        """Splits data to create training and validation datasets
        and applies all the pre-processing transformations.

        Parameters
        ----------
        posteriors : np.ndarray
            The array of posterior samples for the light curve.
        curr_props : str
            The current property values.
        train_index : list of int
            The indices of the training samples.
        val_index : list of int
            The indices of the validation samples.

        Returns
        -------
        tuple of TorchDataset
            The training posteriors and properties, and the validation
            posteriors and properties.
        """
        train_posts = posteriors[train_index]
        val_posts = posteriors[val_index]
        train_props = curr_props[train_index]
        val_props = curr_props[val_index]

        train_posts = adjust_log_dists(train_posts)
        val_posts = adjust_log_dists(val_posts)

        train_props = self.scaler.fit_transform(train_props.reshape(-1, 1))
        val_props = self.scaler.fit_transform(val_props.reshape(-1, 1))

        return train_posts, train_props, val_posts, val_props

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
            The test dataset, the respective light curve test names
            and the list of unscaled properties.
        """
        test_names = names[test_index]
        test_posts = posteriors[test_index]
        raw_test_props = curr_props[test_index]

        test_posts = adjust_log_dists(test_posts)
        test_posts, _, _ = normalize_features(test_posts)

        test_props = self.scaler.fit_transform(raw_test_props.reshape(-1, 1))

        test_dataset = create_dataset(test_posts, test_props)

        return test_dataset, test_names, raw_test_props

    def train(self, posteriors, curr_prop, train_index):
        """Trains the network to predict a physical parameter.

        Parameters
        ----------
        posteriors : np.ndarray
            The array of posterior samples for the light curves.
        curr_prop : str
            The current property values for the light curves.
        train_index : list of int
            The indices of the training samples.
        """
        run_id = f"regression_{self.parameter}"

        train_index, val_index = train_test_split(train_index, shuffle=True, test_size=0.1)

        train_posts, train_props, val_posts, val_props = self.generate_train_data(
            posteriors, curr_prop, train_index, val_index
        )

        train_posts, mean, std = normalize_features(train_posts)
        val_posts, mean, std = normalize_features(val_posts, mean, std)

        train_dataset = create_dataset(train_posts, train_props)
        val_dataset = create_dataset(val_posts, val_props)

        self.config.set_non_tunable_params(
            input_dim=train_posts.shape[1],
            output_dim=1,
            norm_means=mean.tolist(),
            norm_stddevs=std.tolist(),
        )

        # Create regression model
        self.model = SuperphotRegressor.create(self.config)

        # Train and validate multi-layer perceptron
        metrics = self.model.train_and_validate(
            train_data=TrainData(train_dataset, val_dataset),
            num_epochs=self.config.num_epochs,
        )

        # Save model checkpoint
        self.model.save(self.models_dir, name=self.parameter)

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

    def evaluate(
        self,
        names,
        posteriors,
        curr_prop,
        test_index,
    ):
        """Evaluates model on the test dataset.

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
        """
        if self.model is None:
            raise ValueError("Cannot evaluate uninitialized model.")

        if not SupernovaProperties.check_property_exists(self.parameter):
            raise ValueError(f"Invalid physical parameter {self.parameter}.")

        test_dataset, test_names, ground_truths = self.generate_test_data(
            names,
            posteriors,
            curr_prop,
            test_index,
        )

        # Run model inference and obtain predictions
        test_iterator = DataLoader(dataset=test_dataset, batch_size=self.config.batch_size)
        predictions = self.model.get_predictions(test_iterator)
        # Revert predictions transformation
        predictions = self.scaler.inverse_transform(predictions).flatten()

        save_regression_outputs(
            test_names,
            ground_truths,
            predictions,
            save_file=self.output_file,
        )

        write_regression_metrics_to_file(
            config=self.config,
            true_outputs=ground_truths,
            pred_outputs=predictions,
            log_file=self.log_file,
        )
