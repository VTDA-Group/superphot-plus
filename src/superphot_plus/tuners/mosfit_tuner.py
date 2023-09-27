import dataclasses
import os
from functools import partial

import numpy as np
import ray
from joblib import Parallel, delayed
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import train_test_split

from superphot_plus.model.regressor import SuperphotRegressor
from superphot_plus.format_data_ztf import generate_K_fold, normalize_features
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.data import TrainData
from superphot_plus.supernova_properties import SupernovaProperties
from superphot_plus.trainers.mosfit_trainer import MosfitTrainer
from superphot_plus.utils import (
    create_dataset,
    get_regression_session_metrics,
    log_regressor_metrics_to_tensorboard,
)


class MosfitTuner(MosfitTrainer):
    """Tunes mosfit regressor using Ray and K-Fold cross validation."""

    def __init__(self, parameter, sampler, mosfit_dir, num_cpu=2, num_gpu=0):
        """Tunes models using Ray and K-Fold cross validation.

        Parameters
        ----------
        parameter : str
            The name of the supernova property to tune the model on.
        num_cpu : int
            The number of CPUs to use in parallel for each tuning experiment.
            Defaults to 2.
        num_gpu : int
            The number of GPUs to use in parallel for each tuning experiment.
            Defaults to 0.
        """
        super().__init__(
            parameter=parameter,
            sampler=sampler,
            mosfit_dir=mosfit_dir,
        )
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu

    def run(self, data, num_hp_samples=10):
        """Performs model tuning with cross-validation to get
        the best set of hyperparameters.

        Parameters
        ----------
        num_hp_samples : int
            The number of hyperparameters sets to sample from (for model tuning).
            Defaults to 10.
        """
        if data is None:
            raise ValueError("No data has been provided.")

        # Read and generate training data
        names, posteriors, properties = data
        names, _, posteriors, _, properties, _ = train_test_split(
            names, posteriors, properties, shuffle=True, test_size=0.1
        )
        curr_prop = SupernovaProperties.get_property_by_name(properties, self.parameter)

        # Run model tuning
        best_config = self.tune_model(
            posteriors=posteriors,
            curr_prop=curr_prop,
            num_hp_samples=num_hp_samples,
        )

        # Store best model configuration
        best_config_file = os.path.join(self.models_dir, f"{self.parameter}.yaml")
        best_config.write_to_file(best_config_file)

    def generate_hp_sample(self):
        """Generates random set of hyperparameters for classifier tuning.

        Returns
        -------
        ModelConfig
            Generated regressor hyperparameters.
        """
        return ModelConfig(
            neurons_per_layer=tune.choice([128, 256, 512]),
            num_hidden_layers=tune.choice([3, 4, 5]),
            num_folds=tune.choice(list(range(5, 10))),
            num_epochs=tune.choice([250, 500, 750]),
            batch_size=tune.choice([32, 64, 128]),
            learning_rate=tune.loguniform(1e-4, 1e-1),
        )

    def tune_model(self, posteriors, curr_prop, num_hp_samples=10):
        """Invokes the Ray Tune API to start model tuning. Outputs the best
        model configuration to a log file for further reference.

        Parameters
        ----------
        posteriors : np.ndarray
            The array of posterior samples for the light curve.
        curr_prop : str
            The current property values.
        num_hp_samples : int
            The number of hyperparameters sets to sample from (for model tuning).
            Defaults to 10.

        Returns
        -------
        ModelConfig
            The best set of model hyperparameters found.
        """
        # Define hardware resources per trial.
        resources = {"cpu": self.num_cpu, "gpu": self.num_gpu}

        # Define the parameter search configuration.
        config = dataclasses.asdict(self.generate_hp_sample())

        # Reporter to show on command line/output window.
        reporter = CLIReporter(metric_columns=["avg_val_loss"])

        # Start hyperparameter search.
        result = tune.run(
            partial(self.run_cross_validation, posteriors=posteriors, curr_prop=curr_prop),
            config=config,
            search_alg=OptunaSearch(),
            resources_per_trial=resources,
            metric="avg_val_loss",
            mode="min",
            num_samples=num_hp_samples,
            progress_reporter=reporter,
        )

        # Extract the best trial (hyperparameter config) from the search.
        # The best trial is the one with the minimum validation loss for
        # the folds under analysis.
        best_trial = result.get_best_trial()
        best_val_loss = best_trial.last_result["avg_val_loss"]

        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial validation loss: {best_val_loss}")

        return ModelConfig(**best_trial.config)

    def run_cross_validation(self, config, posteriors, curr_prop):
        """Runs cross-fold validation to estimate the best set of
        hyperparameters for the model.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration for model training, drawn from the default
            ModelConfig values. Used as a Dict to comply with the Tune
            API requirements.
        posteriors : np.ndarray
            The array of posterior samples for the light curve.
        curr_prop : str
            The current property values.
        """
        trial_id = tune.get_trial_id()

        # Run Tune in the project's working directory.
        os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

        # Construct training config from dict
        config = ModelConfig(**config)

        # K-Fold on the training data
        kfold = generate_K_fold(
            features=np.zeros(len(curr_prop)),
            num_folds=config.num_folds,
            stratified=False,
        )

        def run_single_fold(fold):
            train_index, val_index = fold

            train_posts, train_props, val_posts, val_props = self.generate_train_data(
                posteriors=posteriors,
                curr_props=curr_prop,
                train_index=train_index,
                val_index=val_index,
            )
            train_posts, mean, std = normalize_features(train_posts)
            val_posts, mean, std = normalize_features(val_posts, mean, std)

            train_dataset = create_dataset(train_posts, train_props)
            val_dataset = create_dataset(val_posts, val_props)

            model = SuperphotRegressor.create(
                config=ModelConfig(
                    input_dim=train_posts.shape[1],
                    output_dim=1,
                    neurons_per_layer=config.neurons_per_layer,
                    num_hidden_layers=config.num_hidden_layers,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    normalization_means=mean.tolist(),
                    normalization_stddevs=std.tolist(),
                )
            )

            # Train and validate multi-layer perceptron
            return model.train_and_validate(
                train_data=TrainData(train_dataset, val_dataset),
                num_epochs=config.num_epochs,
            )

        # Process each fold in parallel.
        fold_metrics = Parallel(n_jobs=-1)(delayed(run_single_fold)(fold) for fold in kfold)

        # Report mean metrics for the current hyperparameter set.
        avg_val_loss = get_regression_session_metrics(metrics=fold_metrics)
        session.report({"avg_val_loss": avg_val_loss})

        # Log average metrics per epoch to plot on Tensorboard.
        log_regressor_metrics_to_tensorboard(metrics=fold_metrics, config=config, trial_id=trial_id)
