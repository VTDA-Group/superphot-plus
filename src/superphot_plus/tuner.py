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

from superphot_plus.format_data_ztf import generate_K_fold, normalize_features, tally_each_class
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.config import SuperphotConfig
from superphot_plus.model.data import TrainData, ZtfData
from superphot_plus.trainer_base import TrainerBase
from superphot_plus.utils import create_dataset, get_session_metrics, log_metrics_to_tensorboard


class SuperphotTuner(TrainerBase):
    """
    Tunes models using Ray and K-Fold cross validation.

    Parameters
    ----------
    sampler : str
        The type of sampler used for the lightcurve fits. Defaults to "dynesty".
    include_redshift : bool
        If True, includes redshift data for training.
    num_cpu : int
        The number of CPUs to use in parallel for each tuning experiment.
        Defaults to 2.
    num_gpu : int
        The number of GPUs to use in parallel for each tuning experiment.
        Defaults to 0.
    """

    def __init__(
        self,
        sampler="dynesty",
        include_redshift=True,
        num_cpu=2,
        num_gpu=0,
    ):
        super().__init__(sampler, include_redshift)
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu

    def generate_hp_sample(self):
        """Generates random set of hyperparameters for tuning."""
        return SuperphotConfig(
            neurons_per_layer=tune.choice([128, 256, 512]),
            num_hidden_layers=tune.choice([3, 4, 5]),
            goal_per_class=tune.choice([100, 500, 1000]),
            num_folds=tune.choice(list(range(5, 10))),
            num_epochs=tune.choice([250, 500, 750]),
            batch_size=tune.choice([32, 64, 128]),
            learning_rate=tune.loguniform(1e-4, 1e-1),
        )

    def run(self, input_csvs=None, num_hp_samples=10):
        """Performs model tuning with cross-validation to get
        the best set of hyperparameters.

        Parameters
        ----------
        input_csvs : list of str
            The list of training CSV files. Defaults to INPUT_CSVS.
        num_hp_samples : int
            The number of hyperparameters sets to sample from (for model tuning).
            Defaults to 10.
        """
        train_data, _ = self.split_train_test(input_csvs)
        best_config = self.tune_model(train_data, num_hp_samples)
        best_config.write_to_file(os.path.join(self.models_dir, "best-config.yaml"))
        return best_config

    def tune_model(self, train_data, num_hp_samples=10):
        """Invokes the Ray Tune API to start model tuning. Outputs the best
        model configuration to a log file for further reference.

        Parameters
        ----------
        train_data : ZtfData
            Contains the ZTF object names, classes and redshifts for training.
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
        reporter = CLIReporter(metric_columns=["avg_val_loss", "avg_val_acc"])

        # Init Ray cluster.
        ray.init()

        # Start hyperparameter search.
        result = tune.run(
            partial(self.run_cross_validation, train_data=train_data),
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

        return SuperphotConfig(**best_trial.config)

    def run_cross_validation(self, config, train_data: ZtfData):
        """Runs cross-fold validation to estimate the best set of
        hyperparameters for the model.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration for model training, drawn from the default
            ModelConfig values. Used as a Dict to comply with the Tune
            API requirements.
        train_data : ZtfData
            Contains the ZTF object names, classes and redshifts for training.
        """
        trial_id = tune.get_trial_id()

        # Run Tune in the project's working directory.
        os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

        # Construct training config from dict
        config = SuperphotConfig(**config)

        tally_each_class(train_data.labels)

        # Stratified K-Fold on the training data
        kfold = generate_K_fold(
            features=np.zeros(len(train_data.labels)), classes=train_data.labels, num_folds=config.num_folds
        )

        def run_single_fold(fold):
            train_index, val_index = fold

            train_features, train_classes, val_features, val_classes = self.generate_train_data(
                train_data=train_data,
                goal_per_class=config.goal_per_class,
                train_index=train_index,
                val_index=val_index,
            )
            train_features, mean, std = normalize_features(train_features)
            val_features, mean, std = normalize_features(val_features, mean, std)

            train_dataset = create_dataset(train_features, train_classes)
            val_dataset = create_dataset(val_features, val_classes)

            model = SuperphotClassifier.create(
                config=SuperphotConfig(
                    input_dim=train_features.shape[1],
                    output_dim=len(self.allowed_types),
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
        avg_val_loss, avg_val_acc = get_session_metrics(metrics=fold_metrics)
        session.report({"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc})

        # Log average metrics per epoch to plot on Tensorboard.
        log_metrics_to_tensorboard(metrics=fold_metrics, config=config, trial_id=trial_id)
