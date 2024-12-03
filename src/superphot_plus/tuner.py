import dataclasses
import os
from functools import partial
import copy
from typing import Optional

import numpy as np
import ray
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from joblib import Parallel, delayed
from snapi import TransientGroup, SamplerResultGroup

from superphot_plus.config import SuperphotConfig
from superphot_plus.trainer_base import TrainerBase
from superphot_plus.utils import (
    get_session_metrics, log_metrics_to_tensorboard,
)


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
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_cpu = kwargs['num_cpu']
        self.num_gpu = kwargs['num_gpu']

    def generate_hp_sample(self):
        """Generates random set of hyperparameters for tuning."""
        config_copy = copy.deepcopy(self.config)
        config_copy.neurons_per_layer = tune.choice([128, 256, 512])
        config_copy.num_hidden_layers = tune.choice([2, 3, 4, 5])
        config_copy.fits_per_majority = tune.choice([1, 5, 10])
        config_copy.num_epochs = tune.choice([250, 500, 750])
        config_copy.batch_size = tune.choice([32, 64, 128])
        config_copy.learning_rate = tune.loguniform(1e-4, 1e-1)

    def run(
        self,
        transient_data: Optional[TransientGroup] = None,
        sampler_results: Optional[SamplerResultGroup] = None,
        num_hp_samples=10
    ):
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
        if sampler_results is None:
            sampler_results = SamplerResultGroup.load(self.config.sampler_results_fn)
        if transient_data is None:
            transient_data = TransientGroup.load(self.config.transient_data_fn)
            
        train_data, _, _ = self.split_train_test(transient_data, sampler_results) # 1st K-fold
        
        best_config = self.tune_model(train_data, num_hp_samples)
        best_config.write_to_file(
            os.path.join(best_config.models_dir, "best-config.yaml")
        )
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

    def run_cross_validation(self, config, train_data: tuple):
        """Runs cross-fold validation to estimate the best set of
        hyperparameters for the model.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration for model training, drawn from the default
            ModelConfig values. Used as a Dict to comply with the Tune
            API requirements.
        train_data : pd.DataFrame
            Contains all samples and classes info.
        """
        trial_id = tune.get_trial_id()

        # Run Tune in the project's working directory.
        os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

        # Construct training config from dict
        config = SuperphotConfig(**config)

        def run_single_fold(fold):
            train_df, val_df = self.split(train_data[0], split_indices=groups)
            train_srg = train_data[1].filter(train_df.index)
            val_srg = train_data[1].filter(val_df.index)
            
            class_dict = {x.Index: x.label for x in train_df.itertuples()}
            train_srg.balance_classes(class_dict, config.fits_per_majority) # custom config's fits per majority
            class_dict = {x.Index: x.label for x in val_df.itertuples()}
            val_srg.balance_classes(class_dict, config.fits_per_majority)
            
            train_df = self.retrieve_sampler_results(self, train_srg, train_df, balance_classes=False)
            val_df = self.retrieve_sampler_results(self, val_srg, val_df, balance_classes=False)
            
            if self.config.input_features is None:
                input_features = train_df.columns[~train_df.columns.isin(['label', 'score', 'sampler'])]
        
            if self.config.use_redshift_features:
                input_features = np.append(input_features, ['redshift', 'abs_mag'])
                
            train_features = train_df.loc[:, input_features]
            val_features = val_df.loc[:, input_features]

            model = self._create_model_instance(config)
            
            # Train and validate multi-layer perceptron
            return model.train_and_validate(
                train_data=(train_features, train_df['label']),
                val_data=(val_features, val_df['label']),
                num_epochs=config.num_epochs,
                rng_seed=config.random_seed,
            )

        # Process each fold in parallel.
        fold_metrics = Parallel(n_jobs=-1)(
            delayed(run_single_fold)(fold) for fold in self.kf.split(train_data[0].index, train_data[0]['label'])
        )

        # Report mean metrics for the current hyperparameter set.
        avg_val_loss, avg_val_acc = get_session_metrics(metrics=fold_metrics)
        session.report({"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc})

        # Log average metrics per epoch to plot on Tensorboard.
        log_metrics_to_tensorboard(metrics=fold_metrics, config=config, trial_id=trial_id)
