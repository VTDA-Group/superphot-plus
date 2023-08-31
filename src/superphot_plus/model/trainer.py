import dataclasses
import os
import shutil
from functools import partial

import numpy as np
import ray
import torch
import yaml
from joblib import Parallel, delayed
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import train_test_split

from superphot_plus.file_paths import (
    BEST_CONFIG_FILE,
    CLASSIFY_LOG_FILE,
    CM_FOLDER,
    DATA_DIR,
    FIT_PLOTS_FOLDER,
    INPUT_CSVS,
    METRICS_DIR,
    MODELS_DIR,
    PROBS_FILE,
)
from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.format_data_ztf import (
    generate_K_fold,
    import_labels_only,
    normalize_features,
    oversample_using_posteriors,
    tally_each_class,
)
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.model.config import ModelConfig, TrainConfig
from superphot_plus.model.data import TestData, TrainData, ZtfData
from superphot_plus.plotting.confusion_matrices import plot_matrices
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import (
    adjust_log_dists,
    create_dataset,
    extract_wrong_classifications,
    write_metrics_to_file,
)


# pylint: disable=too-many-instance-attributes
class CrossValidationTrainer:
    """
    Tunes, trains and evaluates models. Tuning uses K-Fold
    cross validation to estimate model performance.

    Parameters
    ----------
    sampler : str
        The type of sampler used for the lightcurve fits. Defaults to "dynesty".
    include_redshift : bool
        If True, includes redshift data for training.
    extract_wc : bool
        If true, assumes all sample fit plots are saved in
        FIT_PLOTS_FOLDER. Copies plots of wrongly classified samples to
        separate folder for manual followup. Defaults to False.
    probs_file : str
        The file where test probabilities are written. Defaults to PROBS_FILE.
    """

    def __init__(
        self,
        sampler="dynesty",
        include_redshift=True,
        extract_wc=False,
        probs_file=PROBS_FILE,
    ):
        self.allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]

        self.sampler = sampler
        self.include_redshift = include_redshift
        self.extract_wc = extract_wc
        self.probs_file = probs_file

        # Log folders
        self.metrics_dir = METRICS_DIR
        self.models_dir = MODELS_DIR
        self.cm_folder = CM_FOLDER
        self.classify_log_file = CLASSIFY_LOG_FILE
        self.fit_plots_folder = FIT_PLOTS_FOLDER

        # Derive from sampler type
        self.fits_dir = f"{DATA_DIR}/{sampler}_fits"

        self.reset_outputs()

    def reset_outputs(self):
        """Performs cleanup of previous model outputs."""
        for folder in [self.metrics_dir, self.models_dir, self.cm_folder, self.fit_plots_folder]:
            if os.path.exists(folder) and os.path.isdir(folder):
                shutil.rmtree(folder)

        # Recreate output directories
        os.makedirs(self.metrics_dir)
        os.makedirs(self.models_dir)
        os.makedirs(self.cm_folder)
        os.makedirs(self.fit_plots_folder)

    def run(self, input_csvs=None, num_hp_samples=10):
        """Runs the machine learning workflow.

        Performs model tuning with cross-validation to get the best set
        of hyperparameters, retrains the model on the whole training set,
        and evaluates it on a test holdout set. Metrics are plotted and
        logged to files.

        Parameters
        ----------
        input_csvs : list of str
            The list of training CSV files. Defaults to INPUT_CSVS.
        num_hp_samples : int
            The number of hyperparameters sets to sample from (for model tuning).
            Defaults to 10.
        """
        if input_csvs is None:
            input_csvs = INPUT_CSVS

        # 1. Load train and test data (holdout of 10%)
        names, labels, redshifts = import_labels_only(
            input_csvs=input_csvs,
            allowed_types=self.allowed_types,
            fits_dir=self.fits_dir,
            sampler=self.sampler,
        )
        names, test_names, labels, test_labels, redshifts, test_redshifts = train_test_split(
            names, labels, redshifts, stratify=labels, shuffle=True, test_size=0.1
        )
        train_data = ZtfData(names, labels, redshifts)

        # 2. Tune model to find best set of hyperparams (using CV)
        best_config, best_val_loss = self.tune_model(train_data=train_data, num_hp_samples=num_hp_samples)

        # 3. Retrain model on training data
        model = self.train(config=best_config, train_data=train_data, save_model=True)

        # 4. Evaluate model on test dataset
        true_classes, pred_classes, pred_probs = self.evaluate(
            model=model, test_data=ZtfData(test_names, test_labels, test_redshifts)
        )

        # 5. Log evaluation metrics
        write_metrics_to_file(
            config=best_config,
            true_classes=true_classes,
            pred_classes=pred_classes,
            prob_above_07=pred_probs,
            val_loss_avg=best_val_loss,
            log_file=self.classify_log_file,
        )

        if self.extract_wc:
            extract_wrong_classifications(
                true_classes=true_classes,
                pred_classes=pred_classes,
                ztf_test_names=test_names,
            )

        plot_matrices(
            config=best_config,
            true_classes=true_classes,
            pred_classes=pred_classes,
            prob_above_07=pred_probs,
            cm_folder=self.cm_folder,
        )

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
        tuple
            A tuple containing the best train configuration and the respective
            validation loss (mean of all the folds).
        """
        # Define hardware resources per trial.
        resources = {"cpu": 2, "gpu": 0}

        # Define the parameter search configuration.
        config = dataclasses.asdict(TrainConfig())

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

        # Store best config to file
        encoded_string = yaml.dump(best_trial.config, sort_keys=False)
        with open(BEST_CONFIG_FILE, "w", encoding="utf-8") as file_handle:
            file_handle.write(encoded_string)

        return TrainConfig(**best_trial.config), best_val_loss

    def run_cross_validation(self, config, train_data: ZtfData):
        """Runs cross-fold validation to estimate the best set of
        hyperparameters for the model.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration for model training, drawn from the default
            TrainConfig values. Used as a Dict to comply with the Tune
            API requirements.
        train_data : ZtfData
            Contains the ZTF object names, classes and redshifts for training.
        """
        # Construct training config from dict
        config = TrainConfig(**config)

        trial_id = tune.get_trial_id()

        # Run Tune in the project's working directory.
        os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

        # Print original tallies
        tally_each_class(train_data.labels)

        # Stratified K-Fold on the training data
        kfold = generate_K_fold(
            features=np.zeros(len(train_data.labels)), classes=train_data.labels, num_folds=config.num_folds
        )

        def run_single_fold(fold_id, fold):
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

            model = SuperphotClassifier(
                config=ModelConfig(
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
            # TODO: Improve this, because only the last fold models will be
            best_val_loss, val_acc = model.train_and_validate(
                train_data=TrainData(train_dataset, val_dataset),
                run_id=f"{trial_id}-fold-{fold_id}",
                num_epochs=config.num_epochs,
                metrics_dir=self.metrics_dir,
                models_dir=self.models_dir,
                plot_metrics=False,
            )

            return best_val_loss, val_acc

        # Process each fold in parallel.
        results = Parallel(n_jobs=-1)(delayed(run_single_fold)(i, fold) for i, fold in enumerate(kfold))

        # Report metrics for the current hyperparameter set.
        val_losses, val_accs = zip(*results)
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)

        session.report({"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc})

    def train(self, config: TrainConfig, train_data: ZtfData, save_model=True):
        """Trains a model with a specific set of hyperparameters.

        Parameters
        ----------
        config : TrainConfig
            The configuration for model training, drawn from the default
            TrainConfig values.
        train_data : ZtfData
            Contains the ZTF object names, classes and redshifts for training.
        save_model : bool
            If True, the model is saved to disk when training ends. Defaults
            to True.

        Returns
        -------
        model : SuperphotClassifier
            The trained model instance.
        """
        tally_each_class(train_data.labels)  # original tallies

        # Split data into training and validation sets
        train_index, val_index = train_test_split(
            np.arange(0, len(train_data.labels)), stratify=train_data.labels, test_size=0.1
        )

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

        model = SuperphotClassifier(
            config=ModelConfig(
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
        model.train_and_validate(
            train_data=TrainData(train_dataset, val_dataset),
            run_id="final",
            num_epochs=config.num_epochs,
            metrics_dir=self.metrics_dir,
            models_dir=self.models_dir,
            plot_metrics=True,
        )

        # Save best model to disk
        if save_model:
            torch.save(model, f"{DATA_DIR}/model.pt")

        return model

    def evaluate(self, model: SuperphotClassifier, test_data: ZtfData):
        """Evaluates a pretrained model on the test holdout set.

        Parameters
        ----------
        model : SuperphotClassifier
            The pretrained model used for evaluation.
        test_data : ZtfData
            Contains the ZTF object names, classes and redshifts for testing.

        Returns
        -------
        tuple
            A tuple containing the test ground truths, the respective
            predicted classes and the predicted classes for which
            classification confidence exceeded 70%.
        """
        test_features, test_classes, test_names, test_group_idxs = self.generate_test_data(
            test_data=test_data
        )
        test_features, _, _ = normalize_features(test_features)

        results = model.evaluate(
            test_data=TestData(test_features, test_classes, test_names, test_group_idxs),
            probs_csv_path=self.probs_file,
        )

        true_classes, _, pred_classes, pred_probs = zip(results)

        true_classes = np.hstack(true_classes)
        pred_classes = np.hstack(pred_classes)
        pred_probs_above_07 = np.hstack(pred_probs) > 0.7

        true_classes = SnClass.get_labels_from_classes(true_classes)
        pred_classes = SnClass.get_labels_from_classes(pred_classes)

        return true_classes, pred_classes, pred_probs_above_07

    def generate_train_data(self, train_data, goal_per_class, train_index, val_index):
        """Extracts and processes the data for training and validation.
        Oversamples the features to tackle the supernovae class imbalance
        and adjusts them to their log distributions.

        Parameters
        ----------
        train_data : ZtfData
            Contains the ZTF object names, classes and redshifts for training.
        goal_per_class : int
            The number of samples for each supernova class (for oversampling).
        train_index : np.ndarray
            The indices for the training data samples.
        val_index : np.ndarray
            The indices for the validation data samples.

        Returns
        -------
        tuple
            A tuple containing the final training features and respective classes,
            and validation features and respective classes.
        """
        names, labels, redshifts = train_data

        train_names, val_names = names[train_index], names[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]
        train_redshifts, val_redshifts = redshifts[train_index], redshifts[val_index]

        # Convert labels to classes
        train_classes = SnClass.get_classes_from_labels(train_labels)
        val_classes = SnClass.get_classes_from_labels(val_labels)

        train_features, train_classes, train_redshifts = oversample_using_posteriors(
            lc_names=train_names,
            labels=train_classes,
            goal_per_class=goal_per_class,
            fits_dir=self.fits_dir,
            sampler=self.sampler,
            redshifts=train_redshifts,
            oversample_redshifts=self.include_redshift,
        )
        val_features, val_classes, val_redshifts = oversample_using_posteriors(
            lc_names=val_names,
            labels=val_classes,
            goal_per_class=round(0.1 * goal_per_class),
            fits_dir=self.fits_dir,
            sampler=self.sampler,
            redshifts=val_redshifts,
            oversample_redshifts=self.include_redshift,
        )

        # merge redshifts before normalizations
        if self.include_redshift:
            # fmt: off
            train_features = np.hstack((train_features, np.array([train_redshifts, ]).T))
            val_features = np.hstack((val_features, np.array([val_redshifts, ]).T))
            # fmt: on

        train_features = adjust_log_dists(train_features, redshift=self.include_redshift)
        val_features = adjust_log_dists(val_features, redshift=self.include_redshift)

        return train_features, train_classes, val_features, val_classes

    def generate_test_data(self, test_data: ZtfData):
        """Extracts and processes the data for testing, adjusting the
        features to their log distributions.

        Parameters
        ----------
        test_data : ZtfData
            Contains the ZTF object names, classes and redshifts for testing.

        Returns
        -------
        tuple
            A tuple containing the final test features and respective classes,
            the corresponding test ZTF object names and test group indices.
        """
        test_names, test_labels, test_redshifts = test_data

        test_features = []
        test_classes_os = []
        test_group_idxs = []
        test_names_os = []
        test_redshifts_os = []

        test_classes = SnClass.get_classes_from_labels(test_labels)

        for i, test_name in enumerate(test_names):
            test_posts = get_posterior_samples(test_name, self.fits_dir, self.sampler)
            test_features.extend(test_posts)
            test_classes_os.extend([test_classes[i]] * len(test_posts))
            test_names_os.extend([test_names[i]] * len(test_posts))
            if self.include_redshift:
                test_redshifts_os.extend([test_redshifts[i]] * len(test_posts))
            if len(test_group_idxs) == 0:
                start_idx = 0
            else:
                start_idx = test_group_idxs[-1][-1] + 1
            test_group_idxs.append(np.arange(start_idx, start_idx + len(test_posts)))

        test_features = np.array(test_features)
        test_classes = np.array(test_classes_os)
        test_names = np.array(test_names_os)

        if self.include_redshift:
            # fmt: off
            test_features = np.hstack((test_features, np.array([test_redshifts_os, ]).T))
            # fmt: on

        test_features = adjust_log_dists(test_features, redshift=self.include_redshift)

        return test_features, test_classes, test_names, test_group_idxs
