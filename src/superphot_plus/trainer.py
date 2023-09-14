import os

import numpy as np
from sklearn.model_selection import train_test_split

from superphot_plus.file_paths import PROBS_FILE
from superphot_plus.format_data_ztf import normalize_features, tally_each_class
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.data import TestData, TrainData, ZtfData
from superphot_plus.plotting.classifier_results import plot_model_metrics
from superphot_plus.plotting.confusion_matrices import plot_matrices
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.trainer_base import TrainerBase
from superphot_plus.utils import (
    create_dataset,
    extract_wrong_classifications,
    log_metrics_to_tensorboard,
    write_metrics_to_file,
)


class SuperphotTrainer(TrainerBase):
    """
    Trains and evaluates models using K-Fold cross validation.

    The model may be trained from scratch using a specified configuration
    or be loaded from a previous checkpoint stored on disk. In both scenarios
    the model is evaluated on a test holdout set and metrics are generated.

    Parameters
    ----------
    config_name : str
        The name of the pre-trained model configuration to load. This file should
        be located under the specified models directory. Defaults to None.
    sampler : str
        The type of sampler used for the lightcurve fits. Defaults to "dynesty".
    include_redshift : bool
        If True, includes redshift data for training.
    probs_file : str
        The file where test probabilities are written. Defaults to PROBS_FILE.
    """

    def __init__(
        self,
        config_name,
        sampler="dynesty",
        include_redshift=True,
        probs_file=PROBS_FILE,
    ):
        super().__init__(sampler, include_redshift, probs_file)
        self.config_name = config_name
        self.model, self.config = None, None

        self.create_output_dirs(delete_prev=False)

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
            self.model, _ = SuperphotClassifier.load(model_file, config_file)

        self.config = config

    def run(self, input_csvs=None, extract_wc=False, load_checkpoint=False):
        """Runs the machine learning workflow.

        Trains the model on the whole training set and evaluates it on a
        test holdout set. Metrics are plotted and logged to files.

        Parameters
        ----------
        input_csvs : list of str
            The list of training CSV files. Defaults to INPUT_CSVS.
        extract_wc : bool
            If true, assumes all sample fit plots are saved in
            FIT_PLOTS_FOLDER. Copies plots of wrongly classified samples to
            separate folder for manual followup. Defaults to False.
        load_checkpoint : bool
            If true, load pretrained model checkpoint.
        """
        train_data, test_data = self.split_train_test(input_csvs)

        # Loads model and config
        self.setup_model(load_checkpoint)

        if self.model is None:
            self.train(train_data)

        # Evaluate model on test dataset
        self.evaluate(test_data, extract_wc)

    def train(self, train_data: ZtfData):
        """Trains the model with a specific set of hyperparameters.

        Parameters
        ----------
        train_data : ZtfData
            Contains the ZTF object names, classes and redshifts for training.
        """
        run_id = "final"

        tally_each_class(train_data.labels)  # original tallies

        # Split data into training and validation sets
        train_index, val_index = train_test_split(
            np.arange(0, len(train_data.labels)), stratify=train_data.labels, test_size=0.1
        )

        train_features, train_classes, val_features, val_classes = self.generate_train_data(
            train_data=train_data,
            goal_per_class=self.config.goal_per_class,
            train_index=train_index,
            val_index=val_index,
        )
        train_features, mean, std = normalize_features(train_features)
        val_features, mean, std = normalize_features(val_features, mean, std)

        train_dataset = create_dataset(train_features, train_classes)
        val_dataset = create_dataset(val_features, val_classes)

        self.config.set_non_tunable_params(
            input_dim=train_features.shape[1],
            output_dim=len(self.allowed_types),
            norm_means=mean.tolist(),
            norm_stddevs=std.tolist(),
        )

        self.model = SuperphotClassifier.create(self.config)

        # Train and validate multi-layer perceptron
        metrics = self.model.train_and_validate(
            train_data=TrainData(train_dataset, val_dataset), num_epochs=self.config.num_epochs
        )

        # Save model checkpoint
        self.model.save(self.models_dir)

        # Plot training and validation metrics
        plot_model_metrics(
            metrics=metrics,
            num_epochs=self.config.num_epochs,
            plot_name=run_id,
            metrics_dir=self.metrics_dir,
        )

        # Log average metrics per epoch to plot on Tensorboard.
        log_metrics_to_tensorboard(metrics=[metrics], config=self.config, trial_id=run_id)

    def evaluate(self, test_data: ZtfData, extract_wc=False):
        """Evaluates a pretrained model on the test holdout set.

        Parameters
        ----------
        test_data : ZtfData
            Contains the ZTF object names, classes and redshifts for testing.
        extract_wc : bool
            If true, assumes all sample fit plots are saved in
            FIT_PLOTS_FOLDER. Copies plots of wrongly classified samples to
            separate folder for manual followup. Defaults to False.
        Returns
        -------
        tuple
            A tuple containing the test ground truths, the respective
            predicted classes and the predicted classes for which
            classification confidence exceeded 70%.
        """
        if self.model is None:
            raise ValueError("Cannot evaluate uninitialized model.")

        test_features, test_classes, test_names, test_group_idxs = self.generate_test_data(
            test_data=test_data
        )
        test_features, _, _ = normalize_features(test_features)

        results = self.model.evaluate(
            test_data=TestData(test_features, test_classes, test_names, test_group_idxs),
            probs_csv_path=self.probs_file,
        )

        true_classes, _, pred_classes, pred_probs = zip(results)

        true_classes = np.hstack(true_classes)
        pred_classes = np.hstack(pred_classes)
        pred_probs_above_07 = np.hstack(pred_probs) > 0.7

        true_classes = SnClass.get_labels_from_classes(true_classes)
        pred_classes = SnClass.get_labels_from_classes(pred_classes)

        # Log evaluation metrics
        write_metrics_to_file(
            config=self.config,
            true_classes=true_classes,
            pred_classes=pred_classes,
            prob_above_07=pred_probs_above_07,
            log_file=self.classify_log_file,
        )
        plot_matrices(
            config=self.config,
            true_classes=true_classes,
            pred_classes=pred_classes,
            prob_above_07=pred_probs_above_07,
            cm_folder=self.cm_folder,
        )
        if extract_wc:
            extract_wrong_classifications(
                true_classes=true_classes,
                pred_classes=pred_classes,
                ztf_test_names=test_data.names,
            )
