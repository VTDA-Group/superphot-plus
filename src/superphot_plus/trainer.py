import os
import numpy as np
from sklearn.model_selection import train_test_split

from superphot_plus.file_paths import (
    CM_FOLDER,
    FIT_PLOTS_FOLDER,
    INPUT_CSVS,
    PROBS_FILE,
    CLASSIFICATION_DIR,
)
from superphot_plus.format_data_ztf import normalize_features, tally_each_class
from superphot_plus.model.classifier import SuperphotClassifier
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

from superphot_plus.format_data_ztf import import_labels_only, oversample_using_posteriors
from superphot_plus.utils import adjust_log_dists
from superphot_plus.file_utils import get_posterior_samples


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
        classification_dir=CLASSIFICATION_DIR,
    ):
        super().__init__(
            sampler=sampler,
            fits_dir=os.path.join(classification_dir, f"{sampler}_fits"),
            models_dir=os.path.join(classification_dir, "models"),
            metrics_dir=os.path.join(classification_dir, "metrics"),
            output_file=os.path.join(classification_dir, probs_file),
            log_file=os.path.join(classification_dir, "classification_log.txt"),
        )

        # Classification specific
        self.allowed_types = [
            "SN Ia",
            "SN II",
            "SN IIn",
            "SLSN-I",
            "SN Ibc",
        ]
        self.config_name = config_name
        self.include_redshift = include_redshift
        self.cm_folder = CM_FOLDER
        self.fit_plots_folder = FIT_PLOTS_FOLDER

        # Restart output files
        self.clean_outputs(
            additional_dirs=[
                self.cm_folder,
                self.fit_plots_folder,
            ]
        )

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
        # Loads model and config
        self.setup_model(
            SuperphotClassifier,
            config_name=self.config_name,
            load_checkpoint=load_checkpoint,
        )

        train_data, test_data = self.split_train_test(input_csvs)

        if self.model is None:
            self.train(train_data)

        # Evaluate model on test dataset
        self.evaluate(test_data, extract_wc)

    def split_train_test(self, input_csvs=None):
        """Reads data and splits it into training and testing sets.

        Parameters
        ----------
        input_csvs : list of str
            List of input CSV file paths.

        Returns
        -------
        tuple
            The train data and the test data.
        """
        if input_csvs is None:
            input_csvs = INPUT_CSVS

        # Load train and test data (holdout of 10%)
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
        test_data = ZtfData(test_names, test_labels, test_redshifts)

        return train_data, test_data

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
            probs_csv_path=self.output_file,
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
            log_file=self.log_file,
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
