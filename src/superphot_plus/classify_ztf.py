"""This module provides functions to classify supernovae using a
multi-layer perceptron (MLP).

The classification is based on the fit parameters and light curves of
the supernovae."""

import glob
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from superphot_plus.constants import EPOCHS, NUM_FOLDS
from superphot_plus.file_paths import (
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
from superphot_plus.model.config import ModelConfig, NetworkParams
from superphot_plus.model.data import TestData, TrainData
from superphot_plus.plotting.confusion_matrices import plot_confusion_matrix
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import (
    adjust_log_dists,
    calc_accuracy,
    create_dataset,
    extract_wrong_classifications,
    f1_score,
)


# pylint: disable=too-many-instance-attributes)
class CrossValidationTrainer:
    """
    Parameters
    ----------
    num_layers : int
        Number of hidden layers in MLP.
    neurons_per_layer : int
        Number of neurons per hidden layer of MLP.
    goal_per_class : int
        Oversampling such that there are this many fits per supernova type.
    metrics_dir : str
        The directory where model metrics are stored.
    models_dir: str
        The directory where trained models are stored.
    cm_folder : str
        The directory where plots of confusion matrices are stored.
    classify_log_file : str
        The output file where classifications are logged.
    sampler : str
        The type of sampler used for the lightcurve fits.
    include_redshift : bool
        If True, includes redshift data for training.
    """

    def __init__(
            self,
            num_layers,
            neurons_per_layer,
            goal_per_class,
            metrics_dir=METRICS_DIR,
            models_dir=MODELS_DIR,
            cm_folder=CM_FOLDER,
            classify_log_file=CLASSIFY_LOG_FILE,
            sampler="dynesty",
            include_redshift=True,
    ):
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.goal_per_class = goal_per_class
        self.sampler = sampler
        self.include_redshift = include_redshift

        self.metrics_dir = metrics_dir
        self.models_dir = models_dir
        self.cm_folder = cm_folder
        self.classify_log_file = classify_log_file

        # Derive from sampler type
        self.fits_dir = f"{DATA_DIR}/{sampler}_fits"

        # Initialize output directories
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(cm_folder, exist_ok=True)
        os.makedirs(FIT_PLOTS_FOLDER, exist_ok=True)

        # Cleanup previous output content
        for directory in [metrics_dir, models_dir, cm_folder, FIT_PLOTS_FOLDER]:
            files = glob.glob(os.path.join(directory, "*"))
            for f in files:
                os.remove(f)

    def run(
            self,
            input_csvs=None,
            num_epochs=EPOCHS,
            num_folds=NUM_FOLDS,
            csv_path=PROBS_FILE,
            extract_wc=False,
    ):
        """Performs model training and evaluation using K-Fold cross validation.

        Parameters
        ----------
        input_csvs : list of str
            The list of training CSV files.
        num_epochs : int
            Number of training epochs. Defaults to EPOCHS.
        num_folds : int
            The number for K in cross-fold validation. Defaults to NUM_FOLDS.
        csv_path : int
            The file to save test probabilities to. Defaults to PROBS_FILE.
        extract_wc : bool
            If true, assumes all sample fit plots are saved in
            FIT_PLOTS_FOLDER. Copies plots of wrongly classified samples to
            separate folder for manual followup. Defaults to False.
        """
        allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
        output_dim = len(allowed_types)  # number of classes

        if input_csvs is None:
            input_csvs = INPUT_CSVS

        # Write output file header
        with open(csv_path, "w+", encoding="utf-8") as probs_file:
            probs_file.write("Name,Label,pSNIa,pSNII,pSNIIn,pSLSNI,pSNIbc\n")

        names, labels, redshifts = import_labels_only(
            input_csvs=input_csvs,
            allowed_types=allowed_types,
            fits_dir=self.fits_dir,
            sampler=self.sampler,
        )

        tally_each_class(labels)  # original tallies

        kfold = generate_K_fold(np.zeros(len(labels)), labels, num_folds)

        def run_single_fold(fold_id, fold):
            train_index, test_index = fold

            # Separate training from test data
            _, test_names = names[train_index], names[test_index]
            _, test_labels = labels[train_index], labels[test_index]
            _, test_redshifts = redshifts[train_index], redshifts[test_index]

            train_features, train_classes, val_features, val_classes = self.generate_train_data(
                names=names, labels=labels, redshifts=redshifts, train_indices=train_index
            )
            test_features, test_classes, test_names, test_group_idxs = self.generate_test_data(
                test_names=test_names, test_labels=test_labels, test_redshifts=test_redshifts
            )

            # Normalize features
            train_features, mean, std = normalize_features(train_features)
            val_features, mean, std = normalize_features(val_features, mean, std)
            test_features, mean, std = normalize_features(test_features, mean, std)

            # Convert to Torch DataSet objects
            train_dataset = create_dataset(train_features, train_classes)
            val_dataset = create_dataset(val_features, val_classes)

            params = NetworkParams(
                input_dim=train_features.shape[1],
                output_dim=output_dim,
                neurons_per_layer=self.neurons_per_layer,
                num_hidden_layers=self.num_layers,
            )

            model = SuperphotClassifier(
                config=ModelConfig(
                    network_params=params,
                    normalization_means=mean.tolist(),
                    normalization_stddevs=std.tolist(),
                )
            )

            # Train and validate multi-layer perceptron
            best_valid_loss, _ = model.train_and_validate(
                train_data=TrainData(train_dataset, val_dataset),
                run_id=f"fold-{fold_id}",
                num_epochs=num_epochs,
                metrics_dir=self.metrics_dir,
                models_dir=self.models_dir,
                plot_metrics=True,
            )

            # Test model on remaining data
            test_classes, test_names, pred_classes, pred_probs = model.evaluate(
                test_data=TestData(test_features, test_classes, test_names, test_group_idxs),
                probs_csv_path=csv_path,
            )

            return pred_classes, pred_probs > 0.7, test_classes, test_names, best_valid_loss

        r = Parallel(n_jobs=-1)(delayed(run_single_fold)(i, fold) for i, fold in enumerate(kfold))
        (
            predicted_classes_mlp,
            prob_above_07_mlp,
            true_classes_mlp,
            ztf_test_names,
            valid_loss_mlp,
        ) = zip(*r)

        predicted_classes_mlp = np.hstack(tuple(predicted_classes_mlp))
        prob_above_07_mlp = np.hstack(tuple(prob_above_07_mlp))
        true_classes_mlp = np.hstack(tuple(true_classes_mlp))
        ztf_test_names = np.hstack(tuple(ztf_test_names))
        valid_loss_avg = np.mean(valid_loss_mlp)

        true_classes_mlp = SnClass.get_labels_from_classes(true_classes_mlp)
        predicted_classes_mlp = SnClass.get_labels_from_classes(predicted_classes_mlp)

        test_acc = calc_accuracy(predicted_classes_mlp, true_classes_mlp)
        test_f1_score = f1_score(predicted_classes_mlp, true_classes_mlp, class_average=True)

        self.plot_matrices(
            num_epochs=num_epochs,
            true_classes=true_classes_mlp,
            predicted_classes=predicted_classes_mlp,
            prob_above_07=prob_above_07_mlp,
        )

        if extract_wc:
            extract_wrong_classifications(
                true_classes=true_classes_mlp,
                predicted_classes=predicted_classes_mlp,
                ztf_test_names=ztf_test_names,
            )

        self.log_metrics_to_file(
            num_epochs=num_epochs,
            true_classes=true_classes_mlp,
            prob_above_07=prob_above_07_mlp,
            test_f1_score=test_f1_score,
            test_acc=test_acc,
            val_loss_avg=valid_loss_avg,
        )

    def generate_train_data(self, names, labels, redshifts, train_indices):
        """Creates training and validation data, oversampling when needed.

        Parameters
        ----------
        names The full list of ZTF objects.
        labels The full list of supernova labels.
        redshifts The redshifts for each of the ZTF objects.
        train_indices The indices for the training data.

        Returns
        -------
        tuple
            A tuple containing the train features and classes,
            and the validation features and classes, respectively.
        """
        # Set a 10% validation set for the current fold,
        # using stratification on the training classes
        train_index, val_index = train_test_split(
            train_indices, stratify=labels[train_indices], test_size=0.1
        )

        # Separate training into training+validation
        train_names, val_names = names[train_index], names[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]
        train_redshifts, val_redshifts = redshifts[train_index], redshifts[val_index]

        # Convert labels to classes
        train_classes = SnClass.get_classes_from_labels(train_labels)
        val_classes = SnClass.get_classes_from_labels(val_labels)

        train_features, train_classes, train_redshifts = oversample_using_posteriors(
            lc_names=train_names,
            labels=train_classes,
            goal_per_class=self.goal_per_class,
            fits_dir=self.fits_dir,
            sampler=self.sampler,
            redshifts=train_redshifts,
            oversample_redshifts=self.include_redshift,
        )
        val_features, val_classes, val_redshifts = oversample_using_posteriors(
            lc_names=val_names,
            labels=val_classes,
            goal_per_class=round(0.1 * self.goal_per_class),
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

    def generate_test_data(self, test_names, test_labels, test_redshifts):
        """Creates several test groups from testing data.

        Parameters
        ----------
        test_names The list of ZTF test objects.
        test_labels The list of supernova test labels.
        test_redshifts The redshifts for each of the test ZTF objects.

        Returns
        -------
        tuple
            A tuple containing the test features, the test classes,
            the test names and the list of groups, respectively.
        """
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

    def log_metrics_to_file(
            self, num_epochs, true_classes, prob_above_07, test_f1_score, test_acc, val_loss_avg
    ):
        """Outputs the model classification metrics to a file.

        Parameters
        ----------
        num_epochs
            The number of training epochs.
        true_classes
            The classification ground truths.
        prob_above_07
            The class predictions which had a probability of over 70%.
        test_f1_score
            The F1 score calculated with the test data.
        test_acc
            The accuracy over the test data.
        val_loss_avg
            The average validation loss over all the training folds.
        """
        with open(self.classify_log_file, "a+", encoding="utf-8") as the_file:
            the_file.write(str(self.goal_per_class) + " samples per class\n")
            the_file.write(
                str(self.neurons_per_layer) + " neurons per each of " + str(self.num_layers) + " layers\n"
            )
            the_file.write(str(num_epochs) + " epochs\n")
            the_file.write(
                "HOW MANY CERTAIN "
                + str(len(true_classes))
                + " "
                + str(len(true_classes[prob_above_07]))
                + "\n"
            )
            the_file.write(f"MLP class-averaged F1-score: {test_f1_score:.04f}\n")
            the_file.write(f"Accuracy: {test_acc:.04f}\n")
            the_file.write(f"Validation Loss: {val_loss_avg:.04f}\n\n")

    def plot_matrices(self, num_epochs, true_classes, predicted_classes, prob_above_07):
        """Plots confusion matrices.

        Parameters
        ----------
        num_epochs
            The number of training epochs
        true_classes
            The classification ground truths.
        predicted_classes
            The classes predicted for the test data.
        prob_above_07
            The class predictions which had a probability of over 70%.
        """
        fn_prefix = f"cm_{self.goal_per_class}_{num_epochs}_{self.neurons_per_layer}_{self.num_layers}"

        fn_purity = os.path.join(self.cm_folder, fn_prefix + "_p.pdf")
        fn_completeness = os.path.join(self.cm_folder, fn_prefix + "_c.pdf")
        fn_purity_07 = os.path.join(self.cm_folder, fn_prefix + "_p_p07.pdf")
        fn_completeness_07 = os.path.join(self.cm_folder, fn_prefix + "_c_p07.pdf")

        # Plot full and p > 0.7 confusion matrices
        plot_confusion_matrix(true_classes, predicted_classes, fn_purity, True)
        plot_confusion_matrix(true_classes, predicted_classes, fn_completeness, False)

        plot_confusion_matrix(
            true_classes[prob_above_07],
            predicted_classes[prob_above_07],
            fn_purity_07,
            True,
        )
        plot_confusion_matrix(
            true_classes[prob_above_07],
            predicted_classes[prob_above_07],
            fn_completeness_07,
            False,
        )
