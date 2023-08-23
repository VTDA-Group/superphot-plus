import numpy as np
import os

import ray
from joblib import Parallel, delayed
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from sklearn.model_selection import train_test_split
from superphot_plus.classify_ztf import adjust_log_dists
from superphot_plus.file_paths import METRICS_DIR, MODELS_DIR
from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.data import TrainData

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.format_data_ztf import import_labels_only, generate_K_fold, \
    oversample_using_posteriors, normalize_features
from superphot_plus.utils import create_dataset
from ray.tune.search.optuna import OptunaSearch

####################################
######## Data configuration ########
####################################

SAMPLER = "dynesty"
FITS_DIR = "data/dynesty_fits"
INPUT_CSVS = ["data/training_set.csv"]

####################################

def run_tune_params(config):
    """Estimates the model performance for each hyperparameter set.

    Reports the mean metric value for each fold.

    Parameters
    ----------
    config The hyperparameters under search
    num_epochs The number of epochs for training
    metric The metric to estimate model performance
    """
    # Run Tune in the project's working directory.
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    output_dim = len(allowed_types)

    names, labels = import_labels_only(
        input_csvs=INPUT_CSVS,
        allowed_types=allowed_types,
        fits_dir=FITS_DIR,
        needs_posteriors=True,
        redshift=False,
        sampler=SAMPLER,
    )

    # Set aside 10% of data for testing.
    names, test_names, labels, test_labels = train_test_split(names, labels, test_size=0.1)

    test_classes = SnClass.get_classes_from_labels(test_labels)

    test_features = []
    test_classes_os = []
    test_group_idxs = []
    test_names_os = []

    for i, test_name in enumerate(test_names):
        test_posts = get_posterior_samples(test_name, FITS_DIR, SAMPLER)
        test_features.extend(test_posts)
        test_classes_os.extend([test_classes[i]] * len(test_posts))
        test_names_os.extend([test_names[i]] * len(test_posts))
        if len(test_group_idxs) == 0:
            start_idx = 0
        else:
            start_idx = test_group_idxs[-1][-1] + 1
        test_group_idxs.append(np.arange(start_idx, start_idx + len(test_posts)))

    test_data = (np.array(test_features), test_classes, test_names, test_group_idxs)

    # Generate K-folds for the remaining data.
    kfold = generate_K_fold(np.zeros(len(labels)), labels, config["num_folds"])

    def run_single_fold(id, fold):
        """Trains and validates model on single fold."""
        # Get training / test indices
        train_index, val_index = fold

        train_names, val_names = names[train_index], names[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        train_classes = SnClass.get_classes_from_labels(train_labels)
        val_classes = SnClass.get_classes_from_labels(val_labels)

        train_features, train_classes = oversample_using_posteriors(
            train_names, train_classes, config["goal_per_class"], FITS_DIR, SAMPLER
        )
        val_features, val_classes = oversample_using_posteriors(
            val_names, val_classes, round(0.1 * config["goal_per_class"]), FITS_DIR, SAMPLER
        )

        # Normalize the log distributions.
        train_features = adjust_log_dists(train_features)
        val_features = adjust_log_dists(val_features)
        train_features, mean, std = normalize_features(train_features)
        val_features, mean, std = normalize_features(val_features, mean, std)

        # Convert to Torch DataSet objects.
        train_dataset = create_dataset(train_features, train_classes)
        val_dataset = create_dataset(val_features, val_classes)

        model = SuperphotClassifier(
            config=ModelConfig(
                input_dim=train_features.shape[1],
                output_dim=output_dim,
                neurons_per_layer=config["neurons_per_layer"],
                num_hidden_layers=config["num_hidden_layers"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                normalization_means=mean.tolist(),
                normalization_stddevs=std.tolist(),
            ),
            train_data=TrainData(train_dataset, val_dataset),
        )

        # Run MLP for the number of specified epochs.
        best_val_loss, val_acc = model.run(
            run_id=f"fold-{id}",
            num_epochs=config["num_epochs"],
            metrics_dir=METRICS_DIR,
            models_dir=MODELS_DIR
        )

        return best_val_loss, val_acc

    # Process each fold in parallel.
    r = Parallel(n_jobs=-1)(delayed(run_single_fold)(i, fold) for i, fold in enumerate(kfold))

    val_losses = [metric[0] for metric in r]
    val_accs = [metric[1] for metric in r]

    # Report metrics for the current hyperparameter set.
    session.report({
        "avg_val_loss": np.mean(val_losses),
        "avg_val_acc": np.mean(val_accs)
    })


def run_nested_cv(num_samples):
    """Runs Ray Tuner to search hyperparameter space."""

    # Define the parameter search configuration.
    config = {
        "num_folds": tune.choice(np.arange(5, 10)),
        "num_epochs": tune.choice([250, 500, 750]),
        "neurons_per_layer": tune.choice([128, 256, 512]),
        "num_hidden_layers": tune.choice([2, 3, 4]),
        "batch_size": tune.choice([32, 64, 128]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "goal_per_class": tune.choice([100, 500, 1000]),
    }

    # Reporter to show on command line/output window.
    reporter = CLIReporter(metric_columns=["avg_val_loss", "avg_val_acc"])

    # Init Ray cluster.
    ray.init()

    # Start hyperparameter search.
    result = tune.run(
        run_tune_params,
        config=config,
        search_alg=OptunaSearch(),
        resources_per_trial={"cpu": 4, "gpu": 0},
        metric="avg_val_loss",
        mode="min",
        num_samples=num_samples,
        progress_reporter=reporter,
    )

    # Extract the best trial run from the search.
    # The best trial is the one with the min avg validation loss.
    best_trial = result.get_best_trial()
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial validation loss: {best_trial.last_result['avg_val_loss']}")


if __name__ == "__main__":
    run_nested_cv(num_samples=10)
