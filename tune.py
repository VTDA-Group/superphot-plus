"""Entry point to run hyperparameter tuning using Nested CV."""
import json
import numpy as np
import os
import ray

from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial
from joblib import Parallel, delayed
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import train_test_split

from superphot_plus.classify_ztf import adjust_log_dists
from superphot_plus.file_paths import METRICS_DIR, MODELS_DIR, INPUT_CSVS, DATA_DIR
from superphot_plus.format_data_ztf import (
    generate_K_fold,
    import_labels_only,
    normalize_features,
    oversample_using_posteriors,
)
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.model.config import ModelConfig, NetworkParams
from superphot_plus.model.data import TrainData
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.file_paths import BEST_CONFIG_FILE
from superphot_plus.utils import create_dataset


def run_tune_params(config, sampler, include_redshift):
    """Estimates the model performance for each hyperparameter set,
    reporting the mean validation loss and accuracy for each fold.

    Parameters
    ----------
    config : Dict[str, Any]
        Tune run configuration
    sampler : str
        The name of the sampler used for fitting
    include_redshift : bool
        If true, include redshift data
    """
    # Run Tune in the project's working directory.
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    fits_dir = f"{DATA_DIR}/{sampler}_fits"

    # Create output folders.
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    output_dim = len(allowed_types)

    # Load data and set aside 10% of data for testing.

    names, labels, redshifts = import_labels_only(
        allowed_types=allowed_types, input_csvs=INPUT_CSVS, fits_dir=fits_dir, sampler=sampler
    )

    names, _, labels, _, redshifts, _ = train_test_split(names, labels, redshifts, test_size=0.1)

    # Generate K-folds for the remaining data.
    kfold = generate_K_fold(np.zeros(len(labels)), labels, config["num_folds"])

    def run_single_fold(fold_id, fold):
        """Trains and validates model on single fold.

        Parameters
        ----------
        fold_id : str
            An identifier for the current fold
        fold : tuple of ndarray
            The fold for cross validation

        Returns
        -------
        tuple
            The fold's validation loss and accuracy.
        """
        # Get training / test indices
        train_index, val_index = fold

        train_names, val_names = names[train_index], names[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        train_classes = SnClass.get_classes_from_labels(train_labels)
        val_classes = SnClass.get_classes_from_labels(val_labels)

        train_redshifts = redshifts[train_index]
        val_redshifts = redshifts[val_index]

        train_features, train_classes, train_redshifts = oversample_using_posteriors(
            lc_names=train_names,
            labels=train_classes,
            goal_per_class=config["goal_per_class"],
            fits_dir=fits_dir,
            sampler=sampler,
            redshifts=train_redshifts,
            oversample_redshifts=include_redshift,
        )
        val_features, val_classes, val_redshifts = oversample_using_posteriors(
            lc_names=val_names,
            labels=val_classes,
            goal_per_class=round(0.1 * config["goal_per_class"]),
            fits_dir=fits_dir,
            sampler=sampler,
            redshifts=val_redshifts,
            oversample_redshifts=include_redshift,
        )

        if include_redshift:
            train_features = np.hstack(
                (
                    train_features,
                    np.array(
                        [
                            train_redshifts,
                        ]
                    ).T,
                )
            )
            val_features = np.hstack(
                (
                    val_features,
                    np.array(
                        [
                            val_redshifts,
                        ]
                    ).T,
                )
            )

        # Normalize the log distributions.
        train_features = adjust_log_dists(train_features, redshift=include_redshift)
        val_features = adjust_log_dists(val_features, redshift=include_redshift)
        train_features, mean, std = normalize_features(train_features)
        val_features, mean, std = normalize_features(val_features, mean, std)

        # Convert to Torch DataSet objects.
        train_dataset = create_dataset(train_features, train_classes)
        val_dataset = create_dataset(val_features, val_classes)

        network_params = NetworkParams(
            input_dim=train_features.shape[1],
            output_dim=output_dim,
            neurons_per_layer=config["neurons_per_layer"],
            num_hidden_layers=config["num_hidden_layers"],
        )

        model = SuperphotClassifier(
            config=ModelConfig(
                network_params=network_params,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                normalization_means=mean.tolist(),
                normalization_stddevs=std.tolist(),
            )
        )

        # Run classifier for the number of specified epochs.
        best_val_loss, val_acc = model.train_and_validate(
            train_data=TrainData(train_dataset, val_dataset),
            run_id=f"fold-{fold_id}",
            num_epochs=config["num_epochs"],
            metrics_dir=METRICS_DIR,
            models_dir=MODELS_DIR,
        )

        return best_val_loss, val_acc

    # Process each fold in parallel.
    r = Parallel(n_jobs=-1)(delayed(run_single_fold)(i, fold) for i, fold in enumerate(kfold))

    # Report metrics for the current hyperparameter set.
    val_losses = [metric[0] for metric in r]
    val_accs = [metric[1] for metric in r]

    session.report({"avg_val_loss": np.mean(val_losses), "avg_val_acc": np.mean(val_accs)})


def run_nested_cv(num_samples, sampler, include_redshift):
    """Runs Tune experiments to search hyperparameter space.

    Parameters
    ----------
    num_samples : int
        The number of hyperparameter sets to generate
    sampler : str
        The name of the sampler used for fitting
    include_redshift : bool
        If true, include redshift data
    """
    # Define hardware resources per trial.
    resources = {"cpu": 2, "gpu": 0}

    # Define the parameter search configuration.
    config = {
        "num_folds": tune.choice(list(range(5, 10))),
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
        partial(run_tune_params, sampler=sampler, include_redshift=include_redshift),
        config=config,
        search_alg=OptunaSearch(),
        resources_per_trial=resources,
        metric="avg_val_loss",
        mode="min",
        num_samples=num_samples,
        progress_reporter=reporter,
    )

    # Extract the best trial (hyperparameter config) from the search.
    # The best trial is the one with the minimum validation loss for
    # the folds under analysis.
    best_trial = result.get_best_trial()
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial validation loss: {best_trial.last_result['avg_val_loss']}")

    # Store best config to file
    with open(BEST_CONFIG_FILE, "w", encoding="utf-8") as out_file:
        json.dump(best_trial.config, out_file)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Entry point to train and evaluate models using K-Fold cross validation",
    )
    parser.add_argument(
        "--num_samples",
        help="Name of parameter combinations to try",
        default=10,
    )
    parser.add_argument(
        "--sampler",
        help="Name of the sampler to load fits from",
        choices=["dynesty", "nuts", "svi"],
        default="dynesty",
    )
    parser.add_argument(
        "--include_redshifts",
        help="If flag is set, include redshift data for hyperparameter tuning",
        default=False,
        action=BooleanOptionalAction,
    )

    args = parser.parse_args()

    run_nested_cv(
        num_samples=args.num_samples,
        sampler=args.sampler,
        include_redshift=args.include_redshifts,
    )
