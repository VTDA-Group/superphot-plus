import os

import numpy as np
import lightgbm
import pickle

from superphot_plus.format_data_ztf import normalize_features
from superphot_plus.config import SuperphotConfig
from superphot_plus.model.metrics import ModelMetrics
from torch.utils.data import DataLoader, TensorDataset
from superphot_plus.utils import (
    save_test_probabilities,
)

class SuperphotLightGBM:
    """The LightGBM model.

    Parameters
    ----------
    config : SuperphotConfig
        The MLP architecture configuration.
    """

    def __init__(self, config: SuperphotConfig, target_label=None):
        super().__init__()

        # Initialize MLP architecture
        self.config = config
        
        # Model state dictionary
        self.best_model = None
        self.target_label = target_label
        
    def train_and_validate(
        self,
        train_data,
        rng_seed=None,
        **kwargs,
    ):
        """
        Runs LightGBM training and validation.
        
        Parameters
        ----------
        train_data : TrainData
            The training dataset.
        rng_seed : int, optional
            Random state that is seeded. if none, use machine entropy.

        Returns
        -------
        tuple
            A tuple containing arrays of metrics for each epoch
            (training accuracies and losses, validation accuracies and losses).
        """
        train_dataset, valid_dataset = train_data
        train_iterator = DataLoader(
            dataset=train_dataset
        )
        valid_iterator = DataLoader(
            dataset=valid_dataset,
        )
        
        train_feats = np.asarray([x[0].numpy() for x, y in train_iterator])
        train_classes = np.asarray([y[0] for x, y in train_iterator])
        uc, cts = np.unique(train_classes, return_counts=True)
        val_feats = np.asarray([x[0].numpy() for x, y in valid_iterator])
        val_classes = np.asarray([y[0] for x, y in valid_iterator])
        uc, cts = np.unique(val_classes, return_counts=True)
        lightgbm_params = {
            "boosting": "dart",
            "data_sample_strategy": "goss",
            "verbosity": -1,
            "random_state": rng_seed,
            'max_depth': 5,
            'num_leaves': 20,
            'lambda_l1': 5.0,
            'n_estimators': 250,
        }
        
        if self.target_label is not None:
            # Single class classification
            lightgbm_params["objective"] = "binary"
            lightgbm_params["metric"] = "binary_logloss"
            
        else:
            lightgbm_params["objective"] = "multiclass"
            lightgbm_params["metric"] = "multi_logloss"
            lightgbm_params['num_class'] = len(np.unique(train_classes))

        eval_results = {}
        
        classifier = lightgbm.LGBMClassifier(**lightgbm_params)

        classifier.fit(
            train_feats,
            train_classes,
            eval_set=[
                (train_feats, train_classes),
                (val_feats, val_classes),
            ],
            eval_names=['train', 'val'],
            callbacks=[
                lightgbm.log_evaluation,
                lightgbm.record_evaluation(eval_results)
            ],
            eval_metric=['multi_logloss', 'multi_error',]
        )
        if self.target_label is None:
            metrics = ModelMetrics(
                train_acc = 1. - np.array(eval_results['train']['multi_error']),
                train_loss = np.array(eval_results['train']['multi_logloss']),
                val_acc = 1. - np.array(eval_results['val']['multi_error']),
                val_loss = np.array(eval_results['val']['multi_logloss'])
            )

            best_val_loss = np.min(eval_results['val']['multi_logloss'])
        else:
            metrics = ModelMetrics(
                train_acc = np.ones(len(eval_results['train']['binary_logloss'])),
                train_loss = np.array(eval_results['train']['binary_logloss']),
                val_acc = np.ones(len(eval_results['train']['binary_logloss'])),
                val_loss = np.array(eval_results['val']['binary_logloss'])
            )

            best_val_loss = np.min(eval_results['val']['binary_logloss'])
        
        # Save best model state
        self.best_model = classifier

        # Store best validation loss
        self.config.set_best_val_loss(float(best_val_loss))

        return metrics.get_values()
        
    
    def evaluate(self, test_data, overwrite_save=False):
        """Runs model over a group of test samples.

        Parameters
        ----------
        test_data : TestData
            The data to evaluate the model. Consists of test features,
            test classes, test names and a list of grouped indices, respectively.
        probs_csv_path : str, optional
            Where to store the probability results.

        Returns
        -------
        tuple
            A tuple containing the labels, names, predicted labels
            and maximum probabilities.
        """
        test_features, test_classes, test_names = test_data

        labels, pred_labels, max_probs, names, probs_avgs = [], [], [], [], []

        for test_name in np.unique(test_names):
            group_idx_set = ( test_names == test_name )
            true_classes = test_classes[group_idx_set]
            
            probs = self.best_model.predict_proba(
                test_features[group_idx_set],
            )
            probs_avg = np.mean(probs, axis=0)

            if self.target_label is None:
                pred_labels.append(np.argmax(probs_avg))
                max_probs.append(np.amax(probs_avg))
                labels.append(true_classes[0])
            else:
                pred_target = probs_avg[1] > self.config.prob_threshhold
                pred_labels.append(1-int(pred_target))
                max_probs.append(probs_avg[1] if pred_target else probs_avg[0])
                labels.append(1-true_classes[0])
            
            names.append(test_name)
            probs_avgs.append(probs_avg)
            
        save_test_probabilities(
            names,
            np.array(probs_avgs),
            self.config.probs_fn,
            true_labels=labels,
            target_label=self.target_label
        )

        return (
            np.array(labels).astype(int),
            np.array(names),
            np.array(pred_labels).astype(int),
            np.array(max_probs).astype(float),
        )

    def classify_from_fit_params(self, fit_params):
        """Classify one or multiple light curves solely from the fit parameters
        used in the classifier. Excludes t0 and, for redshift-exclusive
        classifier, A. Includes chi-squared value.

        Parameters
        ----------
        fit_params : np.ndarray
            Set of model fit parameters.

        Returns
        ----------
        np.ndarray
            Probability of each light curve being each SN type.
            Sums to 1 along each row.
        """
        fit_params_2d = np.atleast_2d(fit_params)  # cast to 2D if only 1 light curve

        test_features = normalize_features(
            fit_params_2d,
            self.config.normalization_means,
            self.config.normalization_stddevs,
        )[0]
        probs = self.best_model.predict_proba(test_features)
        try:
            return probs.numpy()
        except:
            return probs
            
    def save(self, config_prefix, suffix=''):
        """Save the classifier as file.

        Parameters
        ----------
        models_dir : str
            Directory to write to
        """
        with open(f"{config_prefix}_{suffix}.pt", 'wb') as f:
            pickle.dump(self, f)
        self.config.write_to_file(f"{config_prefix}_{suffix}.yaml")
        
            
    @classmethod
    def load(cls, filename, config_filename=None):
        """Load a classifier that was saved to disk

        Parameters
        ----------
        path : str
            Path where the classifier was saved

        Returns
        -------
        `~Classifier`
            Loaded classifier
        """
        #config = SuperphotConfig.from_file(config_filename)

        with open(filename, 'rb') as f:
            model = pickle.load(f)
        if config_filename is None:
            config = None
        else:
            config = SuperphotConfig.from_file(config_filename)
        return model, config
