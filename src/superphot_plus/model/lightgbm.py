import numpy as np
import lightgbm

from ..constants import EPOCHS
from ..model.metrics import ModelMetrics
from .classifier import SuperphotClassifier


class SuperphotLightGBM(SuperphotClassifier):
    """The LightGBM model.

    Parameters
    ----------
    config : SuperphotConfig
        The MLP architecture configuration.
    """
    
    def train_and_validate(
        self,
        train_data,
        val_data,
        num_epochs=EPOCHS,
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
        (train_feats, train_classes) = train_data
        (val_feats, val_classes) = val_data
        
        train_feats = self.normalize(train_feats)
        val_feats = self.normalize(val_feats)

        lightgbm_params = {
            "boosting": kwargs.get('boosting', 'dart'),
            "data_sample_strategy": kwargs.get('data_sample_strategy', "goss"),
            "verbosity": kwargs.get('verbosity', -1),
            "random_state": rng_seed,
            'max_depth': kwargs.get('max_depth', 5),
            'num_leaves': kwargs.get('num_leaves', 20),
            'lambda_l1': kwargs.get('lambda_l1', 5.0),
            'n_estimators': num_epochs,
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
            eval_metric=['multi_logloss', 'multi_error',],
            callbacks=[
                lightgbm.record_evaluation(eval_results),
                lightgbm.early_stopping(stopping_rounds=20, first_metric_only=True)
            ]
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
        self.set_best_val_loss(float(best_val_loss))

        return metrics