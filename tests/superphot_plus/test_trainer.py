import os

import numpy as np
from sklearn.model_selection import train_test_split

from superphot_plus.constants import BATCH_SIZE, LEARNING_RATE, TRAINED_MODEL_PARAMS
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.data import TestData, TrainData
from superphot_plus.plotting.classifier_results import plot_model_metrics
from superphot_plus.utils import create_dataset, log_metrics_to_tensorboard
from superphot_plus.format_data_ztf import normalize_features


def test_run_classifier(tmp_path):
    """Tests that we can run training and evaluation of the model."""
    run_id = "test-run"

    num_samples = 100
    num_epochs = 5
    num_output_classes = 5

    input_dim, output_dim, neurons_per_layer, num_layers = TRAINED_MODEL_PARAMS

    features = np.random.random((num_samples, input_dim))
    labels = np.random.randint(num_output_classes, size=num_samples)

    test_names = ["ZTF-testname"]
    test_group_idxs = [np.arange(0, 10)]

    # Create a test holdout of 10%
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, stratify=labels, test_size=0.1
    )

    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, stratify=train_labels, test_size=0.1
    )

    train_features, mean, std = normalize_features(train_features)
    val_features, mean, std = normalize_features(val_features, mean, std)

    train_dataset = create_dataset(features, labels)
    val_dataset = create_dataset(val_features, val_labels)

    config = ModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        normalization_means=mean.tolist(),
        normalization_stddevs=std.tolist(),
        neurons_per_layer=neurons_per_layer,
        num_hidden_layers=num_layers,
        num_epochs=num_epochs,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    model = SuperphotClassifier.create(config)

    metrics = model.train_and_validate(
        train_data=TrainData(train_dataset, val_dataset),
        num_epochs=config.num_epochs,
    )

    plot_model_metrics(
        metrics=metrics,
        num_epochs=config.num_epochs,
        plot_name=run_id,
        metrics_dir=tmp_path,
    )

    log_metrics_to_tensorboard(metrics=[metrics], config=config, trial_id=run_id, base_dir=tmp_path)

    model.evaluate(
        test_data=TestData(test_features, test_labels, test_names, test_group_idxs),
        probs_csv_path=os.path.join(tmp_path, "probs_mlp.csv"),
    )

    assert os.path.exists(os.path.join(tmp_path, "accuracy_test-run.pdf"))
    assert os.path.exists(os.path.join(tmp_path, "loss_test-run.pdf"))
    assert os.path.exists(os.path.join(tmp_path, "probs_mlp.csv"))
