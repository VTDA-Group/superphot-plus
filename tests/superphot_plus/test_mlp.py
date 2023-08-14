import datetime
import time
import os

import numpy as np
import torch

from superphot_plus.constants import TRAINED_MODEL_PARAMS
from superphot_plus.mlp import MLP, ModelConfig, ModelData
from superphot_plus.utils import create_dataset, calculate_accuracy, epoch_time


def test_run_mlp(test_data_dir):
    """Tests that we can run training of the model."""

    num_samples = 100
    num_epochs = 5
    num_output_classes = 5

    input_dim, output_dim, neurons_per_layer, num_layers = TRAINED_MODEL_PARAMS

    train_features = np.random.random((num_samples, input_dim))
    train_labels = np.random.randint(num_output_classes, size=num_samples)

    test_features = np.random.random((num_samples, input_dim))
    test_labels = np.random.randint(num_output_classes, size=num_samples)

    test_names = ["ZTF-testname", "ZTF-testname"]
    test_group_idxs = [np.arange(0, 20)]

    train_data = create_dataset(train_features, train_labels)
    val_data = create_dataset(test_features, test_labels)

    config = ModelConfig(input_dim, output_dim, neurons_per_layer, num_layers)
    data = ModelData(train_data, val_data, test_features, test_labels, test_names, test_group_idxs)

    MLP.create(config, data).run(
        num_epochs=num_epochs,
        plot_metrics=True,
        metrics_dir=test_data_dir,
        models_dir=test_data_dir,
    )

    # Check that accuracy and loss plots exist
    acc_plot = f"accuracy_{test_names[0]}.png"
    loss_plot = f"loss_{test_names[0]}.png"

    assert os.path.exists(os.path.join(test_data_dir, acc_plot))
    assert os.path.exists(os.path.join(test_data_dir, loss_plot))


def test_create_dataset():
    """Tests the creation of a TensorDataset."""
    features, labels = np.random.random((2, 5)), [1, 3]

    # Without group indices
    dataset = create_dataset(features, labels)
    assert 2 == len(dataset.tensors)

    # With group indices
    idxs = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    dataset = create_dataset(features, labels, idxs)
    assert 3 == len(dataset.tensors)


def test_calculate_accuracy():
    """Tests the calculation of accuracy for a set of model predictions."""
    y = torch.tensor([4, 0, 2, 3])

    y_pred = torch.tensor(
        [
            [0.0539, -0.2263, -0.7756, -1.6873, 1.4655],
            [2.1537, 0.1596, -1.4326, -1.9861, -0.2499],
            [-0.1134, -0.2378, 1.0218, -1.3380, -0.4937],
            [0.2005, -0.2561, -1.0392, 1.6175, -1.7687],
        ]
    )
    assert 1 == calculate_accuracy(y_pred, y)

    y_pred = torch.tensor(
        [
            [0.0539, -0.2263, -0.7756, -1.6873, 1.4655],
            [2.1537, 0.1596, -1.4326, -1.9861, -0.2499],
            [-0.1134, -0.2378, 1.0218, -1.3380, -0.4937],
            [0.2005, -0.2561, 1.6175, -1.0392, -1.7687],
        ]
    )
    assert 0.75 == calculate_accuracy(y_pred, y)


def test_epoch_time():
    """Tests the calculation of the amount of time an epoch takes to train."""
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(minutes=60, seconds=15)

    start_timestamp = time.mktime(start_time.timetuple())
    end_timestamp = time.mktime(end_time.timetuple())

    elapsed_mins, elapsed_secs = epoch_time(start_timestamp, end_timestamp)

    assert elapsed_mins == 60
    assert elapsed_secs == 15
