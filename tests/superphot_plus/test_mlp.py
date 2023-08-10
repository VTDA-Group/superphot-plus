import datetime
import time

import numpy as np
import torch

from superphot_plus.mlp import calculate_accuracy, create_dataset, epoch_time


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


def test_epoch_time():
    """Tests the calculation of the amount of time an epoch takes to train."""
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(minutes=60, seconds=15)

    start_timestamp = time.mktime(start_time.timetuple())
    end_timestamp = time.mktime(end_time.timetuple())

    elapsed_mins, elapsed_secs = epoch_time(start_timestamp, end_timestamp)

    assert elapsed_mins == 60
    assert elapsed_secs == 15
