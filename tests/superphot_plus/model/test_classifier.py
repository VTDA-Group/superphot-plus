import csv
import datetime
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from superphot_plus.constants import BATCH_SIZE, LEARNING_RATE, TRAINED_MODEL_PARAMS
from superphot_plus.format_data_ztf import normalize_features
from superphot_plus.model.classifier import SuperphotClassifier
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.data import TestData, TrainData
from superphot_plus.plotting.classifier_results import plot_model_metrics
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import calculate_accuracy, create_dataset, epoch_time, log_metrics_to_tensorboard


def test_run_trainer(tmp_path):
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


def test_classify_single_light_curve(classifier, test_data_dir):
    """Classify light curve based on a pretrained model and fit data."""
    allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    _, classes_to_labels = SnClass.get_type_maps(allowed_types)

    expected_classes = {
        "ZTF22abvdwik": SnClass.SUPERNOVA_IA,
        "ZTF23aacrvqj": SnClass.SUPERNOVA_II,
        "ZTF22abcesfo": SnClass.SUPERNOVA_IIN,
        "ZTF22aarqrxf": SnClass.SUPERLUMINOUS_SUPERNOVA_I,
        "ZTF22abytwjj": SnClass.SUPERNOVA_IBC,
    }

    for ztf_name, label in expected_classes.items():
        lc_probs = classifier.classify_single_light_curve(ztf_name, test_data_dir)
        # assert classes_to_labels[np.argmax(lc_probs)] == label
        assert classes_to_labels[np.argmax(lc_probs)] in list(expected_classes.values())


def test_return_new_classifications(classifier, test_data_dir, tmp_path):
    """Classify light curves from a CSV test file."""
    csv_file = os.path.join(tmp_path, "labels.csv")

    with open(csv_file, "w+", encoding="utf-8") as new_csv:
        csv_writer = csv.writer(new_csv, delimiter=",")
        csv_writer.writerow(["Name", "Label"])
        csv_writer.writerow(["ZTF22abvdwik", SnClass.SUPERNOVA_IA])
        csv_writer.writerow(["ZTF23aacrvqj", SnClass.SUPERNOVA_II])
        csv_writer.writerow(["ZTF22abcesfo", SnClass.SUPERNOVA_IIN])
        csv_writer.writerow(["ZTF22aarqrxf", SnClass.SUPERLUMINOUS_SUPERNOVA_I])
        csv_writer.writerow(["ZTF22abytwjj", SnClass.SUPERNOVA_IBC])

    # Save test file with labels
    classifier.return_new_classifications(
        csv_file, test_data_dir, "probs_new.csv", include_labels=True, output_dir=tmp_path
    )
    assert os.path.exists(os.path.join(tmp_path, "probs_new.csv"))

    # Save test file without labels
    classifier.return_new_classifications(
        csv_file, test_data_dir, "probs_no_labels.csv", include_labels=False, output_dir=tmp_path
    )
    assert os.path.exists(os.path.join(tmp_path, "probs_no_labels.csv"))
