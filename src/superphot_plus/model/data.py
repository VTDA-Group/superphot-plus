from dataclasses import dataclass
from typing import List

import numpy as np
from torch.utils.data import TensorDataset


@dataclass
class ZtfData:
    """Holds raw ZTF object data."""

    names: List[int]
    labels: List[int]
    redshifts: List[int]

    def __iter__(self):
        return iter((self.names, self.labels, self.redshifts))


@dataclass
class TrainData:
    """Holds train and validation datasets."""

    train_dataset: TensorDataset
    val_dataset: TensorDataset

    def __iter__(self):
        return iter((self.train_dataset, self.val_dataset))


@dataclass
class TestData:
    """Holds information about testing data."""

    test_features: np.ndarray
    test_classes: np.ndarray
    test_names: np.ndarray
    test_group_idxs: List[int]

    def __iter__(self):
        return iter(
            (
                self.test_features,
                self.test_classes,
                self.test_names,
                self.test_group_idxs,
            )
        )
