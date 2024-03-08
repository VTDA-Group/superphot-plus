import dataclasses
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import yaml
from typing_extensions import Self


# pylint: disable=too-many-instance-attributes
@dataclass
class SuperphotConfig:
    """Holds information about the specific training
    configuration of a model. The default values are
    sampled by ray tune for parameter optimization."""

    create_dirs: Optional[bool] = True
    relative_dirs: Optional[bool] = True
    # File paths
    data_dir: Optional[str] = "."
    fits_dir: Optional[str] = "fits"
    input_csvs: Optional[list] = field(default_factory=lambda: [ "training_set.csv", ])
    models_dir: Optional[str] = 'models'
    
    figs_dir: Optional[str] = 'figs'
    metrics_dir: Optional[str] = 'metrics'
    fit_plots_dir: Optional[str] = 'fits'
    cm_dir: Optional[str] = 'confusion_matrices'
    wrongly_classified_dir: Optional[str] = 'wrongly_classified'
    
    log_fn: Optional[str] = 'results.log'
    probs_dir: Optional[str] = 'probabilities'
    probs_fn: Optional[str] = 'probs_%d.csv'
    prefix: Optional[str] = 'best-model'
    
    # single-target options
    target_label: Optional[str] = None
    prob_threshhold: Optional[float] = 0.5
    
    # Nontunable parameters
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None

    normalization_means: Optional[List[float]] = None
    normalization_stddevs: Optional[List[float]] = None

    # Tunable parameters
    neurons_per_layer: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    goal_per_class: Optional[int] = 4500
    num_folds: Optional[int] = None
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None

    best_val_loss: Optional[float] = None

    device = torch.device("cpu")

    def __post_init__(self):
        """Ensure subdirectory structure exists."""
        if self.relative_dirs:
            self.fits_dir = os.path.join(self.data_dir, self.fits_dir)
            self.input_csvs = [
                os.path.join(self.data_dir, x) for x in self.input_csvs
            ]
            self.models_dir = os.path.join(self.data_dir, self.models_dir)
            self.figs_dir = os.path.join(self.data_dir, self.figs_dir)

            self.metrics_dir = os.path.join(self.figs_dir, self.metrics_dir)
            self.fit_plots_dir = os.path.join(self.figs_dir, self.fit_plots_dir)
            self.cm_dir = os.path.join(self.figs_dir, self.cm_dir)
            self.wrongly_classified_dir = os.path.join(self.figs_dir, self.wrongly_classified_dir)

            self.log_fn = os.path.join(self.data_dir, self.log_fn)
            self.probs_dir = os.path.join(self.data_dir, self.probs_dir)
            self.probs_fn = os.path.join(self.probs_dir, self.probs_fn)
    
        if self.create_dirs:
            for x_dir in [
                self.fits_dir, self.models_dir, self.figs_dir, self.metrics_dir,
                self.fit_plots_dir, self.cm_dir, self.wrongly_classified_dir,
                self.probs_dir
            ]:
                os.makedirs(x_dir, exist_ok=True)

        
    def set_non_tunable_params(self, input_dim, output_dim, norm_means, norm_stddevs):
        """Adds information about the params that are not tunable."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalization_means = norm_means
        self.normalization_stddevs = norm_stddevs

    def set_best_val_loss(self, best_val_loss):
        """Sets the best validation loss from training."""
        self.best_val_loss = best_val_loss

    def write_to_file(self, file: str):
        """Save configuration data to a YAML file."""
        args = dataclasses.asdict(self)
        encoded_string = yaml.dump(args, sort_keys=False, default_flow_style=False)
        with open(file, "w", encoding="utf-8") as file_handle:
            file_handle.write(encoded_string)

    @classmethod
    def from_file(cls, file: str) -> Self:
        """Load configuration data from a YAML file."""
        with open(file, "r", encoding="utf-8") as file_handle:
            metadata = yaml.safe_load(file_handle)
            metadata['prefix'] = file[:-5]
            metadata['relative_dirs'] = False
            return cls(**metadata)
