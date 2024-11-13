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

    create_dirs: bool = True
    relative_dirs: bool = True
    
    # data options
    data_dir: str = 'data'
    transient_data_fn: str = "transients"
    
    # sampling options
    sampler_results_fn: str = "sampler_results"
    sampler: str = "dynesty"
    chisq_cutoff: float = 1.2
    
    # plotting options
    figs_dir: str = 'figs'
    metrics_dir: str = 'metrics'
    fit_plots_dir: str = 'fits'
    cm_dir: str = 'confusion_matrices'
    
    # logging options
    logging: bool = False
    log_fn: str = 'results.log'
    plot: bool = False
    
    # classification options
    load_checkpoint: Optional[str] = None
    models_dir: str = 'models'
    model_type: str = "LightGBM"
    probs_dir: str = 'probabilities'
    probs_fn: str = 'probs.csv'
    prefix: str = 'best-model'
    device = torch.device("cpu")
    include_redshift: bool = False
    fits_per_majority: int = 5
    
    # single-class options
    target_label: Optional[str] = None
    prob_threshhold: Optional[float] = 0.5
    
    # multi-class options
    allowed_types: list[str] = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]
    
    # MLP parameters
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    neurons_per_layer: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    learning_rate: Optional[float] = None
    
    # general training
    n_folds: int = 1
    normalization_means: Optional[List[float]] = None
    normalization_stddevs: Optional[List[float]] = None
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    
    # reproducibility options
    random_seed: int = 42
    
    
    def __post_init__(self):
        """Ensure subdirectory structure exists."""
        if self.relative_dirs:
            self.transient_data_fn = os.path.join(self.data_dir, self.transient_data_fn)
            self.models_dir = os.path.join(self.data_dir, self.models_dir)
            self.sampler_results_fn = os.path.join(self.data_dir, self.sampler_results_fn)

            self.metrics_dir = os.path.join(self.figs_dir, self.metrics_dir)
            self.fit_plots_dir = os.path.join(self.figs_dir, self.fit_plots_dir)
            self.cm_dir = os.path.join(self.figs_dir, self.cm_dir)
            self.wrongly_classified_dir = os.path.join(self.figs_dir, self.wrongly_classified_dir)

            self.log_fn = os.path.join(self.data_dir, self.log_fn)
            self.probs_dir = os.path.join(self.data_dir, self.probs_dir)
            self.probs_fn = os.path.join(self.probs_dir, self.probs_fn)
    
        if self.create_dirs:
            for x_dir in [
                self.models_dir, self.figs_dir, self.metrics_dir,
                self.fit_plots_dir, self.cm_dir, self.wrongly_classified_dir,
                self.probs_dir
            ]:
                os.makedirs(x_dir, exist_ok=True)
                
        if self.include_redshift:
            self.skipped_params = [3,14]
        else:
            self.skipped_params = [0,3,14]
            
        if n_folds < 1:
            raise ValueError("Number of K-folds must be positive")
        if chisq_cutoff <= 0.0:
            raise ValueError("chisq cutoff must be positive")

        
    def set_non_tunable_params(self, input_dim, output_dim, norm_means, norm_stddevs):
        """Adds information about the params that are not tunable."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalization_means = norm_means
        self.normalization_stddevs = norm_stddevs

    def write_to_file(self, file: str):
        """Save configuration data to a YAML file."""
        args = dataclasses.asdict(self)
        encoded_string = yaml.dump(args, sort_keys=False, default_flow_style=False)
        with open(file, "w", encoding="utf-8") as file_handle:
            file_handle.write(encoded_string)
            
    def update(self, **kwargs):
        """Update config attributes."""
        for (k,v) in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_file(cls, file: str) -> Self:
        """Load configuration data from a YAML file."""
        with open(file, "r", encoding="utf-8") as file_handle:
            metadata = yaml.safe_load(file_handle)
            metadata['prefix'] = file.split("/")[-1][:-5]
            return cls(**metadata)
