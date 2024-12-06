"""A collection of constants used by many modules in the package."""

import numpy as np

# Nested sampling parameters
MAX_ITER = 5000
DLOGZ = 0.4
NLIVE = 50

# Numpyro parameters
PAD_SIZE = 30

# Classifier parameters
INPUT_DROPOUT_FRAC = 0.2
HIDDEN_DROPOUT_FRAC = 0.5
EPOCHS = 500
VALID_RATIO = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_FOLDS = 20
