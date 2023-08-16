"""A collection of constants used by many modules in the package."""

import numpy as np

# Nested sampling parameters
MAX_ITER = 5000
DLOGZ = 0.5
NLIVE = 50

# Numpyro parameters
PAD_SIZE = 30

# Classifier parameters
INPUT_DROPOUT_FRAC = 0.2
HIDDEN_DROPOUT_FRAC = 0.5
SEED = 9876
EPOCHS = 500
VALID_RATIO = 0.9
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_FOLDS = 20

# New classifications parameters
TRAINED_MODEL_PARAMS = (13, 5, 128, 3)
MEANS_TRAINED_MODEL = np.array(
    [
        5.18928404e-03,
        1.27960653e00,
        7.16663998e-01,
        1.59438422e00,
        -1.45215889e00,
        9.54225774e-01,
        1.04234314e00,
        1.00733255e00,
        9.99977873e-01,
        9.66049977e-01,
        5.62890503e-01,
        8.63290836e-01,
        -6.27014662e00,
    ]
)

STDDEVS_TRAINED_MODEL = np.array(
    [
        5.00371052e-04,
        3.17145223e-01,
        3.93625733e-01,
        2.58445333e-01,
        3.72614525e-01,
        2.15892037e-01,
        2.44030784e-03,
        1.27667116e-02,
        5.11363683e-05,
        1.18290613e-02,
        6.24454144e-02,
        4.99345471e-02,
        8.98741848e-01,
    ]
)

# Plotting parameters
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
