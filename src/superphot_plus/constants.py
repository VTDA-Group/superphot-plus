"""A collection of constants used by many modules in the package."""

import jax.numpy as jnp
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

PRIOR_MEANS = jnp.array(
    [
        0.0,
        0.0052,
        1.1391,
        0.0,
        0.5990,
        1.4296,
        -1.5364,
        1.0607,
        1.0424,
        1.0075,
        0.999 + 8.9289e-5,
        0.9663,
        0.5488,
        0.8606,
    ]
)
PRIOR_SIGMAS = jnp.array(
    [
        0.5,
        1.5 * 0.000336,
        1.5 * 0.1719,
        50.0,
        1.5 * 0.2073,
        1.5 * 0.1003,
        0.2691,
        1.5 * 0.1544,
        0.0026,
        0.0139,
        1.5 * 4.5055e-05,
        0.0128,
        0.0553,
        0.0388,
    ]
)