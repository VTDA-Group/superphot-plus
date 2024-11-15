"""This module contains scripts to plot sampling results."""
import os

import pacmap
import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

from superphot_plus.plotting.format_params import (
    param_labels,
)

from superphot_plus.plotting.utils import gaussian
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.surveys.surveys import Survey