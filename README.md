# superphot-plus

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[DOI]: TODO
[![PyPI](https://img.shields.io/pypi/v/superphot-plus?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/superphot-plus/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/vtda-group/superphot-plus/smoke-test.yml)](https://github.com/vtda-group/superphot-plus/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/vtda-group/superphot-plus/branch/main/graph/badge.svg)](https://codecov.io/gh/vtda-group/superphot-plus)
[![Read the Docs](https://img.shields.io/readthedocs/superphot-plus)](https://superphot-plus.readthedocs.io/)
[![benchmarks](https://img.shields.io/github/actions/workflow/status/vtda-group/superphot-plus/asv-main.yml?label=benchmarks)](https://vtda-group.github.io/superphot-plus/)

Superphot+ is an end-to-end package that imports supernova photometry, fits light curves to an empirical model, and performs subsequent classification and source parameter estimation. It dramatically expands on the functionalities of the package Superphot<sup>[1](#note1)</sup>, with multiple implemented sampling alternatives, including dynesty, stochastic variational inference, and NUTS. Superphot+ takes advantage of the JAX backend to speed up runtime.

Superphot+ includes functionalities to both generate simulated light curves, and import existing ZTF photometry from both ANTARES and ALeRCE. Classification functions by default label fitted light curves as one of SN Ia, SN II, SN IIn, SLSN-I, and SN Ibc, but alternative pre-trained models and classification labels can be substituted.

Superphot+ is the underlying package used in multiple real-time ANTARES classification filters<sup>[2](#note2)</sup>, as well as the ELASTICC challenge<sup>[3](#note3)</sup>.


<a name="note1"><sup>1</sup></a> https://github.com/griffin-h/superphot/tree/master/superphot

<a name="note2"><sup>2</sup></a> https://antares.noirlab.edu/filters

<a name="note3"><sup>3</sup></a> https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ 


See [ReadTheDocs](https://superphot-plus.readthedocs.io/) for more information.

## Getting started

To install this package for development use:

```
$ git clone http://github.com/vtda-group/superphot-plus
$ cd superphot-plus
$ python3 -m venv venv
$ source venv/bin/activate
$ python -m pip install -e ".[dev]"
```

To install all optional dependencies, including those for inference, plotting, loading
data from alert brokers, model tuning and benchmarking use 
`pip install .[dev,data-generation,sampling,plotting,tuning,benchmarking]`.

You can then run `$ pytest` to verify that all dependencies are correct,
and your environment should be ready for superphot-plussing!

## Working with the models

Inside the `scripts` directory you can find four scripts which focus on the training and tuning of the supernovae classifier and the physical parameter regressor.

Below is a brief guide on how to get started with model training and tuning. To get a full specification of any of the available scripts, run `$ python <script_name.py> --help`.

### Classifier

Both workflows assume training data is loaded from `training_set.csv` (unless specified using the **--input_csvs** argument) and that the respective posterior fits are located under the classification directory, on a subdirectory named `{sampler}_fits`.

To train the classifier run `train_classifier.py`.

```
$ python train_classifier.py --config_name superphot-plus
```

Specify a configuration file name. This file is in the YAML format and contains the configuration of the neural network, as specified in [ModelConfig](https://github.com/VTDA-Group/superphot-plus/blob/main/src/superphot_plus/model/config.py). It should be placed in the classification directory, under "models". Here is a very simple configuration example:

```yaml
neurons_per_layer: 256
num_hidden_layers: 10
goal_per_class: 100
num_epochs: 500
batch_size: 32
learning_rate: 0.0001
```

To get an optimized set of hyperparameters run `tune_classifier.py` and specify the number of neural network configurations to be sampled by Ray.

```
$ python tune_classifier.py --num_hp_samples 5
```

### Regressor

Both workflows assume the posterior samples and supernovae property data are located under the mosfit directory, on subdirectories named `posteriors` and `properties`, respectively. 

To train the regressor run `train_mosfit.py`.

```
$ python train_mosfit.py --parameter bfield
```

You will need to specify the physical property name. Similarly to what is described for the classification task, there should be a file in the mosfit directory, under "models", specifying the desired neural network configuration. This file must be named after the physical parameter.

To get an optimized set of hyperparameters specify the physical property name and the number of neural network configurations to be sampled by Ray.

```
$ python tune_mosfit.py --parameter bfield --num_hp_samples 10
```

## Contributing

[![GitHub issue custom search in repo](https://img.shields.io/github/issues-search/vtda-group/superphot-plus?color=purple&label=Good%20first%20issues&query=is%3Aopen%20label%3A%22good%20first%20issue%22)](https://github.com/vtda-group/superphot-plus/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

See the [contribution guide](https://superphot-plus.readthedocs.io/en/latest/contributing.html) on ReadTheDocs.
