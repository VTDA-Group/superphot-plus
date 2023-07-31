# superphot-plus

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

Superphot+ is an end-to-end package that imports supernova photometry, fits light curves to an empirical model, and performs subsequent classification and source parameter estimation. It dramatically expands on the functionalities of the package Superphot[^1], with multiple implemented sampling alternatives, including dynesty, stochastic variational inference, and NUTS. Superphot+ takes advantage of the JAX backend to speed up runtime.

Superphot+ includes functionalities to both generate simulated light curves, and import existing ZTF photometry from both ANTARES and ALeRCE. Classification functions by default label fitted light curves as one of SN Ia, SN II, SN IIn, SLSN-I, and SN Ibc, but alternative pre-trained models and classification labels can be substituted.

Superphot+ is the underlying package used in multiple real-time ANTARES classification filters[^2], as well as the ELASTICC challenge[^3].


[^1] https://github.com/griffin-h/superphot/tree/master/superphot

[^2] https://antares.noirlab.edu/filters

[^3] https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ 


## Getting started

To install this package for development use:

```
$ git clone http://github.com/lincc-frameworks/superphot-plus
$ cd superphot-plus
$ pip install -e .
$ pip install -e ".[dev]"
```

You can then run `$ pytest` to verify that all dependencies are correct,
and your environment should be ready for superphot-plussing!
