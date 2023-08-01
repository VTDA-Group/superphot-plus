# superphot-plus

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

Superphot+ is an end-to-end package that imports supernova photometry, fits light curves to an empirical model, and performs subsequent classification and source parameter estimation. It dramatically expands on the functionalities of the package Superphot<sup>[1](#note1)</sup>, with multiple implemented sampling alternatives, including dynesty, stochastic variational inference, and NUTS. Superphot+ takes advantage of the JAX backend to speed up runtime.

Superphot+ includes functionalities to both generate simulated light curves, and import existing ZTF photometry from both ANTARES and ALeRCE. Classification functions by default label fitted light curves as one of SN Ia, SN II, SN IIn, SLSN-I, and SN Ibc, but alternative pre-trained models and classification labels can be substituted.

Superphot+ is the underlying package used in multiple real-time ANTARES classification filters<sup>[2](#note2)</sup>, as well as the ELASTICC challenge<sup>[3](#note3)</sup>.


<a name="note1"><sup>1</sup></a> https://github.com/griffin-h/superphot/tree/master/superphot

<a name="note2"><sup>2</sup></a> https://antares.noirlab.edu/filters

<a name="note3"><sup>3</sup></a> https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ 


## Getting started

To install this package for development use:

```
$ git clone http://github.com/lincc-frameworks/superphot-plus
$ cd superphot-plus
$ python3 -m venv venv
$ source venv/bin/activate
$ python -m pip install -e ".[dev]"
```

You can then run `$ pytest` to verify that all dependencies are correct,
and your environment should be ready for superphot-plussing!
