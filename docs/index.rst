Superphot+
========================================================================================


Superphot+ is an end-to-end package that imports supernova photometry, fits light curves 
to an empirical model, and performs subsequent classification and source parameter 
estimation. It dramatically expands on the functionalities of the package Superphot [1]_,
with multiple implemented sampling alternatives, including dynesty, stochastic 
variational inference, and NUTS. Superphot+ takes advantage of the JAX backend to speed
up runtime.

Superphot+ includes functionalities to both generate simulated light curves, and import 
existing ZTF photometry from both ANTARES and ALeRCE. Classification functions by default 
label fitted light curves as one of SN Ia, SN II, SN IIn, SLSN-I, and SN Ibc, but 
alternative pre-trained models and classification labels can be substituted.

Superphot+ is the underlying package used in multiple real-time ANTARES classification 
filters [2]_, as well as the ELASTICC challenge [3]_.


.. [1] https://github.com/griffin-h/superphot/tree/master/superphot 
.. [2] https://antares.noirlab.edu/filters 
.. [3] https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/ 

.. toctree::
   :hidden:

   Home page <self>
   Notebooks <notebooks>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developers

   contributing
   API Reference <autoapi/index>