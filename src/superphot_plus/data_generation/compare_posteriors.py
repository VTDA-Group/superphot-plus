import numpy as np
import matplotlib.pyplot as plt
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.data_generation.mosfit import import_slsn_mosfit
from superphot_plus.samplers.numpyro_sampler import NumpyroSampler, run_mcmc, trunc_norm
from superphot_plus.surveys.surveys import Survey
from superphot_plus.plotting.lightcurves import plot_sampling_lc_fit
from sklearn.linear_model import LinearRegression #Only necessary for Ashley's bad method
plotting = False

# Build a training set...
param_train = np.zeros((10,4)) # 10 = num of samples, 4 = num of parameters
mean_train = np.zeros((10,15)) #10 = num of samples, 15 = num of param

for realization in np.arange(10):
	times, flux, err, bands, bfield, pspin, mejecta, vejecta = import_slsn_mosfit('slsn.json', realization=int(realization+1))

	err = err/np.max(flux)
	flux = flux / np.max(flux)

	lc = Lightcurve(times, flux, err, bands)

	sampler = NumpyroSampler()
	posterior_samples = sampler.run_single_curve(
		lc,
		rng_seed=4,
		priors=Survey.ZTF().priors,
		sampler="svi",
	)
	param_train[int(realization),:] = [bfield, pspin, mejecta, vejecta]
	mean_train[int(realization),:] = posterior_samples.sample_mean()
	if plotting:
		plot_sampling_lc_fit('r'+str(realization), './', times, flux, err, bands, 
		posterior_samples.samples, ['r','g'], 'r','svi')

for i, y in enumerate(param_train.T):
	X = mean_train
	reg = LinearRegression().fit(X, y)
	# Note that this will always be ~0...we have a lot of free param :) 
	print(np.sum(reg.predict(X)-y))




