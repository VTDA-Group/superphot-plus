import numpy as np
import json

def import_slsn_mosfit(mosfit_fn, realization=1):
	'''
	This is specifically for SLSN code, since I happen to know
	which parameters actually matter
	'''

	# A note from Ashley: One mosfit file = multiple realizations, each a dif. model

	# Probably not the right way to do this!


	f = open('../../../data/'+mosfit_fn)
	data = json.load(f)
	f.close()
	top_key = [*data][0]
	t = []
	flux = []
	err = []
	b = []

	for datum in data[top_key]['photometry']:
		if realization==int(datum['realization']):
			if 'upperlimit' in datum:
				#Ignore upper limits
				continue
			else:
				t.append(float(datum['time']))
				flux.append(10.**((float(datum['magnitude'])+48.6)/-2.5))
				err.append(flux[-1] * float(datum['e_magnitude']))
				b.append(datum['band'])
	# Also grab parameters...
	# Note that this will ALWAYS be the 0th model, so it is fine to hard code
	for my_realization in data[top_key]['models'][0]['realizations']:
		if int(my_realization['alias']) == realization:
			bfield = my_realization['parameters']['Bfield']['value']
			pspin = my_realization['parameters']['Pspin']['value']
			mejecta = my_realization['parameters']['mejecta']['value']
			vejecta = my_realization['parameters']['vejecta']['value']
	t = np.asarray(t, dtype=float)
	flux = np.asarray(flux, dtype=float)
	err = np.asarray(err, dtype=float)
	b = np.asarray(b, dtype=str)

	return t, flux, err, b, bfield, pspin, mejecta, vejecta


