#Load h5py file module
import hyperspy.api as hs
import h5py

def loadh5py(i):
	fpdfile = h5py.File(i,'r') #find data file in a read only format
	data = fpdfile['fpd_expt']['fpd_data']['data'][:]
	im = hs.signals.Image(data[:,:,0,:,:])
	return im
