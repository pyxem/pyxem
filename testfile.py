import hyperspy.api as hs
import profiledata
import CoM
from numpy import unravel_index
import copy
from scipy.optimize import curve_fit

im = hs.load("test_hist.hdf5").as_image((0,2)) #load data
im2 = im.to_spectrum() #change signal type
im3 = im2.isig[70.:] #crop out centre beam
im4 = im3.inav[:,13:] #remove platinum
m = im4.create_model() #turns spectrum into a model

################################
powerlaw= hs.model.components.PowerLaw() #creates a powerlaw
m.append(powerlaw)

m.set_signal_range(70,88) #sets powerlaw range to channels 70-88
m.multifit() #fits powerlaw to all images 
m.reset_signal_range() #reset signal range
powerlaw.set_parameters_not_free() #fixes powerlaw shape

im5 = im4 - m.as_signal() #Removes the powerlaw as background
################################

gaussian = hs.model.components.Gaussian() #creates a gaussian component

################################
# Finds the probe position with the highest total intensity which 
# Will be used to set the initial conditions for the Gaussian fit
im6 = im5.sum(-1)
maxSumIndex = unravel_index(im6.data.argmax(),im6.data.shape)
maxSignalPosition = im5.inav[maxSumIndex[1],maxSumIndex[0]]
m3 = maxSignalPosition.create_model()
m3.append(gaussian)
m3.set_signal_range(90,120)
gaussian.centre.value = 105
m3.fit(fitter = "mpfit")

################################
# Fit a Gaussian to the whole data set using the values from the 
# Strongest signal as initial conditions and apply bounds on the 
# Extremes

m2 = im5.create_model()
m2.append(gaussian) 
m2.set_signal_range(90,120)

gaussian.centre.bmin = 100
gaussian.centre.bmax = 110
gaussian.centre.assign_current_value_to_all()

gaussian.A.bmin = 0
gaussian.A.assign_current_value_to_all()

gaussian.sigma.bmin = 0.1
gaussian.sigma.bmax = 15
gaussian.sigma.assign_current_value_to_all()

m2.multifit(fitter="mpfit", bounded = True) 
m2.reset_signal_range()

###############################

centreGaussian = gaussian.centre.as_signal() #create a signal of all centre values
aGaussian = gaussian.A.as_signal() # create a signal of all A values
aGaussianMax = aGaussian.max(axis=1).max(axis=0).data # find the max value of A
aData = copy.deepcopy(aGaussian.data) # create a copy of the A signal data
aData[aData < 20] = 0. #threshold the A data to remove low values
# aData[aData < (aGaussianMax[0] * 0.2)]
centreData = copy.deepcopy(centreGaussian.data) # create a copy of the centre data
centreData[aData == 0.] = 0. #threshold the centre data for low intensity signals
del aData # bookeeping
centreData[centreData == 110.] = 0.
centreData[centreData == 100.] = 0. #remove signals that are just saturated values
centreGaussian.data[centreData == 0.] = 0. #apply the thresholded data map to the original signals
aGaussian.data[centreData == 0.] = 0. 
del centreData #bookeeping

##############################

centreMeans = centreGaussian.data.mean(axis = 1)
aMeans = aGaussian.data.mean(axis = 1)
aMeans[centreMeans <= 100] = 0.
centreMeans [aMeans == 0.] = 0.

##############################
# Fit a Gaussian to the intensities to see how they vary
# p0 is the initial guess for the fitting coefficients
p0 = [114., 15., 8.]
counting = list(range(51))
coeff, var_matrix = curve_fit(CoM.gauss, counting, aMeans, p0=p0)
dataFit = gauss(counting, *coeff)

plt.plot(counting, aMeans, 'ro', label='Data')
plt.plot(counting, dataFit, 'bo', label='Fit')

print 'mean = ', coeff[1]
print 'std = ', coeff[2]
plt.show()
