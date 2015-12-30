import hyperspy.api as hs
import profiledata
import CoM
from numpy import unravel_index

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

m2 = im5.create_model()
m2.append(gaussian) #append gaussian to the model

m2.set_signal_range(90,120) #set the range of the gaussian
#Set bounds and initial values for the A,sigma and centre of Gaussian

gaussian.centre.bmin = 100
gaussian.centre.bmax = 110
gaussian.centre.assign_current_value_to_all()

gaussian.A.bmin = 0
gaussian.A.assign_current_value_to_all()

gaussian.sigma.bmin = 0.1
gaussian.sigma.bmax = 15
#gaussian.sigma.bmax = 25
gaussian.sigma.assign_current_value_to_all()

m2.multifit(fitter="mpfit", bounded = True) #fit the Gaussian with the aforementioned bounds
m2.reset_signal_range()

centreGaussian = gaussian.centre.as_signal()
aGaussian = gaussian.A.as_signal()
aGaussianMax = aGaussian.max(axis=1).max(axis=0).data
'''aGaussian.data[aGaussian.data < (0.6*aGaussianMax)] = 0.
centreGaussian.data[aGaussian.data == 0.] = 0.''' ###The non-fitting of some of the data set is caused by these two lines...

