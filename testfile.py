import hyperspy.api as hs
import profiledata
import CoM
import numpy as np
from numpy import unravel_index
import copy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#im=hs.load("test_hist.hdf5").as_image((0,2)) #load data
im = hs.load("test_hist_cali.hdf5").as_image((0,2)) #load data
im2 = im.to_spectrum() #change signal type
#im3 = im2.isig[70.:] #crop out centre beam
im3 = im2.isig[30.:] #crop out centre beam
im4 = im3.inav[:,13:] #remove platinum
m = im4.create_model() #turns spectrum into a model

################################
powerlaw= hs.model.components.PowerLaw() #creates a powerlaw
m.append(powerlaw)

#m.set_signal_range(70,88) #sets powerlaw range to channels 70-88
m.set_signal_range(30,38)
m.multifit() #fits powerlaw to all images 
m.reset_signal_range() #reset signal range
powerlaw.set_parameters_not_free() #fixes powerlaw shape

im5 = im4 - m.as_signal() #Removes the powerlaw as background
im5.save("test_hist_cali_powerlaw.hdf5")
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
#m3.set_signal_range(90,120)
#gaussian.centre.value = 105
m3.set_signal_range(40,50)
gaussian.centre.value = 45
m3.fit(fitter = "mpfit")
print 'Max signal parameters:'
print gaussian.centre.value
print gaussian.A.value
print gaussian.sigma.value

################################
# Fit a Gaussian to the whole data set using the values from the 
# Strongest signal as initial conditions and apply bounds on the 
# Extremes

m2 = im5.create_model()
m2.append(gaussian) 
#m2.set_signal_range(90,120)
m2.set_signal_range(40,50)

#centreMin = 100.
#centreMax = 110.
centreMin = 40.
centreMax = 50.
gaussian.centre.bmin = centreMin
gaussian.centre.bmax = centreMax
gaussian.centre.assign_current_value_to_all()

gaussian.A.bmin = 0.
gaussian.A.assign_current_value_to_all()

gaussian.sigma.bmin = 0.1
gaussian.sigma.bmax = 10.
gaussian.sigma.assign_current_value_to_all()

m2.multifit(fitter="mpfit", bounded = True) 
m2.reset_signal_range()

###############################

centreGaussian = gaussian.centre.as_signal() #create a signal of all centre values
aGaussian = gaussian.A.as_signal() # create a signal of all A values
sigmaGaussian = gaussian.sigma.as_signal() #create a signal of all sigma values

aGaussianMax = aGaussian.max(axis=1).max(axis=0).data # find the max value of A
aData = copy.deepcopy(aGaussian.data) # create a copy of the A signal data
#aData[aData < 20] = 0. #threshold the A data to remove low values
aData[aData < (aGaussianMax[0] * 0.2)]
centreData = copy.deepcopy(centreGaussian.data) # create a copy of the centre data
centreData[aData == 0.] = 0. #threshold the centre data for low intensity signals
del aData # bookeeping

centreData[centreData == centreMax] = 0.
centreData[centreData == centreMin] = 0. #remove signals that are just saturated values
centreGaussian.data[centreData == 0.] = 0. #apply the thresholded data map to the original signals
aGaussian.data[centreData == 0.] = 0. 
sigmaGaussian.data[centreData == 0.] = 0.
del centreData #bookeeping

##############################

centreMean = centreGaussian.data.mean(axis = 1) #find the means along the axis perp. to growth direction
aMean = aGaussian.data.mean(axis = 1)
sigmaMean = sigmaGaussian.data.mean(axis = 1)
#aMean2 = copy.deepcopy(aMean)
#aMean2[centreMean <= centreMin] = 0.
#centreMean [aMean2 == 0.] = 0.
#del aMean2
centreMeanThresh = [x for x in centreMean if x>= centreMin]
centreMeanThreshCali = CoM.calibration(centreMeanThresh, scale = None)

##############################
# Fit a Gaussian to the intensities to see how they vary
# pInitial is the initial guess for the fitting coefficients
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

#pInitial = [114., 15., 8.]
pInitial = [50.,15.,5.]
lengthOfFitGauss = list(range(len(aMean)))
coeffGauss, varmatrix = curve_fit(gauss, lengthOfFitGauss, aMean, p0=pInitial)
dataFitGauss = gauss(lengthOfFitGauss, *coeffGauss)

'''plt.figure(0)
plt.plot(lengthOfFitGauss, aMean, 'ro', label='Data')
plt.plot(lengthOfFitGauss, dataFitGauss, 'bo', label='Fit')
plt.show()
plt.legend()'''

print 'Most intense Laue zone at row ', coeffGauss[1]
print 'Sigma = ', coeffGauss[2]


#############################
#Fit a polynomial to the thresholded centreMean then translate that back to centreMean

lengthOfFitPoly = list(range(len(centreMeanThreshCali)))
poly = np.polynomial.polynomial.polyfit(lengthOfFitPoly, centreMeanThreshCali, 4, rcond=None, full=False, w=None)

def square(list): return[i**2 for i in list]
def cube(list): return[i**3 for i in list]
def fourth(list): return[i**4 for i in list]
def multiple(poly, list): return[poly*i for i in list]

polyFit = poly[0] +multiple(poly[1], lengthOfFitPoly) + multiple(poly[2],square(lengthOfFitPoly)) + multiple(poly[3], cube(lengthOfFitPoly)) + multiple(poly[4], fourth(lengthOfFitPoly))

'''plt.figure(1)
plt.plot(lengthOfFitPoly,centreMeanThreshCali,'ro', label = 'Data')
plt.plot(lengthOfFitPoly, polyFit, 'bo', label = 'Fit')
plt.show()
plt.legend()'''

errMeanFit = [centreMeanThreshCali[x] - polyFit[x] for x in range(len(centreMeanThreshCali))]

# Changes polyFit into a comparable dataset to centreMean
#preLaueIndex = [n for n, i in enumerate(centreMean) if i >= 100][0]
preLaueIndex = [n for n, i in enumerate(centreMean) if i >= centreMin][0]
postLaueIndex = preLaueIndex + len(centreMeanThreshCali)
centreMeanPoly = [0.] * (preLaueIndex)
listOfZeros = [0.] * (len(centreMean) - postLaueIndex)
centreMeanPoly.extend(polyFit)
centreMeanPoly.extend(listOfZeros)


#plt.plot(lengthOfFitGauss, centreMeanPoly, 'bo', label = 'Data')
#plt.plot(lengthOfFitGauss, centreMean, 'ro', label = 'Fit')
#plt.show()
#plt.legend()

############################
# This section looks at the sigma values, first will remove the 
# sigma value for the probe positions not in the film

sigmaMean2 = copy.deepcopy(sigmaMean)
zeroIndex = [x for x,y in enumerate(centreMeanPoly) if y==0.]
sigmaMean2[zeroIndex] = 0.
sigmaMean3 = [x for x in sigmaMean2 if x > 0] #This just "zooms" in on film sigma values


#Print out a series of useful things
print 'Polynomial Coefficients: ',poly[0],poly[1],poly[2],poly[3]
print 'Max value of centre position: ', max(centreMeanThreshCali)
print 'With intensity: ', aMean[centreMeanPoly.index(max(polyFit))]
print 'Min value of centre position: ', min(centreMeanThreshCali)
print 'With intensity: ', aMean[centreMeanPoly.index(min(polyFit))]
print 'Most intense HOLZ at: ', centreMeanPoly[aMean.tolist().index(max(aMean))]
