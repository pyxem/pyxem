import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import h5py	
import loaddata
import CoM
 
im = loaddata.loadh5py("default1.hdf5") #load data

###############################################################
###############################################################
centre = CoM.centre_of_disk_centre_of_mass(im.data)[2:]

def bounded_integration(image,bounds, centre):
    results = []
    for i in range(0,64):
	    for j in range(0,64):
	    	results.append(CoM.pixel(image[i,j].data,centre,
	    	bounds[0],bounds[1]))

    data = np.array(results)
    data = data.reshape(64,64)
    del results
    return data

bounds = [[0,30],[30,60],[60,90],[90,120],[120,150],[150,180]]
for i in range(0,6):
    images = bounded_integration(im,bounds[i], centre)
    ax = plt.subplot(2,3,i+1)
    plt.imshow(images)
    ax.set_title(str(bounds[i][0]) + "-" + str(bounds[i][1]) + "mrad") 
    
plt.show()

###############################################################
###############################################################






















#rescale central beam
#length = len(rad[:]) - 1 #-1 because it starts from 0
#threshold = 0.1 * np.amax(rad)
#for i in xrange(0, length):
#	if      rad[i] >= threshold:
#                rad[i] = 0

#plt.figure(1)
#plt.plot(rad[:]) #generates the plot of the radial profile for the single i$
#plt.show() #displays the single image.

#summedIntensity = im.sum(-1).sum(-1).plot()
#plt.show(summedIntensity)

#Attempts at removing the central beam, I hope
#thresholdedData1 = CoM.threshold_central_beam1(im, 20)
#thresholdedData2 = CoM.threshold_central_beam2(im, 0.02)

#summedIntensity1 = CoM.summed_intensity(thresholdedData1)
#summedIntensity2 = CoM.summed_intensity(thresholdedData2)
#plt.figure(3)
#plt.imshow(thresholdedData1)
#plt.figure(4)
#plt.imshow(thresholdedData2)
