import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import h5py

#Centre of mass calculation already in Scipy
def centre_of_mass(image):
    centreOfMass = sp.ndimage.measurements.center_of_mass(image)
    return centreOfMass
    
#Centre of Mass calculation with thresholding from Magnus
def centre_of_disk_centre_of_mass(
        image,
        threshold=None):
    if threshold == None:
#            threshold = (image.max()-image.min())*0.5
        threshold = np.mean(image, dtype='float64')
    image[image<threshold] = 0
    image[image>threshold] = 1
    disk_centre = centre_of_mass(image)

    return(disk_centre)
          

#centre profile calculation from StackOverflow
def radial_profile(data,centre):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    r = r.astype(np.int)

    tbin =  np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    #for somevalues, tbin and nr are 0 leading to 1's everywhere       
    radialProfile = tbin / nr

    return radialProfile

def radial_profile_segment(data, centre, Threshold = None):
    y,x = np.indices((data.shape))
    r = np.sqrt((x-centre[0])**2 + (y-centre[1])**2)
    r = r.astype(np.int)
    
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialProfileSegment = tbin/nr
    if Threshold == None:
        Threshold = radialProfileSegment > 0.2*max(
        radialProfileSegment)
         
    radialProfileSegment[radialProfileSegment > Threshold] = 0
    
    if (0 in tbin) and (0 in nr):
        zeroIndex = tbin.index(0)
        radialProfileSegment[zeroIndex] = 0
       
    return radialProfileSegment
    
    	
#integrate in a range
def pixel(image,centre,s,e):
    rad = radial_profile (image, centre)
    sumRange = sp.integrate.simps(rad[s:e])
    return sumRange


#Summed intensities
def summed_intensity(dataSet):
    summedIntensity = dataSet.sum(-1).sum(-1).plot()
    return summedIntensity	

def single_profile(im,i,j):
    singleImage = im[i,j].data #printing radial profile for a particular image of interest

    centre = centre_of_mass(singleImage) #finds the centre of mass 
    rad = radial_profile (singleImage, centre) 

    plt.plot(rad[:])
    plt.show()
    return

def single_profile_threshold(im,i,j):
    singleImage = im[i,j].data
    centre = centre_of_disk_centre_of_mass(im.data, 
        threshold = None)
    rad = radial_profile(singleImage,centre)
    
    plt.plot(rad[:])
    plt.show()
    return
