import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import h5py

def loadh5py(i):
	fpdfile = h5py.File(i,'r') #find data file in a read only format
	data = fpdfile['fpd_expt']['fpd_data']['data'][:]
	im = hs.signals.Image(data[:,:,0,:,:])
	return im

#Centre of Mass calculation with thresholding from Magnus
def centre_of_disk_centre_of_mass(
        image,
        threshold=None):
    if threshold == None:
#            threshold = (image.max()-image.min())*0.5
        threshold = np.mean(image, dtype='float64') * (45/2)
#       The mean of the data set being used to write this is roughly
#       1/45 the max value. This threshold is too low and means that
#       Lower intensity features are interfering with the creation
#       Of the boolean disk
        
    image[image<threshold] = 0
    image[image>threshold] = 1
    booleanArray = image.astype(bool)
    disk_centre = sp.ndimage.measurements.center_of_mass(booleanArray)

    return(disk_centre)
          

#centre profile calculation from StackOverflow
def radial_profile(data,centre):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    r = r.astype(int)
    r_max = np.max(r)
    
#    rings, radius = np.histogram(r, weights = data, bins = r_max)
#    radialProfile = rings / radius[1:]
#   probelm with the rings and radius being different shapes - (61,)
#   and (62,)
#    plt.plot(radius[1:],rings)
#    plt.show()
    tbin =  np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())   
    radialProfile = tbin / nr

    return radialProfile
    	
#integrate in a range
def pixel(image,centre,s,e):
    rad = radial_profile (image, centre)
    sumRange = sp.integrate.simps(rad[s:e])
    return sumRange
    
def calibration(centre, im = None, scale = None):
#	This function takes the raw data input & crops the data set to the last 14 rows
#	This assumes that bulk STO is all that exists in these 14 rows. It then sums
#	Over the 0th and 1st axes into a single	.tif image that we can put into ImageJ
#   To get a "scale". It then applies the calibration to the data and returns it
#   Unchanged other than the addition of the calibration
	
#   If no data is inputted, it asks the user for the name of the data file
	if im is None:	
		data = input("Please enter the name of data file as a string: ")
		im = loadh5py(data)

#   If no scale is inputted, the data is converted to a .tif image and saved
	if scale is None:
		rawSTO = im[:,50:]
		rawSumSTO = rawSTO.sum(0).sum(0)
		saveName = input("Please enter the name of the save file as a string: ")
		rawSumSTO32 = rawSumSTO.change_dtype('float32')
		rawSumSTO32.save(saveName + ".tif")
		print("Please use ImageJ to load the .tif file and find the radius of the STO Laue zone")
		print("The diameter of the STO Laue zone is ...mrad")
		scale = input("Scale found from ImageJ: ")
	
# 	This takes the input of the number of pixels per "unit" and applies this calibration to
#	The dataset and centres the axes

	a2 = im.axes_manager[2]
	a2.scale = scale
	a2.units = "mrad"
	a2.offset = -centre[0]
	
	a3 = im.axes_manager[3]
	a3.scale = scale
	a3.units = "mrad"
	a3.offset = -centre[1]
	return im

def gaussian_fit():
    pass
