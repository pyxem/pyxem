import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import h5py
import copy

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
    
    tbin =  np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())   
    radialProfile = tbin / nr

    return radialProfile
    	   
def diffraction_calibration(centre, im = None, scale = None):
#	This function calibrates the diffraction patters
#   It takes the raw data input & crops the data set to the last 14 rows
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
		print("The diameter of the STO Laue zone is 133.32 mrad")
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

def radial_profile_dataset(i, calibrationScale = None):
#   Creates a hfd5 of the summed radial profiles for an input
#   hfd5 file

    blankFileToSave = np.zeros((64,64,180)) 
#   the length of rad varies and a numpy array needs a 
#   fixed row length for each entry
    
    fpdfile = h5py.File(i,'r') 
#   find data file in a read only format

    data = fpdfile['fpd_expt']['fpd_data']['data'][:]
#   Opens the keys within the h5py file to access image

    im = hs.signals.Image(data[:,:,0,:,:])
#   Copies image from a hfd5 file into hyperspy image class
#   and sets the singular "z" axis to 0, reducing it to 
#   2 navigation (plane) and 2 signal axis

    centre = centre_of_disk_centre_of_mass(copy.deepcopy(
        im.data))[2:]
#   Passes a copy of a numpy array based on the entire image to the 
#   function in CoM to calculate the centre of mass
#   A copy is passed so that the original image isn't altered

    for i in range(0,64):
        for j in range (0,64):
#   Iterates over each individual 256x256 image (4096 in total)

            rad = radial_profile(im[i,j].data, centre)

            if len(rad) > 180:
                while len(rad) > 180:
                    rad = np.delete(rad, [len(rad)-1])
#           Shortens the array of rad to create a uniform 
#           Length          
          
            blankFileToSave[i,j] = rad

#           Passes the numpy array for each image alone with the
#           Average centre of mass calculated earlier to the 
#           Radial profiler in CoM and saves the profile data
#           In a multidimensional numpy array


    s = hs.signals.Image(blankFileToSave)
    s.axes_manager[1].scale = calibrationScale
    s.save("profileddataset.hdf5")        
#   Changes the numpy array with the profiles in it into a hyperspy
#   signal and saves it based on the chosen save name
    del im        
    del blankFileToSave
    del rad
    del centre
    del data
    del fpdfile
    return s	
	    
def _annular_dark_field_image(dataset, centre):
#   Returns a numpy array of the ADF that can be displayed as an image.
#   Input is the radially profiled dataset from the radial_profile_dataset function
    results = []
    for i in range (0,64):
        for i in range (0,64):
            radial_profile = image[i,j].data
            sum_range = sp.integrate.simps(radial_profile[120:180])
            results.append(sum_range)
    adf_array = np.array(results)
    adf_array = data.reshape(64,64)
    del results
    return adf_array
    
def _figure_compare(image_array, centring_method):
#   Input an array of images to be displayed. Initally lets assume we are going to have 14 images.
    a_gauss_bounded         = image_array[0]
    a_gauss_unbounded       = image_array[1]
    centre_gauss_bounded    = image_array[2]
    centre_gauss_unbounded  = image_array[3]
    sigma_gauss_bounded     = image_array[4]
    sigma_gauss_unbounded   = image_array[5]
    a_mean_bounded          = image_array[6]
    a_mean_unbounded        = image_array[7]
    centre_mean_bounded     = image_array[8]
    centre_mean_unbounded   = image_array[9]
    sigma_mean_bounded      = image_array[10]
    sigma_mean_unbounded    = image_array[11]
    adf_image               = image_array[12]
    adf_holz                = image_array[13]


    fig, axarr = plt.subplots(5,3, figsize=(5,10))
    axarr[0][0].imshow(a_gauss_bounded) 
    axarr[0][1].imshow(centre_gauss_bounded)
    axarr[0][2].imshow(sigma_gauss_bounded)

    axarr[1][0].imshow(a_gauss_unbounded)
    axarr[1][1].imshow(centre_gauss_unbounded)
    axarr[1][2].imshow(sigma_gauss_unbounded)
         
    axarr[2][0].imshow(a_mean_bounded)
    axarr[2][1].imshow(centre_mean_bounded) 
    axarr[2][2].imshow(sigma_mean_bounded)
     
    axarr[3][0].imshow(a_mean_unbounded) 
    axarr[3][1].imshow(centre_mean_unbounded)
    axarr[3][2].imshow(sigma_mean_unbounded)
    
    axarr[4][0].imshow(adf_image)
    axarr[4][1].imshow(adf_holz)
    
    fig.tight_layout()
    fig.savefig("compare_images" + centring_method + ".jpg")

