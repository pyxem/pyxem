import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.ndimage.measurements import center_of_mass
import h5py
import copy
from numpy import unravel_index
from scipy.optimize import curve_fit
import os.path
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import math

def _set_metadata_from_hdf5(hdf5_file, signal):
    """Get microscope and scan metadata from fpd HDF5-file reference.
    Will set acceleration voltage and camera length, store it
    in the signal. Will set probe position axis scales.
    Operates in-place.

    Parameters
    ----------
    hdf5_file : hdf5 file reference(?)
        An opened fpd hdf5 file reference
    signal : HyperSpy signal instance
    """
    signal.metadata.add_node("Microscope")
    dm_data = hdf5_file['fpd_expt']['DM0']['tags']['ImageList']['TagGroup0']
    metadata_ref = dm_data['ImageTags']['Microscope Info']

    signal.metadata.Microscope['Voltage'] = metadata_ref['Voltage'].value
    signal.metadata.Microscope['Camera length'] = metadata_ref['STEM Camera Length'].value

    axis_scale_x = hdf5_file['fpd_expt']['fpd_data']['dim2'][0:2]
    axis_scale_y = hdf5_file['fpd_expt']['fpd_data']['dim1'][0:2]
    axis_units_x = hdf5_file['fpd_expt']['fpd_data']['dim2'].attrs['units']
    axis_units_y = hdf5_file['fpd_expt']['fpd_data']['dim1'].attrs['units']

    signal.axes_manager[0].scale = axis_scale_x[1]-axis_scale_x[0]
    signal.axes_manager[1].scale = axis_scale_y[1]-axis_scale_y[0]
    signal.axes_manager[0].units = axis_units_x
    signal.axes_manager[1].units = axis_units_y
    
def _set_hardcoded_metadata(signal):
    """Sets some metadata values, like axis names.
    Operates in-place.
    
    Parameters
    ----------

    """
    signal.axes_manager[0].name = "Probe x"
    signal.axes_manager[1].name = "Probe y"
    signal.axes_manager[2].name = "Detector x"
    signal.axes_manager[3].name = "Detector y"

def load_fpd_dataset(filename, x_range=None, y_range=None):
    """Load data from a fast-pixelated data HDF5-file.
    Can get a partial file by using x_range and y_range.
    
    Parameters
    ----------
    filename : string
        Name of the fpd HDF5 file.
    x_range : tuple, optional
        Instead of returning the full dataset, only parts
        of the x probe positions will be returned. Useful for very large
        datasets. Default is None, which will return all the x probe
        positions. This _might_ greatly increase the loading time.
    y_range : tuple, optional
        Instead of returning the full dataset, only parts
        of the y probe positions will be returned. Useful for very large
        datasets. Default is None, which will return all the y probe
        positions. This _might_ greatly increase the loading time.

    Returns
    -------
    4-D HyperSpy image signal

    """
    fpdfile = h5py.File(filename,'r') #find data file in a read only format
    data_reference = fpdfile['fpd_expt']['fpd_data']['data']
    # Slightly convoluted way of loading parts of a dataset, since it takes
    # a very long time to do slicing directly on a HDF5-file
    if (x_range == None) and (y_range == None):
        data = data_reference[:][:,:,0,:,:]
    elif (not (x_range == None)) and (not(y_range==None)):
        data = data_reference[y_range[0]:y_range[1],x_range[0]:x_range[1],:,:,:][:,:,0,:,:]
    elif x_range == None:
        data = data_reference[y_range[0]:y_range[1],:,:,:,:][:,:,0,:,:]
    elif y_range == None:
        data = data_reference[:,x_range[0]:x_range[1],:,:,:][:,:,0,:,:]
    else:
        print("Something went wrong in the data loading")
    im = hs.signals.Image(data)
    _set_metadata_from_hdf5(fpdfile, im)
    _set_hardcoded_metadata(im)
    fpdfile.close()
    remove_dead_pixels(im)
    return(im)

def remove_dead_pixels(s_fpd):
    """Removes dead pixels from a fpd dataset in-place. 
    Replaces them with an average of the four nereast neighbors.
    
    Parameters
    ----------
    s_fpd : HyperSpy signal
    """
    s_sum = s_fpd.sum(0).sum(0)
    dead_pixels_x, dead_pixels_y = np.where(s_sum.data==0)

    for x, y in zip(dead_pixels_x, dead_pixels_y):
        n_pixel0 = s_fpd.data[:,:,x+1,y]
        n_pixel1 = s_fpd.data[:,:,x-1,y]
        n_pixel2 = s_fpd.data[:,:,x,y+1]
        n_pixel3 = s_fpd.data[:,:,x,y-1]
        s_fpd.data[:,:,x,y] = (n_pixel0+n_pixel1+n_pixel2+n_pixel3)/4


def _get_disk_centre_from_signal(signal, threshold=1.):
    """Get the centre of the disk using thresholded center of mass.
    Threshold is set to the mean of individual diffration images.

    Parameters
    ----------
    signal : HyperSpy 4-D fpd signal
    threshold : number, optional
        The thresholding will be done at mean times 
        this threshold value.

    Returns
    -------
    tuple with center x and y arrays. (com x, com y)"""
        

    mean_diff_array = signal.data.mean(axis=(2,3), dtype=np.float32)*threshold

    com_x_array = np.zeros(signal.data.shape[0:2], dtype=np.float64)
    com_y_array = np.zeros(signal.data.shape[0:2], dtype=np.float64)
    
    image_data = np.zeros(signal.data.shape[2:], dtype=np.uint16)

    # Slow, but memory efficient way
    for x in range(signal.axes_manager[0].size):
        for y in range(signal.axes_manager[1].size):
            image_data[:] = signal.data[x,y,:,:]
            image_data[image_data<mean_diff_array[x,y]] = 0
            image_data[image_data>mean_diff_array[x,y]] = 1
            com_y, com_x = center_of_mass(image_data)
            com_x_array[x,y] = com_x
            com_y_array[x,y] = com_y
    return(com_x_array, com_y_array)

def _get_radial_profile_of_diff_image(diff_image, centre_x, centre_y):
    """Radially integrates a single diffraction image.

    Parameters
    ----------
    diff_image : 2-D numpy array
        Array consisting of a single diffraction image.
    centre_x : number
        Centre x position of the diffraction image.
    centre_y : number
        Centre y position of the diffraction image.

    Returns
    -------
    1-D numpy array of the radial profile."""
#   Radially profiles the data, integrating the intensity in rings out from the centre. 
#   Unreliable as we approach the edges of the image as it just profiles the corners.
#   Less pixels there so become effectively zero after a certain point
    y, x = np.indices((diff_image.shape))
    r = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
    r = r.astype(int)       
    tbin =  np.bincount(r.ravel(), diff_image.ravel())
    nr = np.bincount(r.ravel())   
    radialProfile = tbin / nr

    return(radialProfile)

def _get_lowest_index_radial_array(radial_array):
    """Returns the lowest index of in a radial array.

    Parameters
    ----------
    radial_array : 3-D numpy array
        The last dimension must be the reciprocal dimension.

    Returns
    -------
    Number, lowest index."""

    lowest_index = radial_array.shape[-1]
    for x in range(radial_array.shape[0]):
        for y in range(radial_array.shape[1]):
            radial_data = radial_array[x,y,:]
            lowest_index_in_image = np.where(radial_data==0)[0][0]
            if lowest_index_in_image < lowest_index:
                lowest_index = lowest_index_in_image
    return(lowest_index)

def get_radial_profile_signal(signal):
    """Radially integrates a 4-D pixelated STEM diffraction signal.

    Parameters
    ----------
    signal : 4-D HyperSpy signal
        First two axes: 2 spatial dimensions.
        Last two axes: 2 reciprocal dimensions.

    Returns
    -------
    3-D HyperSpy signal, 2 spatial dimensions,
    1 integrated reciprocal dimension."""

    com_x_array, com_y_array = _get_disk_centre_from_signal(
            signal, threshold=1.)
    
    radial_profile_array = np.zeros(signal.data.shape[0:-1], dtype=np.float64)
    diff_image = np.zeros(signal.data.shape[2:], dtype=np.uint16)
    for x in range(signal.axes_manager[0].size):
        for y in range(signal.axes_manager[1].size):
            diff_image[:] = signal.data[x,y,:,:]
            centre_x = com_x_array[x,y]
            centre_y = com_y_array[x,y]
            radial_profile = _get_radial_profile_of_diff_image(
                    diff_image, centre_x, centre_y)
            radial_profile_array[x,y,0:len(radial_profile)] = radial_profile
    lowest_radial_zero_index = _get_lowest_index_radial_array(
            radial_profile_array)
    signal_radial = hs.signals.Spectrum(
            radial_profile_array[:,:,0:lowest_radial_zero_index])
    # TODO: ADD METADATA AND SCALING
    return(signal_radial)

"""
def radial_profile_dataset(im, centre, size, data_file, bulk_sto, flip = True):
#   Creates a hfd5 of the summed radial profiles for an input
#   hfd5 file
#   im is the dataset to be profiled
#   centre is the centre of each image in the dataset
#   save is the name that the profile will be saved under
#   flip is whether there is a single centre value or an array of centre values
#   rotation is whether the dataset is rotated or not

    
    rad_size = [] 
#   the length of rad varies and a numpy array needs a 
#   fixed row length for each entry

    for i in range(0,size[0]):
        for j in range (0,size[1]):
#   Iterates over each individual 256x256 image

            if flip == True:          
                rad = radial_profile(im[i,j].data, centre[i,j])
            else: rad = radial_profile(im[i,j].data, centre)
            rad_length = len(rad)
            rad_size.append(rad_length)

    min_rad_size = min(rad_size)
    np.save(data_file.replace(".hdf5","") + "_minimum_radial_length", min_rad_size)
    rad_size = np.reshape(rad_size, [size[0],size[1]])
    print rad_size
    centre_index = np.where(rad_size == min_rad_size)
    print centre_index
    min_rad_centre = centre[centre_index[0][0],centre_index[0][1]]
    print min_rad_centre
    del rad_size
    blankFileToSave = np.zeros((size[0],size[1],min_rad_size))
    for i in range(0,size[0]):
        for j in range(0,size[1]):
            
            if flip == True:
                rad = radial_profile(im[i,j].data, centre[i,j])
            else: rad = radial_profile(im[i,j].data, centre)
            
            if len(rad) > min_rad_size:
                while len(rad) > min_rad_size:
                    rad = np.delete(rad, [len(rad)-1])

#           Shortens the array of rad to create a uniform 
#           Length          
          
            blankFileToSave[i,j] = rad

#           Passes the numpy array for each image alone with the
#           Average centre of mass calculated earlier to the 
#           Radial profiler in CoM and saves the profile data
#           In a multidimensional numpy array

    mrad_per_step = _radial_calibration_value(bulk_sto, data_file, min_rad_size, min_rad_centre)
    
    s = hs.signals.Image(blankFileToSave)
    s.axes_manager[1].scale = mrad_per_step[()]
    s.axes_manager[1].units = "mrad"
    s.axes_manager[1].name  = "Scattering Angle"
    s.save(data_file.replace(".hdf5","_dataset_profile"))        
    np.save(data_file.replace(".hdf5","_dataset_profile"), blankFileToSave)
#   Changes the numpy array with the profiles in it into a hyperspy
#   signal and saves it based on the chosen save name
    return blankFileToSave 
#
    
def _dataset_dimensions(dataset):
#   The datasets are different sizes so we need to figure out how many images are in the set
#   so that we can iterate over all of them at a later data. We sum over the signal dimensions
#   then return the shape of the navigation dimensions
    summed_signal = dataset.sum(-1).sum(-1).data
    dataset_shape = np.shape(summed_signal)
    return dataset_shape   
   
'''#Centre of Mass calculation with thresholding from Magnus
def centre_of_disk_centre_of_mass(
        image,
        threshold=None):
    if threshold == None:
#            threshold = (image.max()-image.min())*0.5
        threshold = np.mean(image) * (45/2)
#       The mean of the data set being used to write this is roughly
#       1/45 the max value. This threshold is too low and means that
#       Lower intensity features are interfering with the creation
#       Of the boolean disk
        
    image[image<threshold] = 0
    image[image>threshold] = 1
    booleanArray = image.astype(bool)
    disk_centre = sp.ndimage.measurements.center_of_mass(booleanArray)

    return(disk_centre)'''

def centre_of_disk_centre_of_mass_full(
        image, data_file, size):
#   Thresholds the data to remove as much of the secondary features then find the centre
#   of mass of the central beam. This is then added to a list and saved to be indexed later        
    disk_centre=[]
    for i in range(0,size[0]):
        for j in range(0,size[1]):  
            imagedata = image[i,j].data.astype('float32')
            threshold = np.mean(imagedata) * 3
            imagedata[imagedata < threshold] = 0
            imagedata[imagedata > threshold] = 1
            centre = sp.ndimage.measurements.center_of_mass(imagedata)

            centre2 = [centre[0], centre[1]]
            disk_centre.append(centre2)
    disk_centre_array = np.reshape(disk_centre, (size[0],size[1],2))
    np.save(data_file.replace(".hdf5", "_centre"), disk_centre_array)
    return(disk_centre_array)
          

#centre profile calculation from StackOverflow
def radial_profile(data,centre):
#   Radially profiles the data, integrating the intensity in rings out from the centre. 
#   Unreliable as we approach the edges of the image as it just profiles the corners.
#   Less pixels there so become effectively zero after a certain point
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    r = r.astype(int)       
    tbin =  np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())   
    radialProfile = tbin / nr

    return radialProfile  
    	   
def _radial_calibration_value(sto_summed, data_file, min_rad, centre):
#   Finds the calibration value for a certain camera length
    camera_length = _camera_length(data_file)
    calibration_file = _file_search("calibration_%d" %camera_length + ".npy")
    if calibration_file == False:
        sto_image = _file_search("bulk_sto_calibration%d" %camera_length +".tif")
        if sto_image == False:
            sto_summed.change_dtype('float32')
            sto_summed.save("bulk_sto_calibration%d" %camera_length +".tif")
        print "Please use bulk_sto_calibration%d"%camera_length + ".tif in ImageJ"
        print "Find the diameter of the STO Laue zone and enter it in pixels"
        sto_laue_pixels = input("Laue zone diameter: ")

##########
        x_offset = 128-abs(128-centre[0])
        y_offset = 128-abs(128-centre[1])
        if x_offset < y_offset:
            number_of_pixels = float(x_offset*2.)
        else:
            number_of_pixels = float(y_offset*2.)            
##########
        electron_wavelength = _electron_wavelength(data_file)
         
        sto_a = 3.905 #Angstrom
        sto_110 = 2.*math.pi*math.sqrt(2)/sto_a #Lattice size along 110
        sto_folz_angle = 2.*np.arcsin(math.sqrt(sto_110 * electron_wavelength/(4*math.pi))) #in radians
        sto_folz_mrad  = sto_folz_angle * 1000
        
        mrad_per_pixel = (2. * sto_folz_mrad) / sto_laue_pixels
        pixels_per_step = number_of_pixels/(2.*min_rad)
        mrad_per_step_scalar = mrad_per_pixel * pixels_per_step
        np.save("calibration_%d" %camera_length, mrad_per_step_scalar)

    mrad_per_step = np.load("calibration_%d" %camera_length + ".npy")
        
    return mrad_per_step
    
def _electron_wavelength(data_file):
    microscope_energy = h5py.File(data_file)['fpd_expt']['DM0']['tags']['ImageList']['TagGroup0']['ImageTags']['Microscope Info']['Voltage'].value

#   relativistic de broglie equation    
    electron_wavelength = (12.25 * 10 ** (-10))/math.sqrt(microscope_energy) * 1/(math.sqrt(1+(microscope_energy*1.6*10 ** (-19))/(2*9.11*10**(-31) * (3*10**8)**2))) #in meters
    electron_wavelength_Angstrom = electron_wavelength*10**(10)
    np.save("electron_wavelength_%d" %microscope_energy, electron_wavelength_Angstrom) 
    return electron_wavelength_Angstrom 
                  
def diffraction_calibration(centre, size, im = None, scale = None):
#	This function calibrates the diffraction patterns
#   It takes the raw data input & crops the data set to the last 14 rows
#	This assumes that bulk STO is all that exists in these 14 rows. It then sums
#	Over the 0th and 1st axes into a single	.tif image that we can put into ImageJ
#   To get a "scale". It then applies the calibration to the data and returns it
#   Unchanged other than the addition of the calibration

#   Fairly sure this is completely unnecessary
	
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
    
	if type(centre) is tuple:
	    a2 = im.axes_manager[2]
	    a2.offset = -centre[0]
	    a2.scale = scale
	    a2.units = "mrad"
	    
	    a3 = im.axes_manager[3]
	    a3.offset = -centre[1]
	    a3.scale = scale
	    a3.units = "mrad"
	    
	else:
	    for i in range(0,size[0]):
	        for j in range (0,size[1]):
	            a2 = im[i,j].axes_manager[0]
	            a2.offset = - centre[i,j][0]
	            a2.scale = scale
	            a2.units = "mrad"
	            
	            a3 = im[i,j].axes_manager[1]
	            a3.offset = centre[i,j][1]
	            a3.scale = scale
	            a3.units = "mrad"

	return im 
   
def radial_profile_dataset(im, centre, size, data_file, bulk_sto, flip = True):
#   Creates a hfd5 of the summed radial profiles for an input
#   hfd5 file
#   im is the dataset to be profiled
#   centre is the centre of each image in the dataset
#   save is the name that the profile will be saved under
#   flip is whether there is a single centre value or an array of centre values
#   rotation is whether the dataset is rotated or not

    
    rad_size = [] 
#   the length of rad varies and a numpy array needs a 
#   fixed row length for each entry

    for i in range(0,size[0]):
        for j in range (0,size[1]):
#   Iterates over each individual 256x256 image

            if flip == True:          
                rad = radial_profile(im[i,j].data, centre[i,j])
            else: rad = radial_profile(im[i,j].data, centre)
            rad_length = len(rad)
            rad_size.append(rad_length)

    min_rad_size = min(rad_size)
    np.save(data_file.replace(".hdf5","") + "_minimum_radial_length", min_rad_size)
    rad_size = np.reshape(rad_size, [size[0],size[1]])
    print rad_size
    centre_index = np.where(rad_size == min_rad_size)
    print centre_index
    min_rad_centre = centre[centre_index[0][0],centre_index[0][1]]
    print min_rad_centre
    del rad_size
    blankFileToSave = np.zeros((size[0],size[1],min_rad_size))
    for i in range(0,size[0]):
        for j in range(0,size[1]):
            
            if flip == True:
                rad = radial_profile(im[i,j].data, centre[i,j])
            else: rad = radial_profile(im[i,j].data, centre)
            
            if len(rad) > min_rad_size:
                while len(rad) > min_rad_size:
                    rad = np.delete(rad, [len(rad)-1])

#           Shortens the array of rad to create a uniform 
#           Length          
          
            blankFileToSave[i,j] = rad

#           Passes the numpy array for each image alone with the
#           Average centre of mass calculated earlier to the 
#           Radial profiler in CoM and saves the profile data
#           In a multidimensional numpy array

    mrad_per_step = _radial_calibration_value(bulk_sto, data_file, min_rad_size, min_rad_centre)
    
    s = hs.signals.Image(blankFileToSave)
    s.axes_manager[1].scale = mrad_per_step[()]
    s.axes_manager[1].units = "mrad"
    s.axes_manager[1].name  = "Scattering Angle"
    s.save(data_file.replace(".hdf5","_dataset_profile"))        
    np.save(data_file.replace(".hdf5","_dataset_profile"), blankFileToSave)
#   Changes the numpy array with the profiles in it into a hyperspy
#   signal and saves it based on the chosen save name
    return blankFileToSave
    	
# These five functions are just functions that can be fitted to the mean values if so wished. 
# Produces gaussian fits and polynomials	       
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def square(list): return[i**2 for i in list]
def cube(list): return[i**3 for i in list]
def fourth(list): return[i**4 for i in list]
def multiple(poly, list): return[poly*i for i in list]

def _powerlaw_fitter(data_file, size, bounded_gaussian = True, threshold = True, save_file = True):
#   This function does a lot and is the main part of the program. 
    powerlaw_fit_exist = _file_search(data_file.replace(".hdf5","") + "_powerlaw.hdf5")
    camera_length = _camera_length(data_file)
    mrad_per_step = np.load("calibration_%d" %camera_length + ".npy")
    
    im = hs.load(data_file.replace(".hdf5","") + "_dataset_profile.hdf5").as_image((0,2)) #load data
       
    im2 = im.to_spectrum() #change signal type
        
    central_beam_image = im2[round(0.9*size[0]),round(0.9*size[1])]
    central_beam_plot = central_beam_image.plot()
    plt.show()
    central_beam = input("Please look at the generated image and enter a rough end for the central beam: ")
#    central_beam = 15.
    end_spectrum = input("Enter a value after which there is no signal: ")
    central_beam2 = int(central_beam/(mrad_per_step[()]))
    end_spectrum2 = int(end_spectrum/(mrad_per_step[()]))

    wait = 0
    while wait == False:
        wait = str(input("Enter 1 when all images closed: "))
    
    im3 = im2.isig[central_beam2:end_spectrum2] #crop out centre beam
     
    im3.sum(-1).plot()
    plt.show()
    x_y_platinum = raw_input("Please state which axis (x or y) the platinum lies across (x = l/r, y = t/b): ")    
    if x_y_platinum == 'x':
        platinum_range = input("Please look at the image and enter the column (x value) where the platinum roughly ends: ")
        im4 = im3.inav[:platinum_range,:]
        wait = 0
        while wait == False:
            wait = input("Enter 1 when all images closed: ")    
    else:     
        platinum_range = input("Please look at the image and enter the row (y value) where the platinum roughly ends: ")
        wait = 0
        while wait == False:
            wait = input("Enter 1 when all images closed: ")
    
        im4 = im3.inav[:,platinum_range:] #remove platinum
    
    m = im4.create_model() #turns spectrum into a model
    
    im5 = im4.sum(0)
    im5.plot()
    plt.show()
################################
    
    start_powerlaw = central_beam
    end_powerlaw = float(input("Please look at the model and enter the end signal range for the power law background by navigating to an image with a peak: "))

    wait = 0
    while wait == False:
        wait = str(input("Enter 1 when all images closed: "))
    
    if powerlaw_fit_exist == False:
        powerlaw = hs.model.components.PowerLaw() #creates a powerlaw
        m.append(powerlaw)
        m.set_signal_range(start_powerlaw,end_powerlaw)
        m.multifit() #fits powerlaw to all images 
        m.reset_signal_range() #reset signal range
        powerlaw.set_parameters_not_free() #fixes powerlaw shape

        powerlaw_fit = im4 - m.as_signal() #Removes the powerlaw as background
        powerlaw_fit.save(data_file.replace(".hdf5","_powerlaw"))
    
    else:
        powerlaw_fit = hs.load(data_file.replace(".hdf5","") + "_powerlaw.hdf5")

#def _gaussian_fitter(powerlaw_fit, bounded_gaussian=True):
    
    gaussian_centre_exist = _file_search(data_file.replace(".hdf5","") + "_centre_signal.hdf5")
    gaussian_sigma_exist  = _file_search(data_file.replace(".hdf5","") + "_sigma_signal.hdf5")
    gaussian_A_exist      = _file_search(data_file.replace(".hdf5","") + "_A_signal.hdf5")
    
    gaussian = hs.model.components.Gaussian() #creates a gaussian component    
# Finds the probe position with the highest total intensity which 
# Will be used to set the initial conditions for the Gaussian fit
        
    powerlaw_fit_zeroed = copy.deepcopy(powerlaw_fit)
    powerlaw_fit_zeroed.data[powerlaw_fit_zeroed.data < 0] = 0.
    peak_intensities = powerlaw_fit_zeroed.sum(-1)
    maxSumIndex = unravel_index(peak_intensities.data.argmax(),peak_intensities.data.shape)
    maxSignalPosition = powerlaw_fit_zeroed[maxSumIndex[1],maxSumIndex[0]]

    max_gauss_model = maxSignalPosition.create_model()
    max_gauss_model.append(gaussian)
    max_gauss_model.plot()
    plt.show()
    
    start_gaussian = float(input("Please look at the model and enter the start signal range for the Gaussian signal fit (don't close image yet): "))

    end_gaussian = float(input("Please look at the model and enter the end signal range for the Gaussian signal fit: "))
   
    if gaussian_centre_exist == False or gaussian_sigma_exist  == False or gaussian_A_exist == False:
        max_gauss_model.set_signal_range(start_gaussian, end_gaussian)
        gaussian.centre.value = (start_gaussian + end_gaussian)/2.
        max_gauss_model.fit(fitter = "mpfit")
    
        print gaussian.centre.value
        print gaussian.sigma.value
        print gaussian.A.value
        current_centre = gaussian.centre.value
        current_A = gaussian.A.value
        current_sigma = gaussian.sigma.value
        del max_gauss_model
        del gaussian
        

################################
# Fit a Gaussian to the whole data set using the values from the 
# Strongest signal as initial conditions and apply bounds on the 
# Extremes
        
        gaussian = hs.model.components.Gaussian()
        gauss_model = powerlaw_fit.create_model() 
        gauss_model.append(gaussian) 
        gauss_model.set_signal_range(start_gaussian, end_gaussian)
    
        #centreMin = int(start_gaussian/mrad_per_step)
        #centreMax = int(end_gaussian/mrad_per_step)
        centreMin = int(start_gaussian)
        centreMax = int(end_gaussian)
        gaussian.centre.bmin = centreMin
        gaussian.centre.bmax = centreMax
        gaussian.centre.value = current_centre
        gaussian.centre.assign_current_value_to_all()
    
        gaussian.A.bmin = 0.
        gaussian.A.value = current_A
        gaussian.A.assign_current_value_to_all()
    
        gaussian.sigma.bmin = 0.1
        gaussian.sigma.bmax = 15.
        gaussian.sigma.value = current_sigma
        gaussian.sigma.assign_current_value_to_all()
    
        wait = 0
        while wait == False:
            wait = str(input("Enter 1 after closing all images: "))
    
        if bounded_gaussian == True:
            gauss_model.multifit(fitter="mpfit", bounded = True)
        else: gauss_model.multifit(fitter="mpfit", bounded = False) 
        gauss_model.reset_signal_range()
       
########################################################
    
        centreGaussian = gaussian.centre.as_signal() #create a signal of all centre values
        aGaussian = gaussian.A.as_signal() # create a signal of all A values
        sigmaGaussian = gaussian.sigma.as_signal() #create a signal of all sigma values
          
        if threshold == True:
            sigma_data = copy.deepcopy(sigmaGaussian.data)
            centre_data = copy.deepcopy(centreGaussian.data)
            a_data = copy.deepcopy(aGaussian.data)
            centre_data[sigma_data >= gaussian.sigma.bmax] = 0.
            centre_data[centre_data >= centreMax] = 0.
            centre_data[centre_data <= centreMin] = 0.
            centre_data[a_data <= gaussian.A.bmin] = 0.
                
            centreGaussian.data[centre_data == 0.] = 0. #apply the thresholded data map to the original signals
            aGaussian.data[centre_data == 0.] = 0. 
            sigmaGaussian.data[centre_data == 0.] = 0.
            del centre_data
            del sigma_data
            del a_data
        
        if save_file == True:
            centreGaussian.save(data_file.replace(".hdf5","") + "_centre_signal.hdf5")        
            aGaussian.save(data_file.replace(".hdf5","") + "_A_signal.hdf5")
            sigmaGaussian.save(data_file.replace(".hdf5","") + "_sigma_signal.hdf5")
    
    else:
        centreGaussian  = hs.load(data_file.replace(".hdf5","") + "_centre_signal.hdf5").as_image((1,0))   
        aGaussian       = hs.load(data_file.replace(".hdf5","") + "_A_signal.hdf5").as_image((1,0))
        sigmaGaussian   = hs.load(data_file.replace(".hdf5","") + "_sigma_signal.hdf5").as_image((1,0))
    
    dataset_radial_profile = np.load(data_file.replace(".hdf5","_dataset_profile.npy"))
    adf_ends = [end_powerlaw, end_gaussian]      
    step_array = _adf_step(dataset_radial_profile, adf_ends)
    step_array[:] = [x/mrad_per_step for x in step_array]
    print step_array
    
    return [aGaussian, centreGaussian, sigmaGaussian], peak_intensities, platinum_range, step_array, adf_ends

##############################
    
def _adf_step(im, step_ends):
#   Just creates an array of equal steps through the HOLZ and into the adf    

    adf_step = step_ends[1]
    start_step = step_ends[0]
    step_size = (adf_step - start_step) / 5
    step_array = [start_step]
    for i in range(1,6):
        next_step = start_step + (i*step_size)
        step_array.append(next_step)
        print step_array

    step_array_sorted = np.sort(step_array)
    np.append(step_array_sorted,adf_step)
    return step_array_sorted 
    
    
    
def _clim_set(image_array):
#   Changes the range over which the Gaussian signals are viewed, asking the user to enter ranges
#   until the user is happy
    a_gauss     = image_array[0]
    centre_gauss = image_array[1]
    sigma_gauss = image_array[2]
    
    fig, axarr = plt.subplots(1, 3, figsize = (15,5))
    
    ax_centre_gauss = axarr[0]
    c_centre_gauss = ax_centre_gauss.imshow(centre_gauss.data)
    ax_centre_gauss.set_title("Gaussian Centre")
    fig.colorbar(c_centre_gauss, ax = ax_centre_gauss)
    
    ax_sigma_gauss  = axarr[1]
    c_sigma_gauss = ax_sigma_gauss.imshow(sigma_gauss.data)
    ax_sigma_gauss.set_title("Gaussian Sigma")
    fig.colorbar(c_sigma_gauss, ax = ax_sigma_gauss)

    ax_a_gauss      = axarr[2]
    c_a_gauss = ax_a_gauss.imshow(a_gauss.data)
    ax_a_gauss.set_title("Gaussian A")
    fig.colorbar(c_a_gauss, ax = ax_a_gauss)
        
    fig.show()
    
    centre_clim = input("Please enter values to view centre over as min,max: ")
    print centre_clim
    sigma_clim  = input("Please enter values to view sigma over as min,max: ")
    a_clim      = input("Please enter a minimum value to view A over: ")
    
    wait = 0
    while wait == False:
        wait = str(input("Enter 1 after closing all images: "))
    
    end = 0
    while end == False:
        fig, axarr = plt.subplots(1, 3, figsize = (15,5))
       
        ax_centre_gauss = axarr[0]
        c_centre_gauss = ax_centre_gauss.imshow(centre_gauss.data)
        c_centre_gauss.set_clim(centre_clim)
        ax_centre_gauss.set_title("Gaussian Centre")
        fig.colorbar(c_centre_gauss, ax = ax_centre_gauss)
        
        ax_sigma_gauss  = axarr[1]
        c_sigma_gauss = ax_sigma_gauss.imshow(sigma_gauss.data)
        c_sigma_gauss.set_clim(sigma_clim)
        ax_sigma_gauss.set_title("Gaussian Sigma")
        fig.colorbar(c_sigma_gauss, ax = ax_sigma_gauss)
        
        ax_a_gauss      = axarr[2]
        c_a_gauss = ax_a_gauss.imshow(a_gauss.data)
        c_a_gauss.set_clim(a_clim)
        ax_a_gauss.set_title("Gaussian A")
        fig.colorbar(c_a_gauss, ax = ax_a_gauss)        
        fig.show()
            
        end = input("Is this clim set ok? 0/1: ")
        if end == False:
            centre_clim = input("Please enter values to view centre over as min,max: ")
            sigma_clim  = input("Please enter values to view sigma over as min,max: ")
            a_clim      = input("Please enter a minimum value to view A over: ")
        wait = 0
        while wait == False:
            wait = str(input("Enter 1 after closing all images: "))
            
    return [centre_clim, sigma_clim, a_clim]

def _annular_dark_field_image(radial_dataset, size, start_point, end_point = None):
#   Returns a numpy array of the ADF that can be displayed as an image.
#   Input is the radially profiled dataset, not a HDF5 file.
    results = []
    for i in range (0,size[0]):
        for j in range (0,size[1]):
            radial_profile = radial_dataset[i,j]
            if end_point == None:
                 end_point = len(radial_profile)
            sum_range = sp.integrate.simps(radial_profile[start_point:end_point])
            results.append(sum_range)
    adf_array = np.array(results)
    adf_array = adf_array.reshape(size[0],size[1])
    del results

    return adf_array

def _gaussian_parameter_mean(gaussian_signals, gaussian_range):
    a_gaussian              = gaussian_signals[0].data
    len_a_gaussian = len(a_gaussian[0])
 
###################################################################    
#   This bit just looks for rows where only 25% of the row is 
#   nonzero and then just sets those values to zero as well    
    for j in range(0, len(a_gaussian)):
        x = [i for i in range(0, len_a_gaussian) if a_gaussian[j][i] != 0]
        if len(x) < 0.25 * len_a_gaussian:
            a_gaussian[j] = 0.
###################################################################           
    centre_gaussian         = gaussian_signals[1].data   
    sigma_gaussian          = gaussian_signals[2].data     

    '''    a_gaussian_masked       = np.ma.masked_where(a_gaussian == 0, a_gaussian)

    centre_gaussian_masked0  = np.ma.masked_where(a_gaussian == 0, centre_gaussian)
    centre_gaussian_masked1 = np.ma.masked_where(centre_gaussian < gaussian_range[0], centre_gaussian_masked0)
    centre_gaussian_masked2 = np.ma.masked_where(centre_gaussian > gaussian_range[1], centre_gaussian_masked1)
    

    sigma_gaussian_masked0   = np.ma.masked_where(a_gaussian == 0, sigma_gaussian)  
    sigma_gaussian_masked1 = np.ma.masked_where(centre_gaussian < gaussian_range[0], sigma_gaussian_masked0)
    sigma_gaussian_masked2 = np.ma.masked_where(centre_gaussian > gaussian_range[1], sigma_gaussian_masked1)
    '''
    
    a_mean                 = a_gaussian.mean(axis = 1)
    centre_mean            = centre_gaussian.mean(axis = 1)
    sigma_mean             = sigma_gaussian.mean(axis = 1)
    
###################################################################
#   Looks for mean values that don't exceed the imposed limits    
   
    a_mean_masked0          = np.ma.masked_where(centre_mean < gaussian_range[0], a_mean)
    a_mean_masked1          = np.ma.masked_where(centre_mean > gaussian_range[1], a_mean_masked0)
    centre_mean_masked0          = np.ma.masked_where(centre_mean < gaussian_range[0], centre_mean)
    centre_mean_masked1          = np.ma.masked_where(centre_mean > gaussian_range[1], centre_mean_masked0)
    sigma_mean_masked0          = np.ma.masked_where(centre_mean < gaussian_range[0], sigma_mean)
    sigma_mean_masked1          = np.ma.masked_where(centre_mean > gaussian_range[1], sigma_mean_masked0)
    
    return [a_mean], [centre_mean], [sigma_mean], [centre_mean_masked1], [sigma_mean_masked1], [a_mean_masked1] 
    # Fitting the data with Gaussians and polynomials, commented out for now.
    '''
#pInitial = [114., 15., 8.]
    pInitial = [50.,15.,5.]
    lengthOfFitGauss = list(range(len(a_mean)))
    coeffGauss, varmatrix = curve_fit(gauss, lengthOfFitGauss, a_mean, p0=pInitial)
    dataFitGauss = gauss(lengthOfFitGauss, *coeffGauss)

#plt.figure(0)
#A_mean_image = plt.plot(lengthOfFitGauss, a_mean, 'ro', label='Data')
#plt.plot(lengthOfFitGauss, dataFitGauss, 'bo', label='Fit')
#plt.show()
#plt.legend()

#print 'Most intense Laue zone at row ', coeffGauss[1]
#print 'Sigma = ', coeffGauss[2]


#############################
#Fit a polynomial to the thresholded centre_mean then translate that back to centre_mean

    lengthOfFitPoly = list(range(len(centre_mean_thresholded)))
    poly = np.polynomial.polynomial.polyfit(lengthOfFitPoly, centre_mean_thresholded, 4, rcond=None, full=False, w=None)

    polyFit = poly[0] +multiple(poly[1], lengthOfFitPoly) + multiple(poly[2],square(lengthOfFitPoly)) + multiple(poly[3], cube(lengthOfFitPoly)) + multiple(poly[4], fourth(lengthOfFitPoly))

#plt.figure(1)
#centre_mean_image = plt.plot(lengthOfFitPoly,centre_mean_thresholded,'ro', label = 'Data')
#plt.plot(lengthOfFitPoly, polyFit, 'bo', label = 'Fit')
#plt.show()
#plt.legend()



# Changes polyFit into a comparable dataset to centre_mean

    preLaueIndex = [n for n, i in enumerate(centre_mean) if i >= centreMin][0]
    postLaueIndex = preLaueIndex + len(centre_mean_thresholded)
    centre_meanPoly = [0.] * (preLaueIndex)
    listOfZeros = [0.] * (len(centre_mean) - postLaueIndex)
    centre_meanPoly.extend(polyFit)
    centre_meanPoly.extend(listOfZeros)
    '''
#plt.plot(lengthOfFitGauss, centre_meanPoly, 'bo', label = 'Data')
#plt.plot(lengthOfFitGauss, centre_mean, 'ro', label = 'Fit')
#plt.show()
#plt.legend() 

def _dataset_rotation(image_array, angle_of_rotation):
    adf         = image_array[0]
    adf_step_1  = image_array[1]
    adf_step_2  = image_array[2]
    adf_step_3  = image_array[3]
    adf_step_4  = image_array[4]
    adf_step_5  = image_array[5]
    
    adf_rotated         = sp.ndimage.interpolation.rotate(adf, angle_of_rotation, reshape = False)
    adf_1_rotated       = sp.ndimage.interpolation.rotate(adf_step_1, angle_of_rotation, reshape = False)
    adf_2_rotated       = sp.ndimage.interpolation.rotate(adf_step_2, angle_of_rotation, reshape = False)
    adf_3_rotated       = sp.ndimage.interpolation.rotate(adf_step_3, angle_of_rotation, reshape = False)
    adf_4_rotated       = sp.ndimage.interpolation.rotate(adf_step_4, angle_of_rotation, reshape = False)
    adf_5_rotated       = sp.ndimage.interpolation.rotate(adf_step_5, angle_of_rotation, reshape = False)
        
    return adf_rotated, adf_1_rotated, adf_2_rotated, adf_3_rotated, adf_4_rotated, adf_5_rotated
  
def _signal_rotation(signal, angle_of_rotation):
    a_gauss = signal[0]
    centre_gauss = signal[1]
    sigma_gauss = signal[2]
    
    a_gauss.map(sp.ndimage.rotate, angle = angle_of_rotation, reshape = False)
    centre_gauss.map(sp.ndimage.rotate, angle = angle_of_rotation, reshape = False)
    sigma_gauss.map(sp.ndimage.rotate, angle = angle_of_rotation, reshape = False)
    
    return [a_gauss, centre_gauss, sigma_gauss]
    
def _figure_compare(image_array, peak_intensities, save_name, clim_set, bounded_gaussian = True):
#   Input an array of images to be displayed. Initally lets assume we are going to have 14 images.     
    
    adf                 = image_array[0][0]
    adf_step_1          = image_array[0][1]
    adf_step_2          = image_array[0][2]
    adf_step_3          = image_array[0][3]
    adf_step_4          = image_array[0][4]
    adf_step_5          = image_array[0][5]
    a_gauss             = image_array[1][0]
    centre_gauss        = image_array[1][1]
    sigma_gauss         = image_array[1][2]
    a_mean              = image_array[2][0][0]
    centre_mean         = image_array[2][1][0]
    sigma_mean          = image_array[2][2][0]
    peak_intensities    = peak_intensities
    centre_mean_masked  = image_array [2][3][0]
    sigma_mean_masked   = image_array [2][4][0]
    
     

    fig, axarr = plt.subplots(5,3, figsize = (20,20))
    ax_adf          = axarr[0][0]
    ax_adf.imshow(adf)
    ax_adf.set_title("ADF Image post HOLZ")
    ax_adf_step_1   = axarr[0][1]
    ax_adf_step_1.imshow(adf_step_1)
    ax_adf_step_1.set_title("ADF steps through HOLZ (1)")
    ax_adf_step_2   = axarr[0][2]
    ax_adf_step_2.imshow(adf_step_2)
    ax_adf_step_2.set_title("ADF steps through HOLZ (2)")
    ax_adf_step_3   = axarr[1][0]
    ax_adf_step_3.imshow(adf_step_3)
    ax_adf_step_3.set_title("ADF steps through HOLZ (3)")
    ax_adf_step_4   = axarr[1][1]
    ax_adf_step_4.imshow(adf_step_4)
    ax_adf_step_4.set_title("ADF steps through HOLZ (4)")
    ax_adf_step_5   = axarr[1][2]
    ax_adf_step_5.imshow(adf_step_5)
    ax_adf_step_5.set_title("ADF steps through HOLZ (5)")
    
    ax_a_gauss      = axarr[2][0]
    c_a_gauss = ax_a_gauss.imshow(a_gauss.data)
    c_a_gauss.set_clim(clim_set[2])
    ax_a_gauss.set_title("Gaussian A")
    fig.colorbar(c_a_gauss, ax = ax_a_gauss)
    
    ax_centre_gauss = axarr[2][1]
    c_centre_gauss = ax_centre_gauss.imshow(centre_gauss.data)
    c_centre_gauss.set_clim(clim_set[0][0], clim_set[0][1])
    ax_centre_gauss.set_title("Gaussian Centre")
    fig.colorbar(c_centre_gauss, ax = ax_centre_gauss)
    
    ax_sigma_gauss  = axarr[2][2]
    c_sigma_gauss = ax_sigma_gauss.imshow(sigma_gauss.data)
    c_sigma_gauss.set_clim(clim_set[1][0], clim_set[1][1])
    ax_sigma_gauss.set_title("Gaussian Sigma")
    fig.colorbar(c_sigma_gauss, ax = ax_sigma_gauss)
    
    ax_a_mean       = axarr[3][0]
    ax_a_mean.plot(a_mean, 'ro')
    ax_a_mean.set_title("Mean Gaussian A")
    
    ax_centre_mean  = axarr[3][1]
    ax_centre_mean.plot(centre_mean, 'ro')
    ax_centre_mean.set_title("Mean Gaussian Centre")
    
    ax_sigma_mean   = axarr[3][2]
    ax_sigma_mean.plot(sigma_mean, 'ro')
    ax_sigma_mean.set_title("Mean Gaussian Sigma")
    
    ax_peak_intensities = axarr[4][0]
    c_peak_intensities = ax_peak_intensities.imshow(peak_intensities.data)
    ax_peak_intensities.set_title("Summed intensities")
    
    ax_centre_masked = axarr[4][1]
    ax_centre_masked.plot(centre_mean_masked, 'ro')
    ax_centre_masked.set_title("Mean Centre Values Across LFO Film")
    
    ax_sigma_masked = axarr[4][2]
    ax_sigma_masked.plot(sigma_mean_masked, 'ro')
    ax_sigma_masked.set_title("Mean Sigma Values Across LFO Film")

    fig.tight_layout()
    if bounded_gaussian == True:
        fig.savefig(save_name + "_bounded_gaussian" + ".jpg")
    else: 
        fig.savefig(save_name + "_unbounded_gaussian" + ".jpg")

def _flip_transpose (image_array):
#   Flips and transposes the image. For some reason, the profiler puts the platinum on the left hand side instead of the top
    image_flipped_transposed = np.transpose(image_array)
    return image_flipped_transposed        

def _dark_field_array(radial_profiled_dataset, step_array, size, platinum, data_file):
#   Integrates the profiled dataset in different steps throught the ADF and HOLZ
    #radial_profiled_dataset = radial_profiled_dataset[13:,:]
    adf_array  = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[5], end_point = None))[platinum:,:]
    adf_step_1 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[0], step_array[1]))[platinum:,:]
    adf_step_2 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[1], step_array[2]))[platinum:,:]
    adf_step_3 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[2], step_array[3]))[platinum:,:]
    adf_step_4 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[3], step_array[4]))[platinum:,:]
    adf_step_5 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[4], step_array[5]))[platinum:,:]
    adf = [adf_array, adf_step_1, adf_step_2, adf_step_3, adf_step_4, adf_step_5]
    np.save(data_file.replace(".hdf5","_adf_array"),adf)
    return adf
    
def _lattice_size(data_file, angle):
    microscope_energy = h5py.File(data_file)['fpd_expt']['DM0']['tags']['ImageList']['TagGroup0']['ImageTags']['Microscope Info']['Voltage'].value
    electron_wavelength = np.load("electron_wavelength_%d" %microscope_energy +".npy")
    # divide angle by 1000 to convert to radians
    reciprocal_lattice_size = math.sin(angle/(1000*2))**2 * (4*math.pi)/electron_wavelength[()]
    lattice_size = 2*math.pi/reciprocal_lattice_size
    return lattice_size 
    
def _lattice_error(angle,sigma,data_file):
    electron_wavelength = _electron_wavelength(data_file)
    lattice_error = (angle*10**-3)/electron_wavelength * (sigma*10**-3)
    return lattice_error
    
def _lattice_real_space(data_file, mean_gaussian_centres, mean_gaussian_sigma):
    lattice_size_list  = []
    lattice_error_list = []
    for i in range(0,len(mean_gaussian_centres)):
        lattice_size    = _lattice_size(data_file,mean_gaussian_centres[i])
        lattice_error   = _lattice_error(mean_gaussian_centres[i], mean_gaussian_sigma[i], data_file)
        lattice_size_list.append(lattice_size)
        lattice_error_list.append(lattice_error)
    with h5py.File(data_file.replace(".hdf5","") + '_real_space', 'w') as hf:
        g1 = hf.create_group('real_space_lattice')
        g1.create_dataset('real_space_lattice', data = lattice_size_list)
        g1.create_dataset('real_space_lattice_errors', data = lattice_error_list)
    return        
        
def _real_space_plot(data_file):
    lattice_variables = h5py.File(data_file.replace(".hdf5", "_real_space"), 'r')
    lattice_size = np.array(lattice_variables['real_space_lattice']['real_space_lattice'])
    print lattice_size
    lattice_error = np.array(lattice_variables['real_space_lattice']['real_space_lattice_errors'])
    x = range(len(lattice_size))
    y = lattice_size
    yerr = lattice_error
    plt.figure()
    plt.errorbar(x,y,xerr = 0,yerr=yerr,fmt='o')
    plt.show()   
    return
                    
       
def _sigma_centre_plot(sigma, centre, intensity,save):        
    #host = host_subplot(111, axes_class=AA.Axes)
    #par1 = host.twinx()
    #par2 = host.twinx()
    #p1, = host.plot(sigma,'b:', label = "Peak Width")
    #p2, = par1.plot(centre,'r--', label = "Electron Scattering Angle")
    #p3, = par2.plot(intensity, 'g.', label = "Electron Counts")
    #host.set_xlabel("Step as a Function of Distance from LSMO/LFO interface")
    #host.set_ylabel("Sigma")
    #par1.set_ylabel("Centre (mrad)")
    #par2.axes.get_yaxis().set_ticklabels([])
    #par2.axes.get_yaxis().set_ticks([])
    #host.legend(loc = 4)
    plt.figure(figsize=(6,6))
    host = host_subplot(111,axes_class= AA.Axes)
    par1 = host.twinx()
    p1, = host.plot(centre, 'b:', label = "Electron Scattering Angle")
    p2, = par1.plot(intensity, 'g--', label = "Electron Count")
    host.set_xlabel("Step as a Function of Distance from the LSMO/LFO interface")
    host.set_ylabel("Electron Scattering Angle (mrad)")
    par1.axes.get_yaxis().set_ticklabels([])
    par1.axes.get_yaxis().set_ticks([])
    host.legend(loc=4)
    plt.savefig(save)
    plt.draw()
    plt.show()"""
