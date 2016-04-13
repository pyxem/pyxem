import CoM
import hyperspy.api as hs
import numpy as np
import scipy as sp
import h5py
import copy
import CoM
import matplotlib.pyplot as plt
from PIL import Image

# Asks user for the name of the data set and looks in the keys of the file for the camera length so that calibration doesn't
# need to be manually done every time
data_file = input("Please enter the name of the data file: ")
camera_length = CoM._camera_length(data_file)

##################################################
# Load datafile and remove dead pixels from data. 
dead_pixel_removal = CoM._file_search(data_file.replace(".hdf5","")+"_dead_pixels_removed.hdf5")   

if dead_pixel_removal == False:
    dead_pixel_dataset = CoM.loadh5py(data_file)
    if data_file == "LSMO_LFO_STO7.hdf5":
        dead_pixel_dataset = np.transpose(np.fliplr(dead_pixel_dataset.data))
        # dataset 7 is rotated a full 90 degrees. There seems to be an issue later on with rotating 90 degrees. 
        # I don't want the image arrays reshaped because it interpolates datapoints and I just mask the zeros later on
        # So here we just flip and transpose the dataset before anything is done so that the platinum is, like every
        # other dataset, along the top edge
    
    # Datasets aren't uniform in size so need this to know how to iterate
    size_of_dataset = CoM._dataset_dimensions(copy.deepcopy(dead_pixel_dataset))
        
    # All the dead pixels should appear at the same point in every data set as it is the 
    # same medipix detector however the longer camera length images have lots of zeros from
    # the edge of the aperture. Hence, we run the dead pixel removal for a dataset and save the 
    # indices of the dead pixels so that all future datasets can be corrected similarly 
    dead_pixel_check = CoM._file_search("dead_pixel_indices.npy")

    if dead_pixel_check == False:
        dead_pixel_sum = dead_pixel_dataset.sum(0).sum(0)
        dead_pixel_indices = list(CoM._dead_indices(dead_pixel_sum))
        np.save("dead_pixel_indices", dead_pixel_indices)

    else:
        dead_pixel_indices = np.load("dead_pixel_indices.npy")
    
    hyperspy_dataset = CoM._dead_pixel(dead_pixel_dataset, dead_pixel_indices, data_file)
    del dead_pixel_dataset

else: 
    hyperspy_dataset = hs.load(data_file.replace(".hdf5","")+"_dead_pixels_removed.hdf5")

    # Datasets aren't uniform in size so need this to know how to iterate
    size_of_dataset = CoM._dataset_dimensions(copy.deepcopy(hyperspy_dataset))        
 
####################################################
# Check if the centre's of the image have already been found, if not, perform the calculation, if so, load the array of centres

centre_array = CoM._file_search(data_file.replace(".hdf5", "_centre.npy"))

if centre_array == False:
    dataset_centre = CoM.centre_of_disk_centre_of_mass_full(hyperspy_dataset, data_file, size_of_dataset)

else: 
    dataset_centre = np.load(data_file.replace(".hdf5", "_centre.npy"))

###########################################################
# Radial profiling

# Check if the dataset has been profiled before, if not, perform the radial profile, if so, load the radial profile          
profiled_dataset = CoM._file_search(data_file.replace(".hdf5","") + "_dataset_profile.npy")

if profiled_dataset == False:
    calibrate_dataset = CoM.diffraction_calibration(dataset_centre, size_of_dataset, hyperspy_dataset, 2.2482)
    
    bulk_sto = hyperspy_dataset[:,round(size_of_dataset[1]*0.75):].sum(0).sum(0)
    
# The signal calibration happens here and the calibration value for a certain camera length is saved so that it can be 
# loaded at a later point and the hyperspy signal can be calibrated. Hyperspy signals hold calibration throughout the 
# processing
    
    dataset_radial_profile = CoM.radial_profile_dataset(calibrate_dataset, dataset_centre, size_of_dataset, data_file, bulk_sto, flip = True)
     
else: 
    dataset_radial_profile = np.load(data_file.replace(".hdf5","_dataset_profile.npy"))
    

# This just deletes the dataset from memory as it is no longer needed and can be quite large
del hyperspy_dataset        
#########################################################
# Signal image generation. Returns the gaussian signals as well as key parameters for generating adf images later

gaussian_signals, peak_intensities, platinum_range, step_array, adf_ends = CoM._powerlaw_fitter(data_file, size_of_dataset, bounded_gaussian = True, threshold = True, save_file = True)

adf_image_array = CoM._dark_field_array(dataset_radial_profile, step_array, size_of_dataset, platinum_range, data_file)

#########################################################
# Rotation of data and production of Gaussian mean signals. Checks if the dataset is rotated and then generates an adf
# image that can be used with imageJ to measure the angle from level. The signals and adf images are then rotated accordingly
# so that the mean values are taken orthogonal to the growth direction and electron beam
datasetrotation = input("Is the dataset rotated? 1/0: ")

if datasetrotation == True:
    adf_image = adf_image_array[0]
    plt.imsave(data_file.replace(".hdf5","_adf_image.tif"), adf_image)
    print "Please use the saved image, ", data_file.replace(".hdf5", "_adf_image.tif"), " and imageJ to find the angle of rotation"
    angle_of_rotation = input("What is the angle of rotation (in degrees)?: ")
    gaussian_signal_rotated = CoM._signal_rotation(gaussian_signals, angle_of_rotation)
    adf_array_rotated = CoM._dataset_rotation(adf_image_array, angle_of_rotation)
    gaussian_parameter_means = CoM._gaussian_parameter_mean(gaussian_signal_rotated, adf_ends)
    image_array = [adf_array_rotated, gaussian_signal_rotated, gaussian_parameter_means]
    # image_array[0] is all the adf fields
    # image_array[1] is the signals
    # image_array[2] is the mean values of entire sample and just the film
    clim_set = CoM._clim_set(gaussian_signal_rotated)
    
else: 
    gaussian_parameter_means = CoM._gaussian_parameter_mean(gaussian_signals, adf_ends)
    image_array = [adf_image_array, gaussian_signals, gaussian_parameter_means]
    clim_set = CoM._clim_set(gaussian_signals)

# Saves a h5py file with the adf images and gaussian means. Along with the gaussian signals saved earlier, it means that we
# can access all the produced images at a later date without having to rerun the program
with h5py.File(data_file.replace(".hdf5","") + '_image_array', 'w') as hf:
    g1 = hf.create_group('adf_image')
    g1.create_dataset('adf_image', data = adf_image_array)
    g3 = hf.create_group('gaussian_parameter_means')
    g3.create_dataset('gaussian_parameter_means', data = gaussian_parameter_means)
    
#########################################################
# Final set of images produced and saved
CoM._figure_compare(image_array, peak_intensities, data_file.replace(".hdf5", ""), clim_set, bounded_gaussian = True)
