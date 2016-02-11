import CoM
import hyperspy.api as hs
import numpy as np
import scipy as sp
import h5py
import copy
import CoM
import matplotlib.pyplot as plt
import find_centre_using_split_compare

data_file = input("Please enter the name of the data file: ")
hyperspy_dataset = CoM.loadh5py(data_file)

size_of_dataset = CoM._dataset_dimensions(copy.deepcopy(hyperspy_dataset))

centre_array = input("Is there a saved array of centres? 1/0: ")

if centre_array == False:
    numpy_full_centre = CoM.centre_of_disk_centre_of_mass_full(copy.deepcopy(hyperspy_dataset), data_file, size_of_dataset)

else: 
    numpy_full_centre = np.load(data_file.replace(".hdf5", "_CoM_centre.npy"))
          
profiled_dataset = input("Is there a profiled dataset? 1/0: ")

if profiled_dataset == False:
    numpy_full_dataset = CoM.diffraction_calibration(numpy_full_centre, size_of_dataset, copy.deepcopy(hyperspy_dataset), 2.2482)
    
    numpy_full_radial_profile = CoM.radial_profile_dataset(numpy_full_dataset, numpy_full_centre, "numpy_full_profile_" + data_file.replace(".hdf5", ""), size_of_dataset, flip = True)
    
else: 
    numpy_full_radial_profile = np.load("numpy_full_profile_" + data_file.replace(".hdf5", ".npy"))

del hyperspy_dataset        

gaussian_signals, adf_ends, peak_intensities, platinum_range = CoM._powerlaw_fitter(data_file, "numpy_full_profile_", "numpy_full_powerlaw_", bounded_gaussian = True, threshold = True, save_file = True)

step_array = CoM._adf_step(numpy_full_radial_profile, adf_ends)

numpy_full_image_array = CoM._dark_field_array(numpy_full_radial_profile, step_array, size_of_dataset, platinum_range)
numpy_full_image_array.append(gaussian_signals)

dataset_rotation = input("Is the dataset rotated? 1/0: ")

if dataset_rotation == True:
    adf_image = numpy_full_image_array[0]
    adf_image.save("adf_image_" + data_file.replace(".hdf5", ".tif")
    print "Please use the saved image, adf_image_" + data_file.replace(".hdf5", ".tif") + " and imageJ to find the angle of rotation"
    numpy_full_rotated_image_array = CoM._dataset_rotation(numpy_full_image_array)
    gaussian_variable_means = CoM._gaussian_variable_mean(numpy_full_rotated_image_array[6])
    numpy_full_image_array = numpy_full_rotated_image_array
    numpy_full_image_array.append(gaussian_variable_means)    


else: 
    gaussian_variable_means = CoM._gaussian_variable_mean(gaussian_signals)   
    numpy_full_image_array.append(gaussian_variable_means)

clim_set = CoM._clim_set(numpy_full_image_array)

CoM._figure_compare(numpy_full_image_array, "numpy_full_centre " + data_file.replace(".hdf5", "_centre.np"), clim_set, bounded_gaussian = True)
