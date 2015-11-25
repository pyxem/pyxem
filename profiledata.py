import hyperspy.api as hs
import numpy as np
import h5py
import loaddata
import CoM
import copy


def radial_profile_dataset(i, save):
#   Creates a hfd5 of the summed radial profiles for an input
#   hfd5 file

    blankFileToSave = np.zeros((64,64,179)) 
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

    centre = CoM.centre_of_disk_centre_of_mass(copy.deepcopy(
        im.data))[2:]
#   Passes a copy of a numpy array based on the entire image to the 
#   function in CoM to calculate the centre of mass
#   A copy is passed so that the original image isn't altered

    for i in range(0,64):
        for j in range (0,64):
#   Iterates over each individual 256x256 image (4096 in total)

            rad = CoM.radial_profile(im[i,j].data, centre)

            if len(rad) > 185:
                while len(rad) > 185:
                    rad = np.delete(rad, [len(rad)-1])
#           Shortens the array of rad to create a uniform 
#           Length          
          
            blankFileToSave[i,j] = rad

#           Passes the numpy array for each image alone with the
#           Average centre of mass calculated earlier to the 
#           Radial profiler in CoM and saves the profile data
#           In a multidimensional numpy array


    s = hs.signals.Image(blankFileToSave)
    s.save(save + ".hdf5")        
#   Changes the numpy array with the profiles in it into a hyperspy
#   signal and saves it based on the chosen save name
    del im        
    del blankFileToSave
    
def radial_profile_segment(i, save):
    blankFileToSave = np.zeros((64,64, 185)) #the length of rad varies from 185-188
    fpdfile = h5py.File(i,'r') #find data file in a read only format
    data = fpdfile['fpd_expt']['fpd_data']['data'][:]
    im = hs.signals.Image(data[:,:,0,:,:])
    centre = CoM.centre_of_disk_centre_of_mass(im.data)[2:]

    for i in range(0,64):
        for j in range (0,64):
            rad = CoM.radial_profile_segment(im[i,j].data, centre)
                           
            if len(rad) > 185:
                while len(rad) > 185:
                    rad = np.delete(rad, [len(rad)-1])
            
            
            blankFileToSave[i,j] = rad
            del rad 
     
    del im        
    s = hs.signals.Image(blankFileToSave)
    del blankFileToSave
    s.save(save + ".hdf5")        
    
    return

