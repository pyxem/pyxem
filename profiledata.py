import hyperspy.api as hs
import numpy as np
import h5py
import loaddata
import CoM
import copy


def radial_profile_dataset(i, save):
    blankFileToSave = np.zeros((64,64,179)) #the length of rad varies from 185-188
    fpdfile = h5py.File(i,'r') #find data file in a read only format
    data = fpdfile['fpd_expt']['fpd_data']['data'][:]
    im = hs.signals.Image(data[:,:,0,:,:])
    centre = CoM.centre_of_disk_centre_of_mass(copy.deepcopy(im.data))[2:]
    for i in range(0,64):
        for j in range (0,64):
            rad = CoM.radial_profile(im[i,j].data, centre)
            blankFileToSave[i,j] = rad
      
    s = hs.signals.Image(blankFileToSave)
    s.save(save + ".hdf5")        
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

