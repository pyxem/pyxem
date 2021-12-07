# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from pyxem.utils.indexation_utils import index_dataset_with_template_rotation
from diffsims.generators.library_generator import DiffractionLibraryGenerator

def find_diffraction_calibration(
    patterns,calibration_guess,library_phases,lib_gen,size,max_excitation_error = 0.01, **kwargs
):
    """Finds the optimal diffraction calibration for a pattern or set of patterns by optimizing correlation scores.
    
    Parameters
    ----------
    patterns : hyperspy:Signal2D object
        Diffration patterns to be iteratively matched to find maximum correlation scores.  Should be of known phase.
    calibration_guess : float
        Inital value for the diffraction calibration in inverse Angstoms per pixel.
    library_phases : diffsims:StructureLibrary Object
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.  Used to create the DiffractionLibrary.
    lib_gen : diffsims:DiffractionLibraryGenerator Object
        Computes a library of electron diffraction patterns for specified atomic
        structures and orientations.  Used to create the DiffractionLibrary.
    size : integer
        How many different steps to test for the first two iterations.
    max_excitation_error : float
        Gets passed to get_diffraction_library
    **kwargs to be passed to index_dataset_with_template_rotation
    
    Returns
    -------
    mean_cal : float
        Mean of calibrations found for each pattern.
    full_corrlines : np.array of shape (size*2 + 20, 2 , number of patterns)
        Gives the explicit correlation vs calibration values.
    found_cals : np.array of shape (number of patterns)
        List of optimal calibration values for each pattern.
    """
    
    images = patterns
    
    num_patterns = images.data.shape[0]
    found_cals = np.zeros((num_patterns))
    found_cals[:] = calibration_guess
    full_corrlines = np.zeros((0,2,num_patterns))
    
    
    
    stepsize = 0.01*calibration_guess
    #first set of checks
    corrlines = _calibration_iteration(images,calibration_guess,library_phases,lib_gen,stepsize,size,num_patterns,max_excitation_error,**kwargs)
    full_corrlines = np.append(full_corrlines,corrlines,axis = 0)
    
    #refined calibration checks
    calibration_guess = full_corrlines[full_corrlines[:,1,:].argmax(axis = 0),0,0].mean()
    corrlines = _calibration_iteration(images,calibration_guess,library_phases,lib_gen,stepsize,size,num_patterns,max_excitation_error,**kwargs)
    full_corrlines = np.append(full_corrlines,corrlines,axis = 0)

    #more refined calibration checks with smaller step
    stepsize = 0.001*calibration_guess
    size = 20
    calibration_guess = full_corrlines[full_corrlines[:,1,:].argmax(axis = 0),0,0].mean()    
    
    corrlines = _calibration_iteration(images,calibration_guess,library_phases,lib_gen,stepsize,size,num_patterns,max_excitation_error,**kwargs)
    full_corrlines = np.append(full_corrlines,corrlines,axis = 0)
    found_cals = full_corrlines[full_corrlines[:,1,:].argmax(axis = 0),0,0]
    

    
    mean_cal = found_cals.mean()
    return mean_cal,full_corrlines, found_cals

def _calibration_iteration(images,calibration_guess,library_phases,lib_gen,stepsize,size,num_patterns,max_excitation_error,**kwargs):
    
    corrlines = np.zeros((0,2,num_patterns))
    temp_line = np.zeros((1,2,num_patterns))
    cal_guess_greater = calibration_guess
    cal_guess_lower = calibration_guess
    for i in range(size//2):
        temp_line[0,0,:] = cal_guess_lower
        temp_line[0,1,:] = _create_check_diflib(images,cal_guess_lower,library_phases,lib_gen,num_patterns,max_excitation_error,**kwargs)
        corrlines = np.append(corrlines, temp_line, axis = 0)
        
        temp_line[0,0,:] = cal_guess_greater
        temp_line[0,1,:] = _create_check_diflib(images,cal_guess_greater,library_phases,lib_gen,num_patterns,max_excitation_error,**kwargs)
        corrlines = np.append(corrlines, temp_line, axis = 0)
        
        cal_guess_lower = cal_guess_lower - stepsize
        cal_guess_greater = cal_guess_greater + stepsize
 
    return corrlines
        
def _create_check_diflib(images,cal_guess,library_phases,lib_gen,num_patterns,max_excitation_error,**kwargs):
    
    half_shape = (images.data.shape[-2]//2, images.data.shape[-1]//2)
    reciprocal_r = np.sqrt(half_shape[0]**2 + half_shape[1]**2)*cal_guess
    diff_lib = lib_gen.get_diffraction_library(library_phases,calibration=cal_guess,reciprocal_radius=reciprocal_r,
                                               half_shape=half_shape,with_direct_beam=False,max_excitation_error=max_excitation_error)

    result, phasedict = index_dataset_with_template_rotation(images,
                                                    diff_lib,
                                                    **kwargs
                                                    )
    correlations = result['correlation'][0,:,0]
    return correlations
