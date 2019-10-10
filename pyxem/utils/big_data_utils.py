# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

import hyperspy.api as hs
import pyxem as pxm
import numpy as np

def check_consisent_lists_and_chunks(x_list,y_list,chunk_size):
    pass

def load_and_cast(filepath,x,y,chunk_size):
    """
    This function loads a chunk of a larger diffraction pattern
    """
    s = hs.load(filepath,lazy=True)
    s = s.inav[x:x+chunk_size,y:y+chunk_size]
    s.compute()
    return pxm.ElectronDiffraction2D(s)

def factory(fp,x,y,chunk_size,function):
    """
    This loads a chunk of a signal, and then applies (the user defined) function,
    function must take a single argument; dp
    """
    dp = load_and_cast(fp,x,y,chunk_size)
    analysis_output = function(dp)
    return analysis_output

#TODO: tidy the naming up here
#TODO: faciltiate rectangular regions of interest (although square analysis areas)
def _create_vert(l,start_int,gap):
    """
    Internal function that produces the columns that are then stacked to produce
    an output.
    """
    left = start_int*gap
    right = left + gap
    return np.vstack(tuple([x for x in l[left:right]]))

#TODO: tidy the naming up here
#TODO: faciltiate rectangular regions of interest (although square analysis areas)
def _combine(l,x_list,y_list):
    """
    Internal function that combines the local list 'l' into a correctly shaped
    output object.
    """
    i_max,gap = len(x_list),len(y_list)
    vert_list = []
    i = 0
    while i < i_max:
        vert_list.append(_create_vert(l,i,gap))
        i += 1

    np_output = np.hstack(tuple([x for x in vert_list]))
    return np_output

def main_function(fp, x_list,y_list,function):
    """
    #docstrings tbc
    
    Parameters
    ----------
    
    filepath : str
    
    
    x_list : list
    
    
    y_list : list
    
    
    function : function
        A user defined function that take a ElectronDiffraction2D as an argument and returns the desired output    
    
    Returns
    -------
    
    np_output : np.array
    """
    results_list = []
    chunk_size = x_list[1] - x_list[0] #assumed to be == y_list[i] - y_list[j] for (i-j) == 1
    for x in x_list:
        for y in y_list:
            analyis_output = factory(fp,x,y,chunk_size,function)
            results_list.append(analysis_output.data)
    np_output = _combine(l,x_list,y_list)
    return np_output
