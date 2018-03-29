# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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
import pytest

desired_ratio = 13.5996633 / (1 + np.sqrt(2))**3
#https://en.wikipedia.org/wiki/Truncated_cube#Cartesian_coordinates 
# Octagon side length is unity
    
master_array = (np.random.rand(500000,3))*((1+np.sqrt(2))*np.sqrt(3)/2) #upper octet random points

def build_linear_grid_in_euler(alpha_min,alpha_max,resolution):
    a = np.arange(alpha_min,alpha_max,step=resolution)
    from itertools import product
    return list(product(a,a,a))

master_array = build_linear_grid_in_euler(0,(1+np.sqrt(2))*np.sqrt(3)/2,0.0251) 
ext_corner = np.asarray([1,1,1])*(((1+np.sqrt(2))*np.sqrt(3)/2))
int_corner = ext_corner - (1/np.sqrt(2))*np.asarray([1,1,1])    
    
absent = 0
for p in master_array:
        if np.linalg.norm(int_corner-p) > np.linalg.norm(ext_corner-p) :
            absent += 1
            
found_ratio = (len(master_array) - absent) / len(master_array)    
