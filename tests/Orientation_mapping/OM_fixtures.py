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

import pymatgen as pmg
import numpy as np
import pyxem as pxm
from transforms3d.euler import euler2axangle
from pymatgen.transformations.standard_transformations import RotationTransformation

half_side_length = 72

def create_GaAs():
    Ga = pmg.Element("Ga")
    As = pmg.Element("As")
    lattice = pmg.Lattice.cubic(5.6535)
    return pmg.Structure.from_spacegroup("F23",lattice, [Ga,As], [[0, 0, 0],[0.25,0.25,0.25]])

def create_pair(angle_start,angle_change):
    """ Lists for angles """
    angle_2 = np.add(angle_start,angle_change)
    return [angle_start,angle_start,angle_2,angle_2]

def build_linear_grid_in_euler(alpha_max,beta_max,gamma_max,resolution):
    a = np.arange(0,alpha_max,step=resolution)
    b = np.arange(0,beta_max,step=resolution)
    c = np.arange(0,gamma_max,step=resolution)
    from itertools import product
    return list(product(a,b,c))

def create_sample(edc,structure,angle_start,angle_change):
    dps = []
    for orientation in create_pair(angle_start,angle_change):
        axis, angle = euler2axangle(orientation[0], orientation[1],orientation[2], 'rzxz')
        rotation = RotationTransformation(axis, angle,angle_in_radians=True)
        rotated_structure = rotation.apply_transformation(structure)
        data = edc.calculate_ed_data(rotated_structure,
                                 reciprocal_radius=0.9, #avoiding a reflection issue
                                 with_direct_beam=False)
        dps.append(data.as_signal(2*half_side_length,0.025,1).data)
    dp = pxm.ElectronDiffraction([dps[0:2],dps[2:]])
    dp.set_calibration(1/half_side_length)
    return dp