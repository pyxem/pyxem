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

half_side_length = 72

def create_GaAs():
    Ga = pmg.Element("Ga")
    As = pmg.Element("As")
    lattice = pmg.Lattice.cubic(5.6535)
    return pmg.Structure.from_spacegroup("F23",lattice, [Ga,As], [[0, 0, 0],[0.25,0.25,0.25]])

def create_twin_angles():
    " Returns a list contains two sep by rotation of 20"
    twin_angles_1 = [0,0,0]
    twin_angles_2 = np.add(twin_angles_1,[20,0,0])
    return [twin_angles_1,twin_angles_2]

def build_linear_grid_in_euler(alpha_max,beta_max,gamma_max,resolution):
    a = np.arange(0,alpha_max,step=resolution)
    b = np.arange(0,beta_max,step=resolution)
    c = np.arange(0,gamma_max,step=resolution)
    from itertools import product
    return list(product(a,b,c))