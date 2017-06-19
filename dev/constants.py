# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pycrystem as pc

DISTORTION = np.array(
    [
        [1.45, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00]
    ]
)

cubic = pc.Lattice.cubic(5.65)
coords = [
    [0.0,  0.0,  0.0 ],
    [0.5,  0.5,  0.0 ],
    [0.5,  0.0,  0.5 ],
    [0.0,  0.5,  0.5 ],
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
]
atoms = [
    "Ga",
    "Ga",
    "Ga",
    "Ga",
    "As",
    "As",
    "As",
    "As",
]
GAAS = pc.Structure(cubic, atoms, coords)

cubic = pc.Lattice.cubic(3.52)
coords = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
]
atoms = [
    "Ni",
    "Ni",
    "Ni",
    "Ni",
]
NICKEL = pc.Structure(cubic, atoms, coords)
