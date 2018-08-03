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

import pytest

import numpy as np
import pymatgen as pmg
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

from pyxem.generators.diffraction_generator import DiffractionGenerator
from pyxem.signals.electron_diffraction import ElectronDiffraction

def test_strain_mapping_affine_transform():
    si = pmg.Element("Si")
    lattice = pmg.Lattice.cubic(5.431)
    structure = pmg.Structure.from_spacegroup("Fd-3m",lattice, [si], [[0, 0, 0]])
    ediff = DiffractionGenerator(300., 0.025)
    affines = [[[1, 0, 0], [0, 1, 0], [0, 0,  1]],
           [[1.002, 0, 0], [0, 1, 0], [0, 0,  1]],
           [[1.004, 0, 0], [0, 1, 0], [0, 0,  1]],
           [[1.006, 0, 0], [0, 1, 0], [0, 0,  1]]]

    data = []
    for affine in affines:
        deform = DeformStructureTransformation(affine)
        strained = deform.apply_transformation(structure)
        diff_dat = ediff.calculate_ed_data(strained, 2.5)
        dpi = diff_dat.as_signal(64,0.02, 2.5)
        data.append(dpi.data)
    data = np.array(data)
    dp = ElectronDiffraction(data.reshape((2,2,64,64)))

    m = dp.create_model()
    ref = ScalableReferencePattern(dp.inav[0,0])
    m.append(ref)
    m.multifit()
    disp_grad = ref.construct_displacement_gradient()
    answer = np.array([[[[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00],
         [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00],
         [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]],

        [[  8.20923989e-01,  -1.17750532e-05,   0.00000000e+00],
         [  1.04362853e-06,   1.00000103e+00,   0.00000000e+00],
         [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]],


       [[[  8.17788402e-01,   3.05997521e-05,   0.00000000e+00],
         [  5.14509452e-06,   1.00000089e+00,   0.00000000e+00],
         [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]],

        [[  8.35984229e-01,   3.69499435e-06,   0.00000000e+00],
         [ -4.89047812e-07,   1.00000160e+00,   0.00000000e+00],
         [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]]]])
    np.testing.assert_almost_equal(disp_grad.data, answer, decimal=4)
