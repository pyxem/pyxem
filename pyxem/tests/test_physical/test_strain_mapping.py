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

import diffpy.structure
import numpy as np

from pyxem.components.scalable_reference_pattern import ScalableReferencePattern
from pyxem.generators.diffraction_generator import DiffractionGenerator
from pyxem.signals.electron_diffraction import ElectronDiffraction


def test_strain_mapping_affine_transform():
    latt = diffpy.structure.lattice.Lattice(3, 3, 3, 90, 90, 90)
    atom = diffpy.structure.atom.Atom(atype='Zn', xyz=[0, 0, 0], lattice=latt)
    structure = diffpy.structure.Structure(atoms=[atom], lattice=latt)
    ediff = DiffractionGenerator(300., 0.025)
    affines = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               [[1.04, 0, 0], [0, 1, 0], [0, 0, 1]],
               [[1.08, 0, 0], [0, 1, 0], [0, 0, 1]],
               [[1.12, 0, 0], [0, 1, 0], [0, 0, 1]]]

    data = []
    for affine in affines:
        # same coords as used for latt above
        latt_rot = diffpy.structure.lattice.Lattice(3, 3, 3, 90, 90, 90, baserot=affine)
        structure.placeInLattice(latt_rot)

        diff_dat = ediff.calculate_ed_data(structure, 2.5)
        dpi = diff_dat.as_signal(64, 0.02, 2.5)
        data.append(dpi.data)
    data = np.array(data)
    dp = ElectronDiffraction(data.reshape((2, 2, 64, 64)))

    m = dp.create_model()
    ref = ScalableReferencePattern(dp.inav[0, 0])
    m.append(ref)
    m.multifit()
    disp_grad = ref.construct_displacement_gradient()

    assert disp_grad.data.shape == np.asarray(affines).reshape(2, 2, 3, 3).shape
