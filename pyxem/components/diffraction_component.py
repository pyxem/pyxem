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

"""Forward model component for kinematical electron diffraction.

"""

import numpy as np
from hyperspy.component import Component


class ElectronDiffractionForwardModel(Component):
    """Fits structure deformation to observed diffraction patterns

    Used in a HyperSpy model, this component will iteratively adjust the nine
    entries of a deformation matrix, apply the deformation to the crystal
    structure, and simulate the resultant diffraction pattern, to find the
    best-fit deformation.

    Parameters
    ----------
    electron_diffraction_calculator : DiffractionGenerator
        The model used to simulate electron diffraction patterns from
        structures.
    structure : Structure
        The base crystal structure used for the forward model.
    calibration : float
        Calibration in reciprocal Angstroms per pixel.
    dij : float
        Components of the transformation matrix used to deform the structure.
        Defaults to the identity matrix.

    """

    def __init__(self, electron_diffraction_calculator,
                 structure,
                 calibration,
                 reciprocal_radius,
                 d11=1., d12=0., d13=0.,
                 d21=0., d22=1., d23=0.,
                 d31=0., d32=0., d33=1.):
        Component.__init__(self, ['d11', 'd12', 'd13',
                                  'd21', 'd22', 'd23',
                                  'd31', 'd32', 'd33', ])
        self.electron_diffraction_calculator = electron_diffraction_calculator
        self.structure = structure
        self.reciprocal_radius = reciprocal_radius
        self.calibration = calibration
        self.d11.value = d11
        self.d12.value = d12
        self.d13.value = d13
        self.d21.value = d21
        self.d22.value = d22
        self.d23.value = d23
        self.d31.value = d31
        self.d32.value = d32
        self.d33.value = d33

    def function(self, *args, **kwargs):
        return 1

    def simulate(self):
        """Deforms the structure and simulates the corresponding electron
        diffraction pattern.

        Returns
        -------
        simulation : DiffractionSimulation
            Simulated diffraction pattern of the deformed structure

        """
        diffractor = self.electron_diffraction_calculator
        structure = self.structure
        reciprocal_radius = self.reciprocal_radius
        calibration = self.calibration
        d11 = self.d11.value
        d12 = self.d12.value
        d13 = self.d13.value
        d21 = self.d21.value
        d22 = self.d22.value
        d23 = self.d23.value
        d31 = self.d31.value
        d32 = self.d32.value
        d33 = self.d33.value

        deformation = np.asarray([[d11, d12, d13],
                                  [d21, d22, d23],
                                  [d31, d32, d33]])

        structure.lattice.setLatPar(baserot=deformation)
        simulation = diffractor.calculate_ed_data(structure,
                                                  reciprocal_radius)
        simulation.calibration = calibration
        return simulation
