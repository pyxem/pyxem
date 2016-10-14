# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
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
"""Forward model component for kinematical electron diffraction.

"""


from __future__ import division

import numpy as np
from pycrystem.utils.sim_utils import get_electron_wavelength,\
    get_structure_factors

from pymatgen.transformations.standard_transformations \
    import DeformStructureTransformation

from hyperspy.component import Component


class ElectronDiffractionCalculator(Component):
    """Computes electron diffraction patterns for a crystal structure.

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage of the microscope in kV
    reciprocal_radius : float
        The maximum radius of the sphere of reciprocal space to sample, in
        reciprocal angstroms.
    excitation_error : float
        The maximum extent of the relrods in reciprocal angstroms. Typically
        equal to 1/{specimen thickness}.

    """

    def __init__(self, electron_diffraction_calculator,
                 structure,
                 D11=1., D12=0., D13=0.,
                 D21=0., D22=1., D23=0.,
                 D31=0., D32=0., D33=1.):
        Component.__init__(self, ['D11',
                                  'D12',
                                  'D13',
                                  'D21',
                                  'D22',
                                  'D23',
                                  'D31',
                                  'D32',
                                  'D33',
                                  ])
        self.electron_diffraction_calculator = electron_diffraction_calculator
        self.structure = structure
        self.D11.value = D11
        self.D12.value = D12
        self.D13.value = D13
        self.D21.value = D21
        self.D22.value = D22
        self.D23.value = D23
        self.D31.value = D31
        self.D32.value = D32
        self.D33.value = D33

    def simulate(self):
        """Calculates the Electron Diffraction data for a structure.

        Parameters
        ----------
        structure : Structure
            The structure for which to derive the diffraction pattern. Note that
            the structure must be rotated to the appropriate orientation.

        Returns
        -------
        DiffractionSimulation
            The data associated with this structure and diffraction setup.

        """
        diffractor = self.electron_diffraction_calculator
        structure = self.structure
        D11 = self.D11.value
        D12 = self.D12.value
        D13 = self.D13.value
        D21 = self.D21.value
        D22 = self.D22.value
        D23 = self.D23.value
        D31 = self.D31.value
        D32 = self.D32.value
        D33 = self.D33.value

        deformation = DeformStructureTransformation([[D11, D12, D13],
                                                     [D21, D22, D23],
                                                     [D31, D32, D33]])
        deformed_structure = deformation.apply_transformation(structure)
        return diffractor.calculate_ed_data(deformed_structure)
