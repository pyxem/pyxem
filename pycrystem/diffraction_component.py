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
from hyperspy.component import Component


class ElectronDiffractionCalculator(Component):
    """Computes electron diffraction patterns for a crystal structure.

    1. Calculate reciprocal lattice of structure. Find all reciprocal points
       within the limiting sphere given by :math:`\\frac{2}{\\lambda}`.

    2. For each reciprocal point :math:`\\mathbf{g_{hkl}}` corresponding to
       lattice plane :math:`(hkl)`, compute the Bragg condition
       :math:`\\sin(\\theta) = \\frac{\\lambda}{2d_{hkl}}`

    3. The intensity of each reflection is then given in the kinematic
       approximation as the modulus square of the structure factor.
           .. math::
                I_{hkl} = F_{hkl}F_{hkl}^*

    .. todo::
        Include camera length, when implemented.
    .. todo::
        Refactor the excitation error to a structure property.

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

    def __init__(self,
                 accelerating_voltage,
                 reciprocal_radius,
                 excitation_error,
                 structure,
                 orientation_matrix):
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

        self.wavelength.value = get_electron_wavelength(accelerating_voltage)
        self.reciprocal_radius.value = reciprocal_radius
        self.excitation_error.value = excitation_error

    def forward_calculator(self, structure):
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
        rotation = RotationTransformation(axis, angle,
                                  angle_in_radians=True)
        rotated_structure = rotation.apply_transformation(structure)
        data = diffractor.calculate_ed_data(rotated_structure)

        return DiffractionSimulation(
            coordinates=intersection_coordinates,
            indices=intersection_indices,
            intensities=intersection_intensities
        )
