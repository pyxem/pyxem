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

from pymatgen.transformations.standard_transformations \
    import DeformStructureTransformation

from hyperspy.component import Component


class ElectronDiffractionForwardModel(Component):
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
                 calibration,
                 d11=1., d12=0., d13=0.,
                 d21=0., d22=1., d23=0.,
                 d31=0., d32=0., d33=1.):
        Component.__init__(self, ['d11',
                                  'd12',
                                  'd13',
                                  'd21',
                                  'd22',
                                  'd23',
                                  'd31',
                                  'd32',
                                  'd33',
                                  ])
        self.electron_diffraction_calculator = electron_diffraction_calculator
        self.structure = structure
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

        deformation = DeformStructureTransformation([[d11, d12, d13],
                                                     [d21, d22, d23],
                                                     [d31, d32, d33]])
        deformed_structure = deformation.apply_transformation(structure)
        sim = diffractor.calculate_ed_data(deformed_structure)
        sim.calibration = calibration
        return sim
