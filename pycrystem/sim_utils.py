# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
#
# This file is part of  PyCrystEM.
#
#  PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division

from scipy.constants import h, m_e, e, c, pi
import math
import numpy as np

def get_electron_wavelength(accelerating_voltage):
    """Calculates the (relativistic) electron wavelength in Angstroms
    for a given accelerating voltage in kV.

    Evaluates

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage in kV.

    Returns
    -------
    wavelength : float
        The relativistic electron wavelength in Angstroms.

    """
    E = accelerating_voltage*1e3
    wavelength = h / math.sqrt(2 * m_e * e * E *\
                               (1 + (e / (2 * m_e * c * c)) * E))*1e10
    return wavelength

def get_structure_factors(fractional_coordinates, structure):
    """Get structure factors

    Patterns
    --------
    fractional_coordinates :

    structure

    Returns
    -------
    structure_factors :

    """
    return np.absolute(np.sum([atom.number*np.exp(2*1j*np.pi*np.dot(fractional_coordinates, position)) for position, atom in zip(structure.frac_coords, structure.species)], axis=0))**2
