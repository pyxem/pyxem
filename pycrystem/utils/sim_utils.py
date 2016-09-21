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
from __future__ import division

from math import radians, sin

from scipy.constants import h, m_e, e, c, pi
import math
import numpy as np
from transforms3d.euler import euler2axangle


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


def get_interaction_constant(accelerating_voltage):
    """Calculates the interaction constant, sigma, for a given
    acelerating voltage.

    Evaluates

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage in V.

    Returns
    -------
    sigma : float
        The relativistic electron wavelength in m.

    """
    E = accelerating_voltage
    wavelength = get_electron_wavelength(accelerating_voltage)
    sigma = (2 * pi * (m_e + e * E))

    return sigma


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
    return np.absolute(np.sum([atom.number * np.exp(
        2 * 1j * np.pi * np.dot(fractional_coordinates, position)) for
                               position, atom in
                               zip(structure.frac_coords, structure.species)],
                              axis=0)) ** 2


def equispaced_s2_grid(theta_range, phi_range, resolution=2.5):
    """Creates rotations approximately equispaced on a sphere.

    Parameters
    ----------
    theta_range : tuple of float
        (theta_min, theta_max)
        The range of allowable polar angles.
    phi_range : tuple of float
        (phi_min, phi_max)
        The range of allowable azimuthal angles.
    resolution : float
        The angular resolution of the grid in degrees.

    Returns
    -------
    s2_grid : list of tuple
        tuple[0] is an array specifying the axis of rotation
        tuple[1] is the angle of rotation in radians

    """
    theta_min, theta_max = [radians(t) for t in theta_range]
    phi_min, phi_max = [radians(r) for r in phi_range]
    resolution = radians(resolution)
    n_theta = int(theta_max/resolution)
    theta_grid = np.linspace(theta_min, theta_max, n_theta+1)
    phi_grid = []
    for j, theta in enumerate(theta_grid):
        steps = max(round(sin(theta) * phi_max / theta_max * n_theta), 1)
        phi = phi_min\
            + np.arange(steps) * (phi_max - phi_min) / steps \
            + (j % 2) * (phi_max - phi_min) / steps / 2
        phi_grid.append(phi)
    s2_grid = np.array(
        [(theta, phi) for phis, theta in zip(phi_grid, theta_grid) for phi in
         phis])
    x_rotations = np.zeros((len(s2_grid),))
    s2_grid = [euler2axangle(ai, aj, ak, 'sxyz') for ai, aj, ak in
               zip(x_rotations, s2_grid[:, 0], s2_grid[:, 1])]
    return s2_grid
