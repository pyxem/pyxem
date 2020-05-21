# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

import numpy as np
import scipy.constants as sc


def acceleration_voltage_to_wavelength(acceleration_voltage):
    """Get electron wavelength from the acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float or array-like
        In Volt

    Returns
    -------
    wavelength : float or array-like
        In meters

    Examples
    --------
    >>> import pixstem.diffraction_tools as dt
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> wavelength = dt.acceleration_voltage_to_wavelength(
    ...     acceleration_voltage)
    >>> wavelength_picometer = wavelength*10**12

    """
    E = acceleration_voltage * sc.elementary_charge
    h = sc.Planck
    m0 = sc.electron_mass
    c = sc.speed_of_light
    wavelength = h / (2 * m0 * E * (1 + (E / (2 * m0 * c ** 2)))) ** 0.5
    return wavelength


def diffraction_scattering_angle(acceleration_voltage, lattice_size, miller_index):
    """Get electron scattering angle from a crystal lattice.

    Returns the total scattering angle, as measured from the middle of the
    direct beam (0, 0, 0) to the given Miller index.

    Miller index: h, k, l = miller_index
    Interplanar distance: d = a / (h**2 + k**2 + l**2)**0.5
    Bragg's law: theta = arcsin(electron_wavelength / (2 * d))
    Total scattering angle (phi):  phi = 2 * theta

    Parameters
    ----------
    acceleration_voltage : float
        In Volt
    lattice_size : float or array-like
        In meter
    miller_index : tuple
        (h, k, l)

    Returns
    -------
    angle : float
        Scattering angle in radians.

    Examples
    --------
    >>> import pixstem.diffraction_tools as dt
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> lattice_size = 4e-10  # 4 Ångstrøm, (in meters).
    >>> miller_index = (1, 0, 0)
    >>> scattering_angle = dt.diffraction_scattering_angle(
    ...     acceleration_voltage=acceleration_voltage,
    ...     lattice_size=lattice_size,
    ...     miller_index=miller_index)

    """
    wavelength = acceleration_voltage_to_wavelength(acceleration_voltage)
    H, K, L = miller_index
    a = lattice_size
    d = a / (H ** 2 + K ** 2 + L ** 2) ** 0.5
    scattering_angle = 2 * np.arcsin(wavelength / (2 * d))
    return scattering_angle
