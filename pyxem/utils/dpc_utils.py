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
from hyperspy.signals import Signal2D
import pyxem.utils.diffraction_tools as dt


def make_bivariate_histogram(
    x_position, y_position, histogram_range=None, masked=None, bins=200, spatial_std=3
):
    s0_flat = x_position.flatten()
    s1_flat = y_position.flatten()

    if masked is not None:
        temp_s0_flat = []
        temp_s1_flat = []
        for data0, data1, masked_value in zip(s0_flat, s1_flat, masked.flatten()):
            if not masked_value:
                temp_s0_flat.append(data0)
                temp_s1_flat.append(data1)
        s0_flat = np.array(temp_s0_flat)
        s1_flat = np.array(temp_s1_flat)

    if histogram_range is None:
        if s0_flat.std() > s1_flat.std():
            s0_range = (
                s0_flat.mean() - s0_flat.std() * spatial_std,
                s0_flat.mean() + s0_flat.std() * spatial_std,
            )
            s1_range = (
                s1_flat.mean() - s0_flat.std() * spatial_std,
                s1_flat.mean() + s0_flat.std() * spatial_std,
            )
        else:
            s0_range = (
                s0_flat.mean() - s1_flat.std() * spatial_std,
                s0_flat.mean() + s1_flat.std() * spatial_std,
            )
            s1_range = (
                s1_flat.mean() - s1_flat.std() * spatial_std,
                s1_flat.mean() + s1_flat.std() * spatial_std,
            )
    else:
        s0_range = histogram_range
        s1_range = histogram_range

    hist2d, xedges, yedges = np.histogram2d(
        s0_flat,
        s1_flat,
        bins=bins,
        range=[[s0_range[0], s0_range[1]], [s1_range[0], s1_range[1]]],
    )

    s_hist = Signal2D(hist2d).swap_axes(0, 1)
    s_hist.axes_manager[0].offset = xedges[0]
    s_hist.axes_manager[0].scale = xedges[1] - xedges[0]
    s_hist.axes_manager[1].offset = yedges[0]
    s_hist.axes_manager[1].scale = yedges[1] - yedges[0]
    return s_hist


def bst_to_beta(bst, acceleration_voltage):
    """Calculate beam deflection (beta) values from Bs * t.

    Parameters
    ----------
    bst : NumPy array
        Saturation induction Bs times thickness t of the sample. In Tesla*meter
    acceleration_voltage : float
        In Volts

    Returns
    -------
    beta : NumPy array
        Beam deflection in radians

    Examples
    --------
    >>> import numpy as np
    >>> import pyxem.utils.dpc_utils as dpct
    >>> data = np.random.random((100, 100))  # In Tesla*meter
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> beta = dpct.bst_to_beta(data, acceleration_voltage)

    """
    av = acceleration_voltage
    wavelength = dt.acceleration_voltage_to_wavelength(av)
    e = sc.elementary_charge
    h = sc.Planck
    beta = e * wavelength * bst / h
    return beta


def beta_to_bst(beam_deflection, acceleration_voltage):
    """Calculate Bs * t values from beam deflection (beta).

    Parameters
    ----------
    beam_deflection : NumPy array
        In radians
    acceleration_voltage : float
        In Volts

    Returns
    -------
    Bst : NumPy array
        In Tesla * meter

    Examples
    --------
    >>> import numpy as np
    >>> import pyxem.utils.dpc_utils as dpct
    >>> data = np.random.random((100, 100))  # In radians
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> bst = dpct.beta_to_bst(data, 200000)

    """
    wavelength = dt.acceleration_voltage_to_wavelength(acceleration_voltage)
    beta = beam_deflection
    e = sc.elementary_charge
    h = sc.Planck
    Bst = beta * h / (wavelength * e)
    return Bst


def tesla_to_am(data):
    """Convert data from Tesla to A/m

    Parameters
    ----------
    data : NumPy array
        Data in Tesla

    Returns
    -------
    output_data : NumPy array
        In A/m

    Examples
    --------
    >>> import numpy as np
    >>> import pyxem.utils.dpc_utils as dpct
    >>> data_T = np.random.random((100, 100))  # In tesla
    >>> data_am = dpct.tesla_to_am(data_T)

    """
    return data / sc.mu_0


def acceleration_voltage_to_velocity(acceleration_voltage):
    """Get relativistic velocity of electron from acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float
        In Volt

    Returns
    -------
    v : float
        In m/s

    Example
    -------
    >>> import pyxem.utils.dpc_utils as dpct
    >>> v = dpct.acceleration_voltage_to_velocity(200000) # 200 kV
    >>> round(v)
    208450035

    """
    c = sc.speed_of_light
    av = acceleration_voltage
    e = sc.elementary_charge
    me = sc.electron_mass
    part1 = (1 + (av * e) / (me * c ** 2)) ** 2
    v = c * (1 - (1 / part1)) ** 0.5
    return v


def acceleration_voltage_to_relativistic_mass(acceleration_voltage):
    """Get relativistic mass of electron as function of acceleration voltage.

    Parameters
    ----------
    acceleration_voltage : float
        In Volt

    Returns
    -------
    mr : float
        Relativistic electron mass

    Example
    -------
    >>> import pyxem.utils.dpc_utils as dpct
    >>> mr = dpct.acceleration_voltage_to_relativistic_mass(200000) # 200 kV

    """
    av = acceleration_voltage
    c = sc.speed_of_light
    v = acceleration_voltage_to_velocity(av)
    me = sc.electron_mass
    part1 = 1 - (v ** 2) / (c ** 2)
    mr = me / (part1) ** 0.5
    return mr


def et_to_beta(et, acceleration_voltage):
    """Calculate beam deflection (beta) values from E * t.

    Parameters
    ----------
    et : NumPy array
        Electric field times thickness t of the sample.
    acceleration_voltage : float
        In Volts

    Returns
    -------
    beta: NumPy array
        Beam deflection in radians

    Examples
    --------
    >>> import numpy as np
    >>> import pyxem.utils.dpc_utils as dpct
    >>> data = np.random.random((100, 100))
    >>> acceleration_voltage = 200000  # 200 kV (in Volt)
    >>> beta = dpct.et_to_beta(data, acceleration_voltage)

    """
    av = acceleration_voltage
    e = sc.elementary_charge
    wavelength = dt.acceleration_voltage_to_wavelength(av)
    m = acceleration_voltage_to_relativistic_mass(av)
    h = sc.Planck

    wavelength2 = wavelength ** 2
    h2 = h ** 2

    beta = e * wavelength2 * m * et / h2
    return beta
