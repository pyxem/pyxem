# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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
from diffsims.utils.atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
from diffsims.utils.lobato_scattering_params import ATOMIC_SCATTERING_PARAMS_LOBATO
from scipy import special


def subtract_pattern(z, pattern, *args, **kwargs):
    """Used by hs.map in the ReducedIntensityGenerator1D to subtract a background
    pattern.

    Parameters
    ----------
    z : np.array
        A np.array to be transformed
    pattern : np.array
        A numpy array of a single line profile of the same resolution
        (same number of pixels) as the signal to be subtracted from.
    *args:
        Arguments to be passed to map().
    **kwargs:
        Keyword arguments to be passed to map().
    """
    return z - pattern


def mask_from_pattern(z, pattern, *args, **kwargs):
    """Used by hs.map in the ReducedIntensityGenerator1D to mask using a
    background pattern.

    Parameters
    ----------
    z : np.array
        A np.array to be transformed
    pattern : np.array
        A numpy array consisting of 0s and 1s in a single line profile
        of the same resolution (same number of pixels) as the signal.
        1s in the signal are kept. 0s are masked (into zeroes)
    mask_threshold : int or float
        An integer or float threshold. Any pixel in the
        mask_pattern with lower intensity is kept, any with
        higher or equal is set to zero.
    *args:
        Arguments to be passed to map().
    **kwargs:
        Keyword arguments to be passed to map().
    """
    return z * pattern


def damp_ri_exponential(z, b, s_scale, s_size, s_offset, *args, **kwargs):
    """Used by hs.map in the ReducedIntensity1D to damp the reduced
    intensity signal to reduce noise in the high s region by a factor of
    exp(-b*(s^2)), where b is the damping parameter.

    Parameters
    ----------
    z : np.array
        A reduced intensity np.array to be transformed.
    b : float
        The damping parameter.
    scale : float
        The scattering vector calibation of the reduced intensity array.
    size : int
        The size of the reduced intensity signal. (in pixels)
    *args:
        Arguments to be passed to map().
    **kwargs:
        Keyword arguments to be passed to map().
    """

    scattering_axis = s_scale * np.arange(s_size, dtype="float64") + s_offset
    damping_term = np.exp(-b * np.square(scattering_axis))
    return z * damping_term


def damp_ri_lorch(z, s_max, s_scale, s_size, s_offset, *args, **kwargs):
    """Used by hs.map in the ReducedIntensity1D to damp the reduced
    intensity signal to reduce noise in the high s region by a factor of
    sin(s*delta) / (s*delta), where delta = pi / s_max. (from Lorch 1969).

    Parameters
    ----------
    z : np.array
        A reduced intensity np.array to be transformed.
    s_max : float
        The maximum s value to be used for transformation to PDF.
    scale : float
        The scattering vector calibation of the reduced intensity array.
    size : int
        The size of the reduced intensity signal. (in pixels)
    *args:
        Arguments to be passed to map().
    **kwargs:
        Keyword arguments to be passed to map().
    """

    delta = np.pi / s_max

    scattering_axis = s_scale * np.arange(s_size, dtype="float64") + s_offset
    damping_term = np.sin(delta * scattering_axis) / (delta * scattering_axis)
    damping_term = np.nan_to_num(damping_term)
    return z * damping_term


def damp_ri_updated_lorch(z, s_max, s_scale, s_size, s_offset, *args, **kwargs):
    """Used by hs.map in the ReducedIntensity1D to damp the reduced
    intensity signal to reduce noise in the high s region by a factor of
    3 / (s*delta)^3 (sin(s*delta)-s*delta(cos(s*delta))),
    where delta = pi / s_max.

    From "Extracting the pair distribution function from white-beam X-ray
    total scattering data", Soper & Barney, (2011).

    Parameters
    ----------
    z : np.array
        A reduced intensity np.array to be transformed.
    s_max : float
        The damping parameter, which need not be the maximum scattering
        vector s to be used for the PDF transform.
    scale : float
        The scattering vector calibation of the reduced intensity array.
    size : int
        The size of the reduced intensity signal. (in pixels)
    *args:
        Arguments to be passed to map().
    **kwargs:
        Keyword arguments to be passed to map().
    """

    delta = np.pi / s_max

    scattering_axis = s_scale * np.arange(s_size, dtype="float64") + s_offset
    exponent_array = 3 * np.ones(scattering_axis.shape)
    cubic_array = np.power(scattering_axis, exponent_array)
    multiplicative_term = np.divide(3 / (delta ** 3), cubic_array)
    sine_term = np.sin(delta * scattering_axis) - delta * scattering_axis * np.cos(
        delta * scattering_axis
    )

    damping_term = multiplicative_term * sine_term
    damping_term = np.nan_to_num(damping_term)
    return z * damping_term


def damp_ri_low_q_region_erfc(
    z, scale, offset, s_scale, s_size, s_offset, *args, **kwargs
):
    """Used by hs.map in the ReducedIntensity1D to damp the reduced
    intensity signal in the low q region as a correction to central beam
    effects. The reduced intensity profile is damped by
    (erf(scale * s - offset) + 1) / 2

    Parameters
    ----------
    z : np.array
        A reduced intensity np.array to be transformed.
    scale : float
        A scalar multiplier for s in the error function
    offset : float
        A scalar offset affecting the error function.
    scale : float
        The scattering vector calibation of the reduced intensity array.
    size : int
        The size of the reduced intensity signal. (in pixels)
    *args:
        Arguments to be passed to map().
    **kwargs:
        Keyword arguments to be passed to map().
    """

    scattering_axis = s_scale * np.arange(s_size, dtype="float64") + s_offset

    damping_term = (special.erf(scattering_axis * scale - offset) + 1) / 2
    return z * damping_term
