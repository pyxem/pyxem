# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

# TODO: Delete the whole module come 1.0.0

"""Tools for radial integration"""


from pyxem.utils._deprecated import deprecated
from pyxem.signals.reduced_intensity1d import (
    _damp_ri_exponential,
    _damp_ri_extrapolate_to_zero,
    _damp_ri_lorch,
    _damp_ri_low_q_region_erfc,
    _damp_ri_updated_lorch,
)


@deprecated(since="0.18.0", removal="1.0.0")
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


@deprecated(since="0.18.0", removal="1.0.0")
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


@deprecated(since="0.18.0", removal="1.0.0")
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

    return _damp_ri_exponential(z, b, s_scale, s_size, s_offset, *args, **kwargs)


@deprecated(since="0.18.0", removal="1.0.0")
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

    return _damp_ri_lorch(z, s_max, s_scale, s_size, s_offset, *args, **kwargs)


@deprecated(since="0.18.0", removal="1.0.0")
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

    return _damp_ri_updated_lorch(z, s_max, s_scale, s_size, s_offset, *args, **kwargs)


@deprecated(since="0.18.0", removal="1.0.0")
def damp_ri_extrapolate_to_zero(z, s_min, s_scale, s_size, s_offset, *args, **kwargs):
    """Used by hs.map in the ReducedIntensity1D to extrapolate the reduced
    intensity signal to zero below s_min.

    Parameters
    ----------
    z : np.array
        A reduced intensity np.array to be transformed.
    s_min : float
        Value of s below which data is extrapolated to zero.
    scale : float
        The scattering vector calibation of the reduced intensity array.
    size : int
        The size of the reduced intensity signal. (in pixels)
    *args:
        Arguments to be passed to map().
    **kwargs:
        Keyword arguments to be passed to map().
    """

    return _damp_ri_extrapolate_to_zero(
        z, s_min, s_scale, s_size, s_offset, *args, **kwargs
    )


@deprecated(since="0.18.0", removal="1.0.0")
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

    return _damp_ri_low_q_region_erfc(
        z, scale, offset, s_scale, s_size, s_offset, *args, **kwargs
    )
