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


from hyperspy.signals import Signal1D
import numpy as np

from pyxem.components import ReducedIntensityCorrectionComponent
from scipy import special


class ReducedIntensity1D(Signal1D):
    """Signal class for Reduced Intensity profiles as a function of scattering vector."""

    _signal_type = "reduced_intensity"

    def damp_exponential(self, b: float, inplace: bool = True, *args, **kwargs):
        """Damps the reduced intensity signal to reduce noise in the high s
        region by a factor of exp(-b*(s^2)), where b is the damping parameter.

        Parameters
        ----------
        b : float
            The damping parameter.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().
        """

        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        s_offset = self.axes_manager.signal_axes[0].offset

        return self.map(
            _damp_ri_exponential,
            b=b,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            inplace=inplace,
            *args,
            **kwargs
        )

    def damp_lorch(self, s_max: float = None, inplace: bool = True, *args, **kwargs):
        """Damps the reduced intensity signal to reduce noise in the high s
        region by a factor of sin(s*delta) / (s*delta), where
        delta = pi / s_max. See [1].

        Parameters
        ----------
        s_max : float
            The maximum s value to be used for transformation to PDF.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        References
        ----------
        [1] Lorch, E. (1969). Neutron diffraction by germania, silica and
        radiation-damaged silica glasses. Journal of Physics C: Solid State
        Physics, 2(2), 229.
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        s_offset = self.axes_manager.signal_axes[0].offset
        if not s_max:
            s_max = s_scale * s_size + s_offset

        return self.map(
            _damp_ri_lorch,
            s_max=s_max,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            inplace=inplace,
            *args,
            **kwargs
        )

    def damp_updated_lorch(
        self, s_max: float = None, inplace: bool = True, *args, **kwargs
    ):
        """Damps the reduced intensity signal to reduce noise in the high s
        region by a factor of 3 / (s*delta)^3 (sin(s*delta)-s*delta(cos(s*delta))),
        where delta = pi / s_max. From [1].

        Parameters
        ----------
        s_max : float
            the damping parameter, which need not be the maximum scattering
            vector s to be used for the PDF transform. It's a good guess however
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        References
        ----------
        [1] Soper, A. K., & Barney, E. R. (2011). Extracting the pair
        distribution function from white-beam X-ray total scattering data.
        Journal of Applied Crystallography, 44(4), 714-726.
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        s_offset = self.axes_manager.signal_axes[0].offset
        if not s_max:
            s_max = s_scale * s_size + s_offset

        return self.map(
            _damp_ri_updated_lorch,
            s_max=s_max,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            inplace=inplace,
            *args,
            **kwargs
        )

    def damp_extrapolate_to_zero(self, s_min: float, *args, **kwargs):
        """Extrapolates the reduced intensity to zero linearly below s_min.
        This method is always inplace.

        Parameters
        ----------
        s_min : float
            Value of s below which extrapolation to zero is done.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        s_offset = self.axes_manager.signal_axes[0].offset

        return self.map(
            _damp_ri_extrapolate_to_zero,
            s_min=s_min,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            *args,
            **kwargs
        )

    def damp_low_q_region_erfc(
        self,
        scale: float = 20,
        offset: float = 1.3,
        inplace: bool = True,
        *args,
        **kwargs
    ):
        """Damps the reduced intensity signal in the low q region as a
        correction to central beam effects. The reduced intensity profile is
        damped by (erf(scale * s - offset) + 1) / 2

        Parameters
        ----------
        scale : float
            A scalar multiplier for s in the error function
        offset : float
            A scalar offset affecting the error function.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        s_offset = self.axes_manager.signal_axes[0].offset

        return self.map(
            _damp_ri_low_q_region_erfc,
            scale=scale,
            offset=offset,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            inplace=inplace,
            *args,
            **kwargs
        )

    def fit_thermal_multiple_scattering_correction(
        self, s_max: float = None, plot: bool = False
    ):
        """Fits a 4th order polynomial function to the reduced intensity.
        This is used to calculate the error in the reduced intensity due to
        the effects of multiple and thermal diffuse scattering, which
        results in the earlier background fit being incorrect for either
        low or high angle scattering (or both). A correction is then applied,
        making the reduced intensity oscillate around zero as it should. This
        will distort peak shape. For more detail see [1].

        To use this correction, the fitted data should be fitted to high
        scattering vector, so that the intensity goes to zero at q_max
        (to prevent FFT artifacts).

        Parameters
        ----------
        s_max : float
            Maximum range of fit. The reduced intensity should go to zero
            at this value.
        plot : bool
            Whether to plot the fit after fitting. If True, fit is plotted.
        inplace : bool
            If True (default), this signal is overwritten. Otherwise, returns a
            new signal.
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to be passed to map().

        References
        ----------
        [1] Mu, X. et al. (2013). Evolution of order in amorphous-to-crystalline
        phase transformation of MgF2. Journal of Applied Crystallography, 46(4),
        1105-1116.
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        s_offset = self.axes_manager.signal_axes[0].offset
        if not s_max:
            s_max = s_scale * (s_size + 1) + s_offset

        # scattering_axis = s_scale * np.arange(s_size,dtype='float64')
        fit_model = self.create_model()
        fit_model.append(ReducedIntensityCorrectionComponent())
        fit_model.set_signal_range([0, s_max])
        fit_model.multifit()
        fit_value = fit_model.as_signal()
        if plot:
            fit_model.plot()

        self.data = self.data - fit_value

        return None


def _damp_ri_exponential(z, b, s_scale, s_size, s_offset, *args, **kwargs):
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


def _damp_ri_lorch(z, s_max, s_scale, s_size, s_offset, *args, **kwargs):
    """Damp the reduced intensity signal to reduce noise in the high s region by a factor of
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


def _damp_ri_updated_lorch(z, s_max, s_scale, s_size, s_offset, *args, **kwargs):
    """Damp the reduced intensity signal to reduce noise in the high s region by a factor of
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
    multiplicative_term = np.divide(3 / (delta**3), cubic_array)
    sine_term = np.sin(delta * scattering_axis) - delta * scattering_axis * np.cos(
        delta * scattering_axis
    )

    damping_term = multiplicative_term * sine_term
    damping_term = np.nan_to_num(damping_term)
    return z * damping_term


def _damp_ri_extrapolate_to_zero(z, s_min, s_scale, s_size, s_offset, *args, **kwargs):
    """Extrapolate the reduced intensity signal to zero below s_min.

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

    s_min_num = int((s_min - s_offset) / s_scale)

    s_min_val = z[s_min_num]
    extrapolated_vals = np.arange(s_min_num) * s_scale + s_offset
    extrapolated_vals *= s_min_val / extrapolated_vals[-1]  # scale zero to one

    z[:s_min_num] = extrapolated_vals

    return z


def _damp_ri_low_q_region_erfc(
    z, scale, offset, s_scale, s_size, s_offset, *args, **kwargs
):
    """Damp the reduced intensity signal in the low q region as a correction to central beam
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
        The scattering vector calibration of the reduced intensity array.
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
