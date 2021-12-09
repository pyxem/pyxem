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
"""Signal class for Reduced Intensity profiles as a fucntion of scattering vector."""

from hyperspy.signals import Signal1D

from pyxem.components import ReducedIntensityCorrectionComponent
from pyxem.utils.ri_utils import (
    damp_ri_exponential,
    damp_ri_lorch,
    damp_ri_updated_lorch,
    damp_ri_low_q_region_erfc,
)


class ReducedIntensity1D(Signal1D):
    _signal_type = "reduced_intensity"

    def damp_exponential(self, b, inplace=True, *args, **kwargs):
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
            damp_ri_exponential,
            b=b,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            inplace=inplace,
            *args,
            **kwargs
        )

    def damp_lorch(self, s_max=None, inplace=True, *args, **kwargs):
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
            damp_ri_lorch,
            s_max=s_max,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            inplace=inplace,
            *args,
            **kwargs
        )

    def damp_updated_lorch(self, s_max=None, inplace=True, *args, **kwargs):
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
            damp_ri_updated_lorch,
            s_max=s_max,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            inplace=inplace,
            *args,
            **kwargs
        )

    def damp_low_q_region_erfc(
        self, scale=20, offset=1.3, inplace=True, *args, **kwargs
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
            damp_ri_low_q_region_erfc,
            scale=scale,
            offset=offset,
            s_scale=s_scale,
            s_size=s_size,
            s_offset=s_offset,
            inplace=inplace,
            *args,
            **kwargs
        )

    def fit_thermal_multiple_scattering_correction(self, s_max=None, plot=False):
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

        return
