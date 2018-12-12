# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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
"""Signal class for Reduced Intensity profiles as a fucntion of scattering
vector.

"""

from hyperspy.signals import Signal1D
#??from hyperspy.component import Polynomial
import numpy as np


class ReducedIntensityProfile(Signal1D):
    _signal_type = "reduced_intensity_profile"

    def __init__(self, *args, **kwargs):
        Signal1D.__init__(self, *args, **kwargs)

    def damp_exponential(self,b):
        """ Damps the reduced intensity signal to reduce noise in the high s
        region.

        Parameters
        ----------
        b : the damping parameter, which multiplies the reduced intensity
            profile by exp(-b*(s^2))
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        #should include offset
        scattering_axis = s_scale * np.arange(s_size,dtype='float64')
        damping_term = np.exp(-b * np.square(scattering_axis))
        self.data = self.data * damping_term
        return

    def damp_lorch(self,s_max=None):
        """ Damps the reduced intensity signal to reduce noise in the high s
        region.

        Parameters
        ----------
        q_max : the damping parameter, which should be the maximum q value
        (scattering vector) recorded.
        delta = pi / q_max. The function is damped by sin(q*delta) / (q*delta)
        (from Lorch 1969)
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        if not s_max:
            s_max = s_scale*s_size
        delta = np.pi / s_max

        scattering_axis = s_scale * np.arange(s_size,dtype='float64')
        damping_term = np.sin(delta * scattering_axis) / (delta * scattering_axis)
        damping_term = np.nan_to_num(damping_term)
        self.data = self.data * damping_term
        return

    def damp_updated_lorch(self,s_max=None):
        """ Damps the reduced intensity signal to reduce noise in the high s
        region.

        Parameters
        ----------
        q_max : the damping parameter, which need not be the maximum q
        value (scattering vector) recorded. It's a good guess however
        delta = pi / q_max. The function is damped by
        3 / (q*delta)^3 (sin(q*delta)-q*delta(cos(q*delta)))
        from "Extracting the pair distribution function from white-beam X-ray
        total scattering data", Soper & Barney, 2011
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        if not s_max:
            s_max = s_scale*s_size
        delta = np.pi / s_max

        scattering_axis = s_scale * np.arange(s_size,dtype='float64')
        exponent_array = 3*np.ones(scattering_axis.shape)
        cubic_array = np.power(scattering_axis,exponent_array)
        multiplicative_term = np.divide(3/(delta**3),cubic_array)
        sine_term = (np.sin(delta*scattering_axis)
                    -delta*scattering_axis*np.cos(delta*scattering_axis))

        damping_term = multiplicative_term*sine_term
        damping_term = np.nan_to_num(damping_term)
        self.data = self.data * damping_term
        return

    def fit_thermal_multiple_scattering_correction(self, s_max):
        """ Fits a 4th order polynomial function to the reduced intensity.
        This is used to calculate the error in the reduced intensity due to
        the effects of multiple and thermally diffuse scattering, which
        results in the earlier background fit being incorrect for either
        low or high angle scattering (or both). A correction is then applied,
        making the reduced intensity oscillate around zero as it should. This
        will distort peak shape. For more detail see Mu et al (2014):
        "Evolution of order in amorphous-to-crystalline phase transformation
        of MgF2".

        To use this correction, the fitted data should be fitted to high
        scattering vector, so that the intensity goes to zero at q_max
        (to prevent FFT artifacts).

        Parameters
        ----------
        s_max : maximum range of fit. The reduced intensity should go to zero
                at this value.
        """
        s_scale = self.axes_manager.signal_axes[0].scale
        s_size = self.axes_manager.signal_axes[0].size
        if not s_max:
            s_max = s_scale*s_size

        #scattering_axis = s_scale * np.arange(s_size,dtype='float64')
        fit_model = self.signal.create_model()
        fit_model.append(Polynomial(4))
        fit_model.set_signal_range([0,s_max])
        fit_model.multifit()
        fit_value = fit_model.as_signal()
        fit_model.plot()


        self.data = self.data - fit_value

        return
