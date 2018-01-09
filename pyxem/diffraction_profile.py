# -*- coding: utf-8 -*-
# Copyright 2018 The pyXem developers
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
"""Signal class for Electron Diffraction radial profiles

"""

from hyperspy.components1d import Voigt, Exponential, Polynomial
from hyperspy.signals import Signal1D

from .atomic_scattering_component import AtomicScatteringFunction

import numpy as np

class DiffractionProfile(Signal1D):
    _signal_type = "diffraction_profile"

    def __init__(self, *args, **kwargs):
        Signal1D.__init__(self, *args, **kwargs)

        #TODO: This is for one DP - use map!

    def get_rpdf(self, elements, fracs, s, cut_off, damp=1):
        """Calculate the reduced pair distribution function (rpdf) from a given
        diffracted intensity profile via the reduced intensity.

        Parameters
        ----------
        elements : list
            List of the elements present in the specimen.

        fracs : list
            List of atomic fraction of each element present in the specimen. Must be
            specified in the same order as the elements.

        s : array
            The scattering vector s axis in Angstroms

        cut_off : int
            The direct beam cut off position in pixels.

        damp : int
            Damping exponent. Defaults to 1, which corresponds to no damping.

        Returns
        -------
        rpdf : array
            The reduced pair distribution function.
            
        """
        m = self.create_model()
        Nf2 = AtomicScatteringFunction(elements, fracs)
        m.append(Nf2)
        m.set_signal_range(s[cut_off], np.amax(s))
        m.fit()
        bkg = m.as_signal()

        phi = 4 * (np.pi) * s[cut_off:] * (self.data[cut_off:] - bkg.data[cut_off:]) / (Nf2.N.value * Nf2.fq[cut_off:])

        reduced_intensity = Signal1D(phi * np.exp(-damp * (s[cut_off:]**2)))
        #TODO: This is stupid!
        r = np.linspace(0, 20, 10000)
        r1 = r[cut_off:]
        rvec = r1.reshape(1, r1.size)
        sin = np.sin(np.vstack(s[cut_off:] * 4 * np.pi)@rvec)

        return Signal1D(4*(reduced_intensity[cen:]@sin))
