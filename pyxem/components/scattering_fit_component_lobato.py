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
"""
A HyperSpy fitting component to fit the independent atomic scattering
background to a radial profile.

"""
import numpy as np

from hyperspy.component import Component
from diffsims.utils.lobato_scattering_params import ATOMIC_SCATTERING_PARAMS_LOBATO


class ScatteringFitComponentLobato(Component):
    def __init__(self, elements, fracs, N=1.0, C=0.0):
        """
        A scattering component to fit background atomic scattering.
        Calculates the sum of the squares sum_squares = sum (ci * fi**2 )
        and the square of the sum square_sum = (sum(ci * fi))**2
        for atomic fraction ci with electron scattering factor fi.
        The latter is used for normalisation.

        N and C are fitting parameters for the Component class.

        The fitted function f is of the form:

        f = sum(fi), fi = ai(2 + bi*g^2) / (1 + bi*g^2)^2
        where 1 < i < 5, ai, and bi are tabulated constants, and
        g = 1/d = 2sin(theta)/lambda is the scattering vector magnitude.
        The factors are tabulated in ATOMIC_SCATTERING_PARAMS_LOBATO.


        The parametrisation is detailed in [1].

        Parameters
        ----------
        elements: list of str
            A list of elements present (by symbol).
        fracs: list of float
            A list of fraction of the respective elements. Should sum to 1.
        N : float
            The "slope" of the fit.
        C : float
            An additive constant to the fit.

        References
        ----------
        [1] Lobato, I., & Van Dyck, D. (2014). An accurate parameterization for
        scattering factors, electron densities and electrostatic potentials for
        neutral atoms that obey all physical constraints. Acta Crystallographica
        Section A: Foundations and Advances, 70(6), 636-649.

        """
        Component.__init__(self, ["N", "C"])
        self._whitelist["elements"] = ("init,sig", elements)
        self._whitelist["fracs"] = ("init,sig", fracs)
        self.elements = elements
        self.fracs = fracs
        params = []
        for e in elements:
            params.append(ATOMIC_SCATTERING_PARAMS_LOBATO[e])
        self.params = params
        self.N.value = N
        self.C.value = C

    def function(self, x):
        """
        The actual fitted function. Based on the HyperSpy Component class.
        The function "function" is called innately by HyperSpy's fitting
        algorithm, and represents the appropriate atomic scattering factor
        function presented respectively in Lobato & Van Dyck (2014).

        """
        N = self.N.value
        C = self.C.value
        params = self.params
        fracs = self.fracs

        sum_squares = np.zeros(x.size)
        square_sum = np.zeros(x.size)

        for i, element in enumerate(params):
            fi = np.zeros(x.size)
            for n in range(len(element)):  # 5 parameters per element
                fi += (
                    element[n][0]
                    * (2 + element[n][1] * np.square(x))
                    * np.divide(1, np.square(1 + element[n][1] * np.square(x)))
                )
            elem_frac = fracs[i]
            sum_squares += np.square(fi) * elem_frac
            square_sum += fi * elem_frac

        self.square_sum = np.square(square_sum)
        # square sum is kept for normalisation.
        return N * sum_squares + C
