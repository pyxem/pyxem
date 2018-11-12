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
"""
A HyperSpy fitting component to fit the independent atomic scattering
background to a radial profile.

"""
import numpy as np

from hyperspy.component import Component
from pyxem.utils.lobato_scattering_params import ATOMIC_SCATTERING_PARAMS

class ScatteringFitComponentLobato(Component):

    def __init__(self, elements, fracs, N = 1., C = 0.):
        """
        N and C are fitting parameters for the Component class.

        Parameters
        ----------
        N = the "slope"
        C = an additive constant

        sum_squares = sum (ci * fi**2 )
        square_sum = (sum(ci * fi))**2 for atomic fraction ci with
        electron scattering factor fi. Used for normalisation.
        """
        Component.__init__(self, ['N', 'C'])
        self.elements = elements
        self.fracs = fracs
        params = []
        for e in elements:
            params.append(ATOMIC_SCATTERING_PARAMS[e])
        self.params = params

    def function(self, x):
        N = self.N.value
        C = self.C.value
        params = self.params
        fracs = self.fracs

        sum_squares = np.zeros(x.size)
        square_sum = np.zeros(x.size)

        for i, element in enumerate(params):
            fi = np.zeros(x.size)
            for n in range(len(element)): #5 parameters per element
                fi += (element[n][0] * (2 + element[n][1] * np.square(2*x))
                    * np.divide(1,np.square(1 + element[n][1] *
                    np.square(2*x))))
            elem_frac = fracs[i]
            sum_squares += np.square(fi)*elem_frac
            square_sum += fi*elem_frac

        self.square_sum = np.square(square_sum)
        #square sum is kept for normalisation.
        return N * sum_squares + C
