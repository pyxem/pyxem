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
from pyxem.utils.atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
from pyxem.utils.lobato_scattering_params import ATOMIC_SCATTERING_PARAMS_LOBATO

class ScatteringFitComponent(Component):

    def __init__(self, elements, fracs, N=1., C=0., type='lobato'):
        """
        N and C are fitting parameters for the Component class.

        Parameters
        ----------
        N = the "slope"
        C = an additive constant
        type = the type of scattering parameter fit done.
                Options are:
                    - lobato: Fit to Lobato & Van Dyck (2014)
                    - xtables: Fit to International Tables Vol. C, table 4.3.2.3

        sum_squares = sum (ci * fi**2 )
        square_sum = (sum(ci * fi))**2 for atomic fraction ci with
        electron scattering factor fi. Used for normalisation.
        """
        Component.__init__(self, ['N', 'C'])
        self.type = type
        self.elements = elements
        self.fracs = fracs
        params = []
        if self.type == 'lobato':
            for e in elements:
                params.append(ATOMIC_SCATTERING_PARAMS_LOBATO[e])
        elif self.type == 'xtables':
            for e in elements:
                params.append(ATOMIC_SCATTERING_PARAMS[e])
        else:
            raise NotImplementedError("The parameters `{}` are not implemented."
                                      "See documentation for available "
                                      "implementations.".format(type))
            # As in International Tables for Crystallography, 2006, 4.3.2
        self.params = params

    def function(self, x):
        N = self.N.value
        C = self.C.value
        params = self.params
        fracs = self.fracs

        sum_squares = np.zeros(x.size)
        square_sum = np.zeros(x.size)

        if self.type == 'lobato':
            for i, element in enumerate(params):
                fi = np.zeros(x.size)
                for n in range(len(element)): #5 parameters per element
                    fi += (element[n][0] * (2 + element[n][1] * np.square(2*x))
                        * np.divide(1,np.square(1 + element[n][1] *
                        np.square(2*x))))
                elem_frac = fracs[i]
                sum_squares += np.square(fi)*elem_frac
                square_sum += fi*elem_frac

        elif self.type == 'xtables':
            for i, element in enumerate(params):
                fi = np.zeros(x.size)
                for n in range(len(element)): #5 parameters per element
                    fi += element[n][0] * np.exp(-element[n][1] * (np.square(x)))
                elem_frac = fracs[i]
                sum_squares += np.square(fi)*elem_frac
                square_sum += fi*elem_frac

        else:
            print('Error. Specified type does not exist.')
            return

        self.square_sum = N * np.square(square_sum)
        #square sum is kept for normalisation.
        return N * sum_squares + C
