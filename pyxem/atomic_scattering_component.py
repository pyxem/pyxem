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


from hyperspy.component import Component
from .utils.atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
import numpy as np


class AtomicScatteringFunction(Component):
    """Atomic scattering function for specified elements and corresponding
    atomic fractions.

    Parameters
    ----------
    elements : list
        List of the elements present in the specimen.

    fracs : list
        List of atomic fraction of each element present in the specimen. Must be
        specified in the same order as the elements.

    Attributes
    ----------
    N : float
        Scaling factor applied to the scattering function to be fitted.

    C : float
        Constant background term to be fitted.

    """

    def __init__(self, elements, fracs, N=1., C=0.):

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

        # gq is sum(f**2*frac) and is used for the fitting
        gq = np.zeros(x.size)
        # fq is sum(f*frac)**2 and is needed in the denominator of phi
        fq = np.zeros(x.size)
        for j in range(len(fracs)):
            # Finding f for each element
            f = np.zeros(x.size)
            for i in range(5):
                f = f + params[j][i][0]*np.exp(-params[j][i][1]*(x**2))
            gq += (f**2)*fracs[j]
            fq += f*fracs[j]
        # TODO: This is a bit weird, where is this used?
        fq2 = fq**2
        self.fq2 = fq2

        return N * gq + C
