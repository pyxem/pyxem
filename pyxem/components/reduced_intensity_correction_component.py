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
A HyperSpy fitting component to fit a 4th order polynomial that goes to zero
intensity at zero scattering to the reduced intensity profile. This is done to
reduce the effects of multiple scattering in the pdf.

"""
from hyperspy.component import Component


class ReducedIntensityCorrectionComponent(Component):
    def __init__(self, a=0.0, b=0.0, c=0.0, d=0.0):
        """
        Parameters
        ----------
        a : float
        b : float
        c : float
        d : float
            a, b, c, and d are the coefficients of the 1st, 2nd, 3rd, and 4th
            order terms respectively of the returned polynomial.

        Returns
        -------
        p : polynomial of the form ax + bx^2 + cx^3 + dx^4

        """

        Component.__init__(self, ("a", "b", "c", "d"))

    def function(self, x):
        a = self.a.value
        b = self.b.value
        c = self.c.value
        d = self.d.value

        p = a * x + b * (x ** 2) + c * (x ** 3) + d * (x ** 4)
        return p
