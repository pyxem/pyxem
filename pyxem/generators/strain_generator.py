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
Low level interface for generating strain results from diffraction vectors
"""

def get_strain_field():
    """
    Vectors Strained : Signal2D
    Vector Unstrained : np.array(2x2)
    kwargs: To be passed to the hyperspy map function

    returns Signal2D of [exx,eyy,theta,exy]
    """
    if 'inplace' in kwargs:
        raise ValueError
    #return vectors_strained.map(_get_single_strain_field(),vector_unstrained=vector_unstrained,kwargs)

    pass

def _get_single_strain_field():
    """
    Vector Strained:   (2x2) np.array
    Vector Unstrained: (2x2) np.array

    returns [exx eyy theta exy]

    raises: error if either vector moves by way to much
    """
