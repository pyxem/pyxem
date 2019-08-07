# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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

# over importing
import hyperspy.api as hs
import pyxem as pxm
import numpy as np
from pyxem.generators.subpixelrefinement_generator import SubpixelrefinementGenerator
from pyxem.signals.tensor_field import *
from pyxem.generators.displacement_gradient_tensor_generator import *


def load_and_dp(fp,x,y,chunk):
    """
    This function loads a chunk of signal
    """
    s = hs.load(fp,lazy=True)
    s = s.inav[x:x+chunk,y:y+chunk]
    s.compute()
    return pxm.ElectronDiffraction2D(s)

def factory(fp,x,y,chunk,function):
    """
    This loads a chunk of a signal, and then applies (the user defined) function,
    function must take a single argument; dp
    """
    dp = load_and_dp(fp,x,y,chunk)
    vectors = function(dp)
    return vectors


def _create_vert(l,start_int,gap):
    """
    Internal function that produces the columns that are then stacked to produce
    an output.
    """
    left = start_int*gap
    right = left + gap
    return np.vstack(tuple([x for x in l[left:right]]))

def _combine(l,x_list,y_list):
    """
    Internal function that combines the local list 'l' into a correctly shaped
    output object.
    """
    i_max,gap = len(x_list),len(y_list)
    vert_list = []
    i = 0
    while i < i_max:
        vert_list.append(_create_vert(l,i,gap))
        i += 1

    Vs_final = np.hstack(tuple([x for x in vert_list]))
    return Vs_final

def main_function(fp, x_list,y_list,function):
    """
    Produces a full output for fp, x_list,y_list, function, this assumes x_list
    and y_list are linearly spaced.
    """
    l = []
    for x in x_list:
        for y in y_list:
            dp = factory(fp,x,y,x_list[1]-x_list[0],function)
            l.append(dp.data)
    z = _combine(l,x_list,y_list)
    return z
