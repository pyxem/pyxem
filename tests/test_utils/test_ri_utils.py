## -*- coding: utf-8 -*-
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

import pytest
import numpy as np
from pyxem.utils.ri_utils import *

def test_as_signal_generation():

    N = 1.
    C = 0.
    elements = ['Cu']
    fracs = [1]
    s_size = 10
    s_scale = 0.1
    types = ['lobato','xtables','not_implemented']

    for type in types:
        signal, normalisation = scattering_to_signal(elements,fracs,N,C,s_size,
                                                        s_scale,type)
        if type == 'lobato':
            expected = np.array([])
            assert_almost_equal(signal,expected)
        elif type == 'xtables':
            expected = np.array([])
            assert_almost_equal(signal,expected)
        else:
            #expect error
            continue


    return
