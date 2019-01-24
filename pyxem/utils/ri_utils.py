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

import numpy as np
from pyxem.utils.atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
from pyxem.utils.lobato_scattering_params import ATOMIC_SCATTERING_PARAMS_LOBATO

def scattering_to_signal(elements, fracs, N, C, s_size, s_scale, type='lobato'):

    params = []

    if type == 'lobato':
        for e in elements:
            params.append(ATOMIC_SCATTERING_PARAMS_LOBATO[e])
    elif type == 'xtables':
        for e in elements:
            params.append(ATOMIC_SCATTERING_PARAMS[e])
    else:
        raise NotImplementedError("The parameters `{}` are not implemented."
                                  "See documentation for available "
                                  "implementations.".format(type))



    x_size = N.data.shape[0]
    y_size = N.data.shape[1]

    sum_squares = np.zeros((x_size,y_size,s_size))
    square_sum = np.zeros((x_size,y_size,s_size))

    x = np.arange(s_size)*s_scale

    if type == 'lobato':
        for i, element in enumerate(params):
            fi = np.zeros(s_size)
            for n in range(len(element)): #5 parameters per element
                fi += (element[n][0] * (2 + element[n][1] * np.square(2*x))
                    * np.divide(1,np.square(1 + element[n][1] *
                    np.square(2*x))))
            elem_frac = fracs[i]
            sum_squares += np.square(fi)*elem_frac
            square_sum += fi*elem_frac

    elif type == 'xtables':
        for i, element in enumerate(params):
            fi = np.zeros(s_size)
            for n in range(len(element)): #5 parameters per element
                fi += element[n][0] * np.exp(-element[n][1] * (np.square(x)))
            elem_frac = fracs[i]
            sum_squares += np.square(fi)*elem_frac
            square_sum += fi*elem_frac



    signal = N.data.reshape(x_size,y_size,1) * sum_squares + C.data.reshape(x_size,y_size,1)
    square_sum = N.data.reshape(x_size,y_size,1) * square_sum

    return signal, square_sum
