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

from hyperspy.signals import Signal2D
import numpy as np
from pyxem.signals import push_metadata_through



class StrainMap(Signal2D):
    _signal_type = "strain_map"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        # check init dimension are correct
        self.current_basis_x = [1,0]
        self.current_basis_y = [0,1]


    def rotate_strain_basis(self,x_new):
        # following https://www.continuummechanics.org/stressxforms.html
        # retrived August 2019
        from hyperspy.api import transpose

        def _get_rotation_matrix(x_new):
            # ONLY WORKS FOR self.current_basis_x = [1,0] etc
            try:
                rotation_angle = np.arctan(x_new[1]/x_new[0])
            except ZeroDivisionError:
               rotation_angle = np.deg2rad(90) #check sign on this

            # angle sign agrees with https://en.wikipedia.org/wiki/Rotation_matrix
            R    = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle)],
                             [np.sin(rotation_angle), np.cos(rotation_angle)]])
            return R

        R = _get_rotation_matrix(x_new)
        ratio_array = np.divide(x_new,np.matmul(R,[1,0]))
        if not np.allclose(ratio_array[0],ratio_array[1]):
            print(x_new)
            print(np.matmul(R,[1,0]))
            raise ValueError("Bad rotation matrix")

        def apply_rotation(transposed_strain_map,R=R):
                sigmaxx_old = transposed_strain_map[0]
                sigmayy_old = transposed_strain_map[1]
                sigmaxy_old = transposed_strain_map[2]

                z = np.asarray([[sigmaxx_old,sigmaxy_old],
                                [sigmaxy_old,sigmayy_old]])
                new = np.matmul(R.T,np.matmul(z,R))
                return [new[0,0],new[1,1],new[0,1],transposed_strain_map[3]]

        transposed = transpose(self)[0]
        transposed_to_new_basis = transposed.map(apply_rotation,R=R,inplace=False)

        return StrainMap(transposed_to_new_basis.T)
