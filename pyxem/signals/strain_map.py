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

def _get_rotation_matrix(x_new):
        """Internal function to get the rotation matrix that takes [1,0] to x_new

        Parameters
        ----------
        x_new : list
            The coordinates of a point that lies on the new 'x' axis
        Returns
        -------
        R : 2 x 2 numpy asarray
            Contains the correct rotation matrix
        """
        # TODO: facilitate rotations where the original basis is something
        #other than [1,0], probably via a strain_map kwarg

        try:
            rotation_angle = np.arctan(x_new[1]/x_new[0])
        except ZeroDivisionError: #Taking x --> y
           rotation_angle = np.deg2rad(90)

        # angle sign agrees with https://en.wikipedia.org/wiki/Rotation_matrix
        R    = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle)],
                         [np.sin(rotation_angle), np.cos(rotation_angle)]])
        return R


class StrainMap(Signal2D):
    _signal_type = "strain_map"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        # check init dimension are correct

        if 'current_basis_x' in kwargs.keys():
            self.current_basis_x = kwargs['current_basis_x']
        else:
            self.current_basis_x = [1,0]

    def rotate_strain_basis(self,x_new):
        """

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        We follows the mathmatical formalism described in (among other places)
        "https://www.continuummechanics.org/stressxforms.html" (August 2019)
        """

        if self.current_basis_x != [1,0]:
            return ValueError("This functional must act on a strain map in the [1,0] basis")

        from hyperspy.api import transpose
        R = _get_rotation_matrix(x_new)

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

        return StrainMap(transposed_to_new_basis.T,current_basis_x=x_new)
