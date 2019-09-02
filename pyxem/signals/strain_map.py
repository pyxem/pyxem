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
from pyxem.signals import push_metadata_through, transfer_signal_axes


def _get_rotation_matrix(x_new):
    """Calculate the rotation matrix mapping [1,0] to x_new.

    Parameters
    ----------
    x_new : list
        Coordinates of a point on the new 'x' axis.
     
    Returns
    -------
    R : 2 x 2 numpy.array()
        The rotation matrix.
    """
    try:
        rotation_angle = np.arctan(x_new[1] / x_new[0])
    except ZeroDivisionError:  # Taking x --> y
        rotation_angle = np.deg2rad(90)

    # angle sign agrees with https://en.wikipedia.org/wiki/Rotation_matrix
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])
    return R


class StrainMap(Signal2D):
    """
    Class for storing strain maps, if created within pyxem conventions are:
    The 'y-axis' is 90 degrees from the 'x-axis'
    Positive rotations are anticlockwise.
    """

    _signal_type = "strain_map"

    def __init__(self, *args, **kwargs):
        self, args, kwargs = push_metadata_through(self, *args, **kwargs)
        super().__init__(*args, **kwargs)

        # check init dimension are correct

        if 'current_basis_x' in kwargs.keys():
            self.current_basis_x = kwargs['current_basis_x']
        else:
            self.current_basis_x = [1, 0]

        self.current_basis_y = np.matmul(np.asarray([[0, 1], [-1, 0]]), self.current_basis_x)

    def rotate_strain_basis(self, x_new):
        """ Rotates a strain map to a new basis.

        Parameters
        ----------
        x_new : list
            The coordinates of a point on the new 'x' axis

        Returns
        -------
        StrainMap :
            StrainMap in the new (rotated) basis.

        Notes
        -----
        Conventions are described in the class documentation.
        
        We follow mathmatical formalism described in:
        "https://www.continuummechanics.org/stressxforms.html" (August 2019)
        """

        def apply_rotation(transposed_strain_map, R):
            """ Rotates a strain matrix to a new basis, for which R maps x_old to x_new """
            sigmaxx_old = transposed_strain_map[0]
            sigmayy_old = transposed_strain_map[1]
            sigmaxy_old = transposed_strain_map[2]

            z = np.asarray([[sigmaxx_old, sigmaxy_old],
                            [sigmaxy_old, sigmayy_old]])

            new = np.matmul(R.T, np.matmul(z, R))
            return [new[0, 0], new[1, 1], new[0, 1], transposed_strain_map[3]]

        def apply_rotation_complete(self, R):
            """ Mapping solution to return a (unclassed) strain map in a new basis """
            from hyperspy.api import transpose
            transposed = transpose(self)[0]
            transposed_to_new_basis = transposed.map(apply_rotation, R=R, inplace=False)
            return transposed_to_new_basis.T

        """ Core functionality """

        if self.current_basis_x != [1, 0]:
            # this takes us back to [1,0] if our current map is in a diferent basis
            R = _get_rotation_matrix(self.current_basis_x).T
            strain_map_core = apply_rotation_complete(self, R)
        else:
            strain_map_core = self

        R = _get_rotation_matrix(x_new)
        transposed_to_new_basis = apply_rotation_complete(strain_map_core, R)
        meta_dict = self.metadata.as_dictionary()

        strainmap = StrainMap(transposed_to_new_basis, current_basis_x=x_new, metadata=meta_dict)
        return transfer_signal_axes(strainmap,self)
