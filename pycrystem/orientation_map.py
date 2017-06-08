# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

from hyperspy.signals import Signal2D
import numpy as np

"""
Signal class for crystallographic orientation maps.
"""


class OrientationMap(Signal2D):
    """
    Signal class for crystallographic orientation maps.
    """
    _signal_type = "tensor_field"

    def __init__(self, *args, **kwargs):
        Signal2D.__init__(self, *args, **kwargs)
        # Check that the signal dimensions are (3,3) for it to be a valid
        # TensorField

    def get_angle_map(self):
        pass

    def get_grains(self):
        pass

    def save_txt(self):
        """Save the orientation map as a txt file compatible with other
        packages.
        """
        pass
