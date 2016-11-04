# -*- coding: utf-8 -*-
# Copyright 2016 The PyCrystEM developers
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

from pycrystem.tensor_field import BaseTensorField

"""
Signal class for Tensor Fields
"""


class TensorField2D(Signal2D):
    _signal_type = "tensor_field"

    def __init__(self, *args, **kwargs):
        BaseTensorField.__init__(self, *args, **kwargs)
        # Check that the signal dimensions are (3,3) for it to be a valid
        # TensorField
