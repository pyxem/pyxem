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
"""Signal class for two-dimensional diffraction variance."""

from hyperspy.signals import Signal2D

from pyxem.signals.diffraction2d import Diffraction2D


class DiffractionVariance2D(Diffraction2D):
    _signal_type = "diffraction_variance"

    pass


class ImageVariance(Signal2D):
    _signal_type = "image_variance"

    pass
