# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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
from hyperspy.signals import Signal1D


class DiffractionVectors1D(Signal1D):
    """A Collection of Diffraction Vectors with a defined 1D set of vectors.

    For every navigation position there is specifically one vector that represents
    the dataset.

    Examples of DiffractionVectors1D Signals:
        - STEM_DPC
        - STRAIN Maps
        - Diffraction shifts/ Centers

    Attributes
    ----------
    column_scale : np.array()
        The scale for each column in the signal.  For converting the real values
        to pixel values in some image.

    column_offsets : np.array()
        The offsets for each column in the signal.  For converting the real values
        to pixel values in some image.



    """

    _signal_dimension = 1
    _signal_type = "diffraction_vectors"

    def __init__(self, *args, **kwargs):
        self.column_scale = kwargs.get("column_scale", None)
        self.column_offsets = kwargs.get("column_offsets", None)
        super().__init__(*args, **kwargs)
