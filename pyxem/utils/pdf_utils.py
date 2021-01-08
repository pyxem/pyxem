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

import numpy as np


def normalise_pdf_signal_to_max(z, index_min, *args, **kwargs):
    """Used by hs.map in the PairDistributionFunction1D to normalise the signal
    to the maximum in the signal.

    Parameters
    ----------
    z : np.array
        A pair distribution function np.array to be transformed
    index_min : int
        The minimum scattering vector s to be considered, given as the lowest
        index in the array to consider for finding the present maximum.
        This is to prevent the effect of large oscillations at r=0.
    *args:
        Arguments to be passed to map().
    **kwargs:
        Keyword arguments to be passed to map().
    """

    max_val = np.max(z[index_min:])
    return np.divide(z, max_val)
