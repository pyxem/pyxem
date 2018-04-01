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

def index_magnitudes(x, simulation, tolerance):
    magsx = x[0]
    sim_mags = np.array(simulation.magnitudes)
    sim_hkls = np.array(simulation.hkls)

    indexation = np.zeros(len(magsx), dtype=object)

    for i in np.arange(len(magsx)):
        diff = np.absolute((sim_mags - magsx[i]) / magsx[i] * 100)

        hkls = sim_hkls[np.where(diff < tolerance)]
        diffs = diff[np.where(diff < tolerance)]

        indices = np.array((hkls, diffs))
        indexation[i] = np.array((magsx[i], indices))

    return indexation

def zone_axis_from_indexed_vectors(structure, hkl1, hkl2):
    """Calculate zone axis from two indexed vectors.

    Parameters
    ----------
    structure : Structure
        Structure against which data was indexed.
    hkl1 : np.array
        First indexed g-vector.
    hkl2 : np.array
        Second indexed g-vector.

    Returns
    -------

    """
    l = structure.lattice

    Ai = l.inv_matrix

    gto1 = np.dot(Ai, hkl1)
    gto2 = np.dot(Ai, hkl2)

    n1 = gto1/np.linalg.norm(gto1)
    n2 = gto2/np.linalg.norm(gto2)

    u = np.cross(n1, n2)

    return u / max(np.abs(u))
