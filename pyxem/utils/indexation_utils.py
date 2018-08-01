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

def get_vector_pair_indexation(z, structure, edc, max_length,
                               magnitudes, indexation,
                               mag_threshold, angle_threshold):
    """Determine valid indexations for pairs of vectors based on
    their magnitude and the angle between them.

    Parameters
    ----------
    structure : Structure
        The structure for which to calculate the interplanar angle.
    edc : DiffractionGenerator
        Miller indices of first plane.
    guni : DiffractionVectors
        Miller indices of second plane.
    maximum_length : float
        maximum g-vector magnitude for simulation.
    mag_threshold : float
        maximum percentage error in g-vector magnitude for indexation.
    angle_threshold : float
        maximum angular error in radians for g-vector pair interplanar angle.

    Returns
    -------
    indexed_pairs : array
        Array containing possible consistent indexations for each pair
        of g-vectors.

    """
    vectors = z[0]
    #set various theoretical structural parameters
    rl = structure.lattice.reciprocal_lattice_crystallographic
    recip_pts = rl.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_length)
    calc_peaks = np.asarray(sorted(recip_pts, key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])))
    #convert array of unique vectors to polar coordinates
    gpolar = np.array(_cart2polar(vectors.data.T[0], vectors.data.T[1]))
    #assign empty list for
    indexed_pairs = []
    #iterate through all pairs calculating theoretical interplanar angle
    for comb in itertools.combinations(np.arange(len(vectors)), 2):
        i, j = comb[0], comb[1]
        #get hkl values for all planes in indexed family
        hkls1 = calc_peaks.T[0][np.where(np.isin(calc_peaks.T[1], indexation[i][1][1]))]
        hkls2 = calc_peaks.T[0][np.where(np.isin(calc_peaks.T[1], indexation[j][1][1]))]
        #assign empty array for indexation results
        phis = np.zeros((len(hkls1), len(hkls2)))
        #iterate through all pairs of indices
        for prod in itertools.product(np.arange(len(hkls1)), np.arange(len(hkls2))):
            m, n = prod[0], prod[1]
            hkl1, hkl2 = hkls1[m], hkls2[n]
            phis[m,n] = get_interplanar_angle(structure, hkl1, hkl2)
        #calculate experimental interplanar angle
        phi_expt = gpolar[1][j] - gpolar[1][i]
        #compare theory with experiment with threshold on mag of difference
        phi_diffs = phis - phi_expt
        valid_pairs = np.array(np.where(np.abs(phi_diffs)<angle_threshold))
        #obtain Miller indices corresponding to planes satisfying mag + angle.
        indexed_pairs.append([vectors.data[i], hkls1[valid_pairs[0]], vectors.data[j], hkls2[valid_pairs[1]]])
    #results give two arrays containing Miller indices for each reflection in pair that are self consistent.
    return np.array(indexed_pairs)

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
