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

from heapq import nlargest
from operator import itemgetter

from pyxem.utils import correlate
from pyxem.utils.vector_utils import get_rotation_matrix_between_vectors


def correlate_library(image, library, n_largest, mask, keys=[]):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image).

    Parameters
    ----------
    image : np.array()
        The experimental diffraction pattern of interest.
    library : DiffractionLibrary
        The library of diffraction simulations to be correlated with the
        experimental data.
    n_largest : int
        The number of well correlated simulations to be retained.
    mask : bool array
        A mask for navigation axes 1 indicates positions to be indexed.

    Returns
    -------
    out_arr : np.array()
        A numpy array containing the top n correlated simulations for the
        experimental pattern of interest.

    See also
    --------
    pyxem.utils.correlate and the correlate method of IndexationGenerator.

    """
    i = 0
    out_arr = np.zeros((n_largest * len(library), 5))
    if mask == 1:
        for key in library.keys():
            correlations = dict()
            for orientation, diffraction_pattern in library[key].items():
                # diffraction_pattern here is in fact a library of
                # diffraction_pattern_properties
                correlation = correlate(image, diffraction_pattern)
                correlations[orientation] = correlation
                res = nlargest(n_largest, correlations.items(),
                               key=itemgetter(1))
            for j in np.arange(n_largest):
                out_arr[j + i * n_largest][0] = i
                out_arr[j + i * n_largest][1] = res[j][0][0]
                out_arr[j + i * n_largest][2] = res[j][0][1]
                out_arr[j + i * n_largest][3] = res[j][0][2]
                out_arr[j + i * n_largest][4] = res[j][1]
            i = i + 1

    else:
        for j in np.arange(n_largest):
            for k in [0, 1, 2, 3, 4]:
                out_arr[j + i * n_largest][k] = np.nan
        i = i + 1
    return out_arr


def index_magnitudes(z, simulation, tolerance):
    """Assigns hkl indices to peaks in the diffraction profile.

    Parameters
    ----------
    simulation : DiffractionProfileSimulation
        Simulation of the diffraction profile.
    tolerance : float
        The n orientations with the highest correlation values are returned.

    Returns
    -------
    indexation : np.array()
        indexation results.

    """
    mags = z
    sim_mags = np.array(simulation.magnitudes)
    sim_hkls = np.array(simulation.hkls)
    indexation = np.zeros(len(mags), dtype=object)

    for i in np.arange(len(mags)):
        diff = np.absolute((sim_mags - mags.data[i]) / mags.data[i] * 100)

        hkls = sim_hkls[np.where(diff < tolerance)]
        diffs = diff[np.where(diff < tolerance)]

        indices = np.array((hkls, diffs))
        indexation[i] = np.array((mags.data[i], indices))

    return indexation


class Solution(object):
    def __str__(self):
        return str(self.__dict__)


def _eval_solution(solution, qs, A0_inv,
                   eval_tol=0.25,
                   miller_set=None,
                   seed=None,
                   seed_hkl_tol=0.1,
                   indexed_peak_ids=[]):
    """Assigns hkl indices to pairs of diffraction vectors.

    Parameters
    ----------
    ks : np.array()
        The experimentally measured diffraction vectors, associated with a
        particular probe position, to be indexed.
    library : DiffractionLibrary
        Library of reciprocal space vectors to be matched to the vectors.
    mag_threshold : float
        The number of well correlated simulations to be retained.
    angle_threshold : bool array
        A mask for navigation axes 1 indicates positions to be indexed.

    Returns
    -------
    indexation : np.array()
        A numpy array containing the indexation results.

    """
    R = solution.R
    R_inv = np.linalg.inv(R)
    hkls = A0_inv.dot(R_inv.dot(qs.T)).T
    rhkls = np.rint(hkls)
    ehkls = np.abs(hkls - rhkls)
    solution.hkls = hkls
    solution.rhkls = rhkls
    solution.ehkls = ehkls

    if miller_set is None:
        pair_ids = np.where(np.max(ehkls, axis=1) < eval_tol)[0]  # indices of matched peaks
    else:
        _pair_ids = np.where(np.max(ehkls, axis=1) < eval_tol)[0]
        pair_ids = []
        for _pair_id in _pair_ids:
            abs_hkl = np.abs(rhkls[_pair_id])
            if norm(miller_set - abs_hkl, axis=1).min() < epsilon:
                pair_ids.append(_pair_id)
    pair_ids = list(set(pair_ids) - set(indexed_peak_ids))

    nb_pairs = len(pair_ids)
    nb_peaks = len(qs)
    match_rate = float(nb_pairs) / float(nb_peaks)
    solution.pair_ids = pair_ids
    solution.match_rate = match_rate
    solution.nb_pairs = nb_pairs
    # evaluation metrics
    solution.seed_error = ehkls[seed,:].max()
    solution.total_score = match_rate
    if len(pair_ids) == 0:
        # no matching peaks, set error to 1
        solution.total_error = 1.
    else:
        # naive error of matching peaks
        solution.total_error = ehkls[pair_ids].mean()

    return solution


def refine(solution, qs, refine_cycle):
    """Assigns hkl indices to pairs of diffraction vectors.

    Parameters
    ----------
    ks : np.array()
        The experimentally measured diffraction vectors, associated with a
        particular probe position, to be indexed.
    library : DiffractionLibrary
        Library of reciprocal space vectors to be matched to the vectors.
    mag_threshold : float
        The number of well correlated simulations to be retained.
    angle_threshold : bool array
        A mask for navigation axes 1 indicates positions to be indexed.

    Returns
    -------
    indexation : np.array()
        A numpy array containing the indexation results.

    """
    A_refined = solution.A.copy()
    def _fun(x, *argv):
        asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
        h, k, l, qx, qy, qz = argv
        r1 = (asx*h + bsx*k + csx*l - qx)
        r2 = (asy*h + bsy*k + csy*l - qy)
        r3 = (asz*h + bsz*k + csz*l - qz)
        return r1**2. + r2**2. + r3**2.

    def _gradient(x, *argv):
        asx, bsx, csx, asy, bsy, csy, asz, bsz, csz = x
        h, k, l, qx, qy, qz = argv
        r1 = (asx*h + bsx*k + csx*l - qx)
        r2 = (asy*h + bsy*k + csy*l - qy)
        r3 = (asz*h + bsz*k + csz*l - qz)
        g_asx, g_bsx, g_csx = 2.*h*r1, 2.*k*r1, 2.*l*r1
        g_asy, g_bsy, g_csy = 2.*h*r2, 2.*k*r2, 2.*l*r2
        g_asz, g_bsz, g_csz = 2.*h*r3, 2.*k*r3, 2.*l*r3
        return np.asarray((g_asx, g_bsx, g_csx,
                           g_asy, g_bsy, g_csy,
                           g_asz, g_bsz, g_csz))
    rhkls = solution.rhkls
    pair_ids = solution.pair_ids
    for i in range(refine_cycle):
        for j in range(len(pair_ids)):  # refine by each reflection
            pair_id = pair_ids[j]
            x0 = A_refined.reshape((-1))
            rhkl = rhkls[pair_id,:]
            q = qs[pair_id,:]
            args = (rhkl[0], rhkl[1], rhkl[2], q[0], q[1], q[2])
            res = fmin_cg(_fun, x0, fprime=_gradient, args=args, disp=0)
            A_refined = res.reshape((3,3))
    eXYZs = np.abs(A_refined.dot(rhkls.T) - qs.T).T
    dists = norm(eXYZs, axis=1)
    pair_dist = dists[pair_ids].mean()

    if pair_dist < solution.pair_dist:
        solution.A_refined = A_refined
        solution.pair_dist_refined = pair_dist
        solution.hkls_refined = np.linalg.inv(A_refined).dot(qs.T).T
        solution.rhkls_refined = np.rint(solution.hkls_refined)
        solution.ehkls_refined = np.abs(solution.hkls_refined - solution.rhkls_refined)
    else:
        solution.A_refined = solution.A.copy()
        solution.pair_dist_refined = solution.pair_dist
        solution.hkls_refined = solution.hkls.copy()
        solution.rhkls_refined = solution.rhkls.copy()
        solution.ehkls_refined = solution.ehkls.copy()

    return solution


def match_vectors(ks,
                  library,
                  mag_threshold,
                  angle_threshold,
                  keys=[],
                  *args,
                  **kwargs):
    """Assigns hkl indices to pairs of diffraction vectors.

    Parameters
    ----------
    ks : np.array()
        The experimentally measured diffraction vectors, associated with a
        particular probe position, to be indexed.
    library : DiffractionLibrary
        Library of reciprocal space vectors to be matched to the vectors.
    mag_threshold : float
        The number of well correlated simulations to be retained.
    angle_threshold : bool array
        A mask for navigation axes 1 indicates positions to be indexed.

    Returns
    -------
    indexation : np.array()
        A numpy array containing the indexation results.

    """
    i = 0
    indexation = np.zeros((n_largest * len(library), 5))
    for key in library.keys():
        correlations = dict()
        qs = peaks[:, 4:7]
        unindexed_peak_ids = list(set(range(min(qs.shape[0], seed_pool_size))) - set(indexed_peak_ids))
        seed_pool = list(combinations(unindexed_peak_ids, 2))
        good_solutions = []
        # collect good solutions
        for i in tqdm(range(len(seed_pool))):
            seed = seed_pool[i]
            q1, q2 = qs[seed,:]
            q1_len, q2_len = norm(q1), norm(q2)
            if q1_len < q2_len:
                q1, q2 = q2, q1
                q1_len, q2_len = q2_len, q1_len
            angle = calc_angle(q1, q2)
            match_ids = np.where((np.abs(q1_len - table['LA'][:,0]) < seed_len_tol) *
                            (np.abs(q2_len - table['LA'][:,1]) < seed_len_tol) *
                            (np.abs(angle - table['LA'][:,2]) < seed_angle_tol))[0]
            for match_id in match_ids:
                hkl1 = table['hkl1'][match_id]
                hkl2 = table['hkl2'][match_id]
                ref_q1, ref_q2 = A0.dot(hkl1), A0.dot(hkl2)
                solution = Solution()
                solution.R = calc_rotation_matrix(q1, q2, ref_q1, ref_q2)
                solution = eval_solution(solution, qs, A0_inv, eval_tol=eval_tol,
                                         miller_set=miller_set, seed=seed,
                                         seed_hkl_tol=seed_hkl_tol,
                                         indexed_peak_ids=indexed_peak_ids)
                # only keep solution from good seed
                if solution.seed_error <= seed_hkl_tol:
                    good_solutions.append(solution)
        # pick up best solution
        if len(good_solutions) > 0:
            # best solution has highest total score and lowest total error
            good_solutions.sort(key=lambda x: x.total_score, reverse=True)
            best_score = good_solutions[0].total_score
            best_solutions = [solution for solution in good_solutions if solution.total_score==best_score]
            best_solutions.sort(key=lambda x: x.total_error, reverse=False)
            best_solution = best_solutions[0]
        else:
            best_solution = None
        # refine best solution if exists
        if best_solution is None:
            final_solution = Solution()
            final_solution.R = np.identity(3)
            final_solution.match_rate = 0.
        else:
            best_solution.A = best_solution.R.dot(A0)
            # Fourier space error between peaks and predicted spots
            eXYZs = np.abs(best_solution.A.dot(best_solution.rhkls.T) - qs.T).T
            dists = norm(eXYZs, axis=1)
            # average distance between matched peaks and the correspoding predicted spots
            best_solution.pair_dist = dists[best_solution.pair_ids].mean()
            best_solution.A = best_solution.R.dot(A0)
            # refine A matrix with matched pairs to minimize norm(AH-q)
            final_solution = refine(best_solution, qs, refine_cycles)

    return final_solution
