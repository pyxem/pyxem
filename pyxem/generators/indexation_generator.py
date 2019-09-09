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

"""Indexation generator and associated tools.

"""

import numpy as np
import hyperspy.api as hs

from pyxem.signals.indexation_results import TemplateMatchingResults
from pyxem.signals.indexation_results import VectorMatchingResults

from pyxem.signals import transfer_navigation_axes

from pyxem.utils.indexation_utils import correlate_library
from pyxem.utils.indexation_utils import index_magnitudes
from pyxem.utils.indexation_utils import match_vectors
from pyxem.utils.indexation_utils import OrientationResult

from collections import namedtuple
from operator import attrgetter

from transforms3d.euler import mat2euler, euler2mat
from pyxem.utils.vector_utils import detector_to_fourier
from diffsims.utils.sim_utils import get_electron_wavelength

import lmfit

class IndexationGenerator():
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    signal : ElectronDiffraction2D
        The signal of electron diffraction patterns to be indexed.
    diffraction_library : DiffractionLibrary
        The library of simulated diffraction patterns for indexation.
    """

    def __init__(self,
                 signal,
                 diffraction_library):
        self.signal = signal
        self.library = diffraction_library

    def correlate(self,
                  n_largest=5,
                  mask=None,
                  inplane_rotations=np.arange(0, 360, 1),
                  max_peaks=100,
                  *args,
                  **kwargs):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : int
            The n orientations with the highest correlation values are returned.
        mask : Array
            Array with the same size as signal (in navigation) True False
        inplane_rotations : ndarray
            Array of inplane rotation angles in degrees. Defaults to 0-360 degrees
            at 1 degree resolution.
        max_peaks : int
            Maximum number of peaks to consider when comparing a template to
            the diffraction pattern. The strongest peaks are kept.
        *args : arguments
            Arguments passed to map().
        **kwargs : arguments
            Keyword arguments passed map().

        Returns
        -------
        matching_results : TemplateMatchingResults
            Navigation axes of the electron diffraction signal containing
            correlation results for each diffraction pattern, in the form
            [Library Number , [z, x, z], Correlation Score]

        """
        signal = self.signal
        library = self.library
        inplane_rotations = np.deg2rad(inplane_rotations)
        num_inplane_rotations = inplane_rotations.shape[0]
        sig_shape = signal.axes_manager.signal_shape
        signal_half_width = sig_shape[0] / 2

        if mask is None:
            # Index at all real space pixels
            mask = 1

        # Create a copy of the library, cropping and padding the peaks to match
        # max_peaks. Also create rotated pixel coordinates according to
        # inplane_rotations
        rotation_matrices_2d = np.array([[[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]] for t in inplane_rotations])
        cropped_library = {}

        for phase_name, phase_entry in library.items():
            num_orientations = len(phase_entry['orientations'])
            intensities_jagged = phase_entry['intensities']
            intensities = np.zeros((num_orientations, max_peaks))
            pixel_coords_jagged = phase_entry['pixel_coords']
            pixel_coords = np.zeros((num_inplane_rotations, num_orientations, max_peaks, 2))
            for i in range(num_orientations):
                num_peaks = min(pixel_coords_jagged[i].shape[0], max_peaks)
                highest_intensity_indices = np.argpartition(intensities_jagged[i], -num_peaks)[-num_peaks:]
                intensities[i, :num_peaks] = intensities_jagged[i][highest_intensity_indices]
                # Get and compute pixel coordinates for all rotations about the
                # center, clipped to the detector size and rounded to integer positions.
                pixel_coords[:, i, :num_peaks] = np.clip(
                    (signal_half_width + rotation_matrices_2d @ (
                        pixel_coords_jagged[i][highest_intensity_indices].T - signal_half_width)).transpose(0, 2, 1),
                    a_min=0,
                    a_max=np.array(sig_shape) - 1)

            np.rint(pixel_coords, out=pixel_coords)
            cropped_library[phase_name] = {
                'orientations': phase_entry['orientations'],
                'pixel_coords': pixel_coords.astype('int'),
                'intensities': intensities,
                'pattern_norms': np.linalg.norm(intensities, axis=1),
            }

        matches = signal.map(correlate_library,
                             library=cropped_library,
                             n_largest=n_largest,
                             mask=mask,
                             inplace=False,
                             **kwargs)

        matching_results = TemplateMatchingResults(matches)
        matching_results = transfer_navigation_axes(matching_results, signal)

        return matching_results


class ProfileIndexationGenerator():
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    profile : ElectronDiffraction1D
        The signal of diffraction profiles to be indexed.
    library : ProfileSimulation
        The simulated profile data.

    """

    def __init__(self, magnitudes, simulation, mapping=True):
        self.map = mapping
        self.magnitudes = magnitudes
        self.simulation = simulation

    def index_peaks(self,
                    tolerance=0.1,
                    *args,
                    **kwargs):
        """Assigns hkl indices to peaks in the diffraction profile.

        Parameters
        ----------
        tolerance : float
            The n orientations with the highest correlation values are returned.
        keys : list
            If more than one phase present in library it is recommended that
            these are submitted. This allows a mapping from the number to the
            phase.  For example, keys = ['si','ga'] will have an output with 0
            for 'si' and 1 for 'ga'.
        *args : arguments
            Arguments passed to the map() function.
        **kwargs : arguments
            Keyword arguments passed to the map() function.

        Returns
        -------
        matching_results : ProfileIndexation

        """
        return index_magnitudes(np.array(self.magnitudes), self.simulation, tolerance)


def get_nth_best_solution(single_match_result, rank=0):
    """Get the nth best solution by match_rate from a pool of solutions

    Parameters
    ----------
    single_match_result : VectorMatchingResults, TemplateMatchingResults
        Pool of solutions from the vector matching algorithm
    rank : int
        The rank of the solution, i.e. rank=2 returns the third best solution

    Returns
    -------
    VectorMatching: 
        best_fit : `OrientationResult`
            Parameters for the best fitting orientation
            Library Number, rotation_matrix, match_rate, error_hkls, total_error
    TemplateMatching: np.array
            Parameters for the best fitting orientation
            Library Number , [z, x, z], Correlation Score
    """
    try:
        print(single_match_result.shape)
        try:
            best_fit = sorted(single_match_result[0].tolist(), key=attrgetter('match_rate'), reverse=True)[rank]
            print("1", type(best_fit))
        except AttributeError:
            best_fit = sorted(single_match_result.tolist(), key=attrgetter('match_rate'), reverse=True)[rank]
            print("2", type(best_fit))
    except:
        srt_idx = np.argsort(single_match_result[:, 2])[rank]
        best_fit = single_match_result[rank]
    return best_fit


def _refine_best_orientation(single_match_result, 
                             vectors,
                             library,
                             accelarating_voltage,
                             camera_length,
                             rank=0,
                             index_error_tol=0.2,
                             method="leastsq"
                             ):
    """
    Refine a single orientation agains the given cartesian vector coordinates.

    Parameters
    ----------
    single_match_result : VectorMatchingResults
        Pool of solutions from the vector matching algorithm
    rank : int
        The rank of the solution, i.e. rank=2 returns the third best solution    solution : list
        np.array containing the initial orientation
    vectors : DiffractionVectors
        DiffractionVectors to be indexed.
    structure_library : :obj:`diffsims:StructureLibrary` Object
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    index_error_tol : float
        Max allowed error in peak indexation for classifying it as indexed,
        calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
    method : str
        Minimization algorithm to use, choose from: 
        'leastsq', 'nelder', 'powell', 'cobyla', 'least-squares'.
        See `lmfit` documentation (https://lmfit.github.io/lmfit-py/fitting.html)
        for more information.

    Returns
    -------
    result : OrientationResult
        Container for the orientation refinement results
    """
    solution = get_nth_best_solution(single_match_result, rank=rank)
    
    result = _refine_orientation(solution, 
                                 vectors, 
                                 library,
                                 accelarating_voltage=accelarating_voltage,
                                 camera_length=camera_length,
                                 index_error_tol=index_error_tol,
                                 method=method)

    return result


def _refine_orientation(solution, 
                        k_xy,
                        structure_library, 
                        accelarating_voltage,
                        camera_length,
                        index_error_tol=0.2,
                        method="leastsq",
                        verbose=False,
                        ):
    """
    Refine a single orientation agains the given cartesian vector coordinates.

    Parameters
    ----------
    solution : OrientationResult
        Namedtuple containing the starting orientation
    k_xy : DiffractionVectors
        DiffractionVectors (x,y pixel format) to be indexed.
    structure_library : :obj:`diffsims:StructureLibrary` Object
        Dictionary of structures and associated orientations for which
        electron diffraction is to be simulated.
    index_error_tol : float
        Max allowed error in peak indexation for classifying it as indexed,
        calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
    method : str
        Minimization algorithm to use, choose from: 
        'leastsq', 'nelder', 'powell', 'cobyla', 'least-squares'.
        See `lmfit` documentation (https://lmfit.github.io/lmfit-py/fitting.html)
        for more information.
    verbose : bool
        Be more verbose

    Returns
    -------
    result : OrientationResult
        Container for the orientation refinement results
    """

    # prepare reciprocal_lattice
    structure = structure_library.structures[solution.phase_index]
    lattice_recip = structure.lattice.reciprocal()
    
    def objfunc(params, k_xy, lattice_recip, wavelength, camera_length):
        cx = params["center_x"].value
        cy = params["center_y"].value
        ai = params["ai"].value
        aj = params["aj"].value
        ak = params["ak"].value
        scale = params["scale"].value
        
        rotmat = euler2mat(ai, aj, ak)

        k_xy = (k_xy + np.array((cx, cy)) * scale)
        cart = detector_to_fourier(k_xy, wavelength, camera_length)
        
        intermediate = cart.dot(rotmat.T) # Must use the transpose here
        hklss = lattice_recip.fractional(intermediate) * scale
        
        rhklss = np.rint(hklss)
        ehklss = np.abs(hklss - rhklss)
        
        return ehklss
    
    ai, aj, ak = mat2euler(solution.rotation_matrix)
    
    params = lmfit.Parameters()
    params.add("center_x", value=solution.center_x, vary=False)
    params.add("center_y", value=solution.center_y, vary=False)
    params.add("ai", value=ai, vary=True)
    params.add("aj", value=aj, vary=True)
    params.add("ak", value=ak, vary=True)
    params.add("scale", value=solution.scale, vary=True, min=0.8, max=1.2)
    
    wavelength = get_electron_wavelength(accelarating_voltage)
    camera_length = camera_length * 1e10
    args = k_xy, lattice_recip, wavelength, camera_length 
    
    res = lmfit.minimize(objfunc, params, args=args, method=method)
    
    if verbose:
        lmfit.report_fit(res)
            
    p = res.params

    ai, aj, ak = p["ai"].value, p["aj"].value, p["ak"].value
    scale = p["scale"].value
    center_x = params["center_x"].value
    center_y = params["center_y"].value
    
    rotation_matrix = euler2mat(ai, aj, ak)
    
    k_xy = (k_xy + np.array((center_x, center_y)) * scale)
    cart = detector_to_fourier(k_xy, wavelength=wavelength, camera_length=camera_length)
    
    intermediate = cart.dot(rotation_matrix.T) # Must use the transpose here
    hklss = lattice_recip.fractional(intermediate)

    rhklss = np.rint(hklss)
    
    error_hkls = res.residual.reshape(-1, 3)
    error_mean = np.mean(error_hkls)
    
    valid_peak_mask = np.max(error_hkls, axis=-1) < index_error_tol
    valid_peak_count = np.count_nonzero(valid_peak_mask, axis=-1)
    
    num_peaks = len(k_xy)
    
    match_rate = (valid_peak_count * (1 / num_peaks)) if num_peaks else 0
    
    orientation = OrientationResult(phase_index=solution.phase_index,
                                               rotation_matrix=rotation_matrix,
                                               match_rate=match_rate,
                                               error_hkls=error_hkls,
                                               total_error=error_mean,
                                               scale=scale,
                                               center_x=center_x,
                                               center_y=center_y)

    res = np.empty(2, dtype=np.object)
    res[0] = orientation
    res[1] = rhklss

    return res


class VectorIndexationGenerator():
    """Generates an indexer for DiffractionVectors using a number of methods.

    Attributes
    ----------
    vectors : DiffractionVectors
        DiffractionVectors to be indexed.
    vector_library : DiffractionVectorLibrary
        Library of theoretical diffraction vector magnitudes and inter-vector
        angles for indexation.

    Parameters
    ----------
    vectors : DiffractionVectors
        DiffractionVectors to be indexed.
    vector_library : DiffractionVectorLibrary
        Library of theoretical diffraction vector magnitudes and inter-vector
        angles for indexation.
    """

    def __init__(self,
                 vectors,
                 vector_library):
        if vectors.cartesian is None:
            raise ValueError("Cartesian coordinates are required in order to index "
                             "diffraction vectors. Use the calculate_cartesian_coordinates "
                             "method of DiffractionVectors to obtain these.")
        else:
            self.vectors = vectors
            self.library = vector_library

    def index_vectors(self,
                      mag_tol,
                      angle_tol,
                      index_error_tol,
                      n_peaks_to_index,
                      n_best,
                      *args,
                      **kwargs):
        """Assigns hkl indices to diffraction vectors.

        Parameters
        ----------
        mag_tol : float
            The maximum absolute error in diffraction vector magnitude, in units
            of reciprocal Angstroms, allowed for indexation.
        angle_tol : float
            The maximum absolute error in inter-vector angle, in units of
            degrees, allowed for indexation.
        index_error_tol : float
            Max allowed error in peak indexation for classifying it as indexed,
            calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
        n_peaks_to_index : int
            The maximum number of peak to index.
        n_best : int
            The maximum number of good solutions to be retained.
        *args : arguments
            Arguments passed to the map() function.
        **kwargs : arguments
            Keyword arguments passed to the map() function.

        Returns
        -------
        indexation_results : VectorMatchingResults
            Navigation axes of the diffraction vectors signal containing vector
            indexation results for each probe position.
        """
        vectors = self.vectors
        library = self.library

        matched = vectors.cartesian.map(match_vectors,
                                        library=library,
                                        mag_tol=mag_tol,
                                        angle_tol=np.deg2rad(angle_tol),
                                        index_error_tol=index_error_tol,
                                        n_peaks_to_index=n_peaks_to_index,
                                        n_best=n_best,
                                        inplace=False,
                                        *args,
                                        **kwargs)
        indexation = matched.isig[0]
        rhkls = matched.isig[1].data

        indexation_results = VectorMatchingResults(indexation)
        indexation_results.vectors = vectors
        indexation_results.hkls = rhkls
        indexation_results = transfer_navigation_axes(indexation_results,
                                                      vectors.cartesian)

        vectors.hkls = rhkls

        return indexation_results

    def refine_best_orientation(self, 
                                orientations,
                                accelarating_voltage,
                                camera_length,
                                rank=0,
                                index_error_tol=0.2,
                                method="leastsq"):
        """Refines the best orientation and assigns hkl indices to diffraction vectors.

        Parameters
        ----------
        rank : int
            The rank of the solution, i.e. rank=2 returns the third best solution
        index_error_tol : float
            Max allowed error in peak indexation for classifying it as indexed,
            calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
        method : str
            Minimization algorithm to use, choose from: 
            'leastsq', 'nelder', 'powell', 'cobyla', 'least-squares'.
            See `lmfit` documentation (https://lmfit.github.io/lmfit-py/fitting.html)
            for more information.

        Returns
        -------
        indexation_results : VectorMatchingResults
            Navigation axes of the diffraction vectors signal containing vector
            indexation results for each probe position.
        """
        vectors = self.vectors
        library = self.library

        matched = orientations.map(_refine_best_orientation,
                                        vectors=vectors,
                                        library=library,
                                        accelarating_voltage=accelarating_voltage,
                                        camera_length=camera_length,
                                        index_error_tol=index_error_tol,
                                        method=method,
                                        rank=rank,
                                        parallel=False, inplace=False)

        indexation = matched.isig[0]
        rhkls = matched.isig[1].data

        indexation_results = VectorMatchingResults(indexation)
        indexation_results.vectors = vectors
        indexation_results.hkls = rhkls
        indexation_results = transfer_navigation_axes(indexation_results,
                                                      vectors.cartesian)

        vectors.hkls = rhkls

        return indexation_results

    def refine_all_orientations(self):
        pass

