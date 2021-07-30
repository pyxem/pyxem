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

"""Indexation generator and associated tools."""

import numpy as np
import lmfit
from transforms3d.euler import mat2euler, euler2mat

from diffsims.utils.sim_utils import get_electron_wavelength

from pyxem.signals import TemplateMatchingResults, VectorMatchingResults
from pyxem.utils.indexation_utils import (
    zero_mean_normalized_correlation,
    fast_correlation,
    index_magnitudes,
    match_vectors,
    OrientationResult,
    get_nth_best_solution,
    index_dataset_with_template_rotation,
)
from pyxem.utils.vector_utils import detector_to_fourier
from pyxem.utils.signal import (
    select_method_from_method_dict,
    transfer_navigation_axes,
)

class IndexationGenerator:
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    signal : ElectronDiffraction2D
        The signal of electron diffraction patterns to be indexed.
    diffraction_library : DiffractionLibrary
        The library of simulated diffraction patterns for indexation.

    Returns
    -------
    ValueError : "use TemplateIndexationGenerator or VectorIndexationGenerator"
    """

    def __init__(self, signal, diffraction_library):
        raise ValueError("use TemplateIndexationGenerator or VectorIndexationGenerator")


def _correlate_templates(image, library, n_largest, method, mask):
    r"""Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image).

    Calculated using the normalised (see return type documentation) dot
    product, or cosine distance,

    .. math:: fast_correlation
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\\sqrt{\\sum_{j=1}^m T^2(x_j, y_j)}}

    .. math:: zero_mean_normalized_correlation
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)- avg(P)avg(T)}{\\sqrt{\\sum_{j=1}^m (T(x_j, y_j)-avg(T))^2+\sum_{j=1}^m P(x_j,y_j)-avg(P)}}
        for a template T and an experimental pattern P.

    Parameters
    ----------
    image : numpy.array
        The experimental diffraction pattern of interest.
    library : DiffractionLibrary
        The library of diffraction simulations to be correlated with the
        experimental data.
    n_largest : int
        The number of well correlated simulations to be retained per phase
    method : str
        Name of method used to compute correlation between templates and diffraction patterns. Can be
        'fast_correlation' or 'zero_mean_normalized_correlation'.
    mask : bool
        A mask for navigation axes. 1 indicates positions to be indexed.


    Returns
    -------
    top_matches : numpy.array
        Array of shape (<num phases>*n_largest, 5) containing the top n
        correlated simulations for the experimental pattern of interest, where
        each entry is on the form [phase index,alpha,beta,gamma,correlation].


    References
    ----------
    E. F. Rauch and L. Dupuy, “Rapid Diffraction Patterns identification through
       template matching,” vol. 50, no. 1, pp. 87–99, 2005.

    """
    phase_count = len(library.keys())
    top_matches = np.zeros((n_largest * phase_count, 5))

    # return for the masked data
    if mask != 1:
        return top_matches

    if method == "zero_mean_normalized_correlation":
        nb_pixels = image.shape[0] * image.shape[1]
        average_image_intensity = np.average(image)
        image_std = np.linalg.norm(image - average_image_intensity)

    for phase_number, phase in enumerate(library.keys()):
        saved_results = np.zeros((n_largest, 5))
        saved_results[:, 0] = phase_number

        for entry_number in np.arange(len(library[phase]["orientations"])):
            orientations = library[phase]["orientations"][entry_number]
            pixel_coords = library[phase]["pixel_coords"][entry_number]
            intensities = library[phase]["intensities"][entry_number]

            # Extract experimental intensities from the diffraction image
            image_intensities = image[pixel_coords[:, 1], pixel_coords[:, 0]]

            if method == "zero_mean_normalized_correlation":
                corr_local = zero_mean_normalized_correlation(
                    nb_pixels,
                    image_std,
                    average_image_intensity,
                    image_intensities,
                    intensities,
                )

            elif method == "fast_correlation":
                corr_local = fast_correlation(
                    image_intensities,
                    intensities,
                    library[phase]["pattern_norms"][entry_number],
                )

            if corr_local > np.min(saved_results[:, 4]):
                row_index = np.argmin(saved_results[:, 4])
                saved_results[row_index, 1:4] = orientations
                saved_results[row_index, 4] = corr_local

        phase_sorted = saved_results[saved_results[:, 4].argsort()]
        start_slot = phase_number * n_largest
        end_slot = (phase_number + 1) * n_largest
        top_matches[start_slot:end_slot, :] = phase_sorted
    return top_matches


class AcceleratedIndexationGenerator:
    """
    Generates a template-based indexer that also calculates relative
    rotation of templates.

    Parameters
    ----------
    signal : ElectronDiffraction2D
        The signal of electron diffraction patterns to be indexed.
    diffraction_library : DiffractionLibrary
        The library of simulated diffraction patterns for indexation.

    Notes
    -----
    To be used with minimal template libraries whereby the first euler
    angle is 0. It is this angle that is optimized during indexation.
    """
    def __init__(self,signal,diffraction_library):
        # test that the first euler angle is always 0

        for phase in diffraction_library:
            if not np.allclose(np.sum(diffraction_library[phase]["orientations"][:,0]), 0):
                raise ValueError("Invalid diffraction library! Templates must be generated from orientations where "
                        "the first Euler angle is 0")
        self.signal = signal
        self.library = diffraction_library


    def correlate(self,
                  n_largest=5,
                  include_phases=None,
                  **kwargs,
                  ):
        """
        Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : int, optional
            Number of best solutions to return, in order of descending match
        include_phases : list, optional
            Names of phases in the library to do an indexation for. By default this is
            all phases in the library.
        n_keep : int, optional
            Number of templates to do a full matching on in the second matching step
        frac_keep : float, optional
            Fraction (between 0-1) of templates to do a full matching on. When set
            n_keep will be ignored
        delta_r : float, optional
            The sampling interval of the radial coordinate in pixels
        delta_theta : float, optional
            The sampling interval of the azimuthal coordinate in degrees
        max_r : float, optional
            Maximum radius to consider in pixel units. By default it is the
            distance from the center of the image to a corner
        intensity_transform_function : Callable, optional
            Function to apply to both image and template intensities on an
            element by element basis prior to comparison
        find_direct_beam : bool, optional
            Whether to optimize the direct beam, otherwise the center of the image
            is chosen
        direct_beam_positions : 2-tuple of floats or 3D numpy array of shape (scan_x, scan_y, 2), optional
            (x, y) coordinates of the direct beam in pixel units. Overrides other
            settings for finding the direct beam
        normalize_images : bool, optional
            normalize the images to calculate the correlation coefficient
        normalize_templates : bool, optional
            normalize the templates to calculate the correlation coefficient
        parallelize_polar_conversion : bool, optional
            use multiple workers for converting the dataset to polar coordinates. Overhead
            could make this slower on some hardware and for some datasets.
        chunks : string or 4-tuple, optional
            internally the work is done on dask datasets and this parameter determines
            the chunking. If set to None then no re-chunking will happen if the dataset
            was loaded lazily. If set to "auto" then dask attempts to find the optimal
            chunk size. If None, no changes will be made to the chunking.
        parallel_workers: int, optional
            the number of workers to use in parallel. If set to "auto", the number
            will be determined from os.cpu_count()

        Returns
        -------
        result : dict

        Notes
        -----
        Internally, this code is compiled to LLVM machine code, so stack traces are often hard to follow on failure. As such it is
        important to be careful with your parameters selection.
        """
        result =  index_dataset_with_template_rotation(self.signal,
                                                    self.library,
                                                    phases=include_phases,
                                                    n_best=n_largest,
                                                    **kwargs)
        return result






class TemplateIndexationGenerator:
    """Generates an indexer for data using a number of methods.

    Parameters
    ----------
    signal : ElectronDiffraction2D
        The signal of electron diffraction patterns to be indexed.
    diffraction_library : DiffractionLibrary
        The library of simulated diffraction patterns for indexation.
    """

    def __init__(self, signal, diffraction_library):
        self.signal = signal
        self.library = diffraction_library

    def correlate(
        self,
        n_largest=5,
        method="fast_correlation",
        mask=None,
        print_help=False,
        **kwargs,
    ):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : int
            The n orientations with the highest correlation values for each phase are returned.
        method : str
            Name of method used to compute correlation between templates and diffraction patterns. Can be
            'fast_correlation' or 'zero_mean_normalized_correlation'.
        mask : hs.BaseSignal or None
            Only apply the method a unmasked (value=1) pixel, default is None (index all pixels)
        print_help : bool
            Display information about the method used.
        **kwargs : arguments
            Keyword arguments passed map().

        Returns
        -------
        matching_results : TemplateMatchingResults
        """
        signal = self.signal
        library = self.library

        method_dict = {
            "fast_correlation": fast_correlation,
            "zero_mean_normalized_correlation": zero_mean_normalized_correlation,
        }

        if mask is None:
            # Index at all real space pixels
            mask = 1

        # tests if selected method is valid and can print help for selected method.
        _ = select_method_from_method_dict(method, method_dict, print_help)

        # adds a normalisation to library #TODO: Port to diffsims
        for phase in library.keys():
            # initialise a container for the norms
            norm_array = np.ones(library[phase]["intensities"].shape[0])

            for i, intensity_array in enumerate(library[phase]["intensities"]):
                norm_array[i] = np.linalg.norm(intensity_array)
                library[phase]["pattern_norms"] = norm_array

        matches = signal.map(
            _correlate_templates,
            library=library,
            n_largest=n_largest,
            method=method,
            mask=mask,
            inplace=False,
            **kwargs,
        )

        matching_results = TemplateMatchingResults(matches)

        return matching_results


class ProfileIndexationGenerator:
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

    def index_peaks(self, tolerance=0.1, *args, **kwargs):
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


def _refine_best_orientations(
    single_match_result,
    vectors,
    library,
    accelarating_voltage,
    camera_length,
    n_best=5,
    rank=0,
    index_error_tol=0.2,
    method="leastsq",
    vary_angles=True,
    vary_center=False,
    vary_scale=False,
    verbose=False,
):
    """
    Refine a single orientation agains the given cartesian vector coordinates.

    Parameters
    ----------
    single_match_result : VectorMatchingResults
        Pool of solutions from the vector matching algorithm
    n_best : int
        Refine the best `n` orientations starting from `rank`.
        With `n_best=0` (default), all orientations are refined.
    rank : int
        The rank of the solution to start from.
    solution : list
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
    vary_angles : bool,
        Free the euler angles (rotation matrix) during the refinement.
    vary_center : bool
        Free the center of the diffraction pattern (beam center) during the refinement.
    vary_scale : bool
        Free the scale (i.e. pixel size) of the diffraction vectors during refinement.

    Returns
    -------
    result : OrientationResult
        Container for the orientation refinement results
    """
    if not isinstance(single_match_result[0], tuple):  # pragma: no cover
        single_match_result = single_match_result[0]
    n_matches = len(single_match_result)

    if n_best == 0:
        n_best = n_matches - rank

    n_best = min(n_matches, n_best)

    top_matches = np.empty(n_best, dtype="object")
    res_rhkls = []

    for i in range(rank, rank + n_best):
        if verbose:  # pragma: no cover
            print(f"# {i}/{n_best} ({n_matches})")

        solution = get_nth_best_solution(single_match_result, "vector", rank=i)

        result = _refine_orientation(
            solution,
            vectors,
            library,
            accelarating_voltage=accelarating_voltage,
            camera_length=camera_length,
            index_error_tol=index_error_tol,
            method=method,
            vary_angles=vary_angles,
            vary_center=vary_center,
            vary_scale=vary_scale,
            verbose=verbose,
        )

        top_matches[i] = result[0]
        res_rhkls.append(result[1])

    res = np.empty(2, dtype=object)
    res[0] = top_matches
    res[1] = np.asarray(res_rhkls)
    return res


def _refine_orientation(
    solution,
    k_xy,
    structure_library,
    accelarating_voltage,
    camera_length,
    index_error_tol=0.2,
    method="leastsq",
    vary_angles=True,
    vary_center=False,
    vary_scale=False,
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
    vary_angles : bool,
        Free the euler angles (rotation matrix) during the refinement.
    vary_center : bool
        Free the center of the diffraction pattern (beam center) during the refinement.
    vary_scale : bool
        Free the scale (i.e. pixel size) of the diffraction vectors during refinement.
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

        k_xy = k_xy + np.array((cx, cy)) * scale
        cart = detector_to_fourier(k_xy, wavelength, camera_length)

        intermediate = cart.dot(rotmat.T)  # Must use the transpose here
        hklss = lattice_recip.fractional(intermediate) * scale

        rhklss = np.rint(hklss)
        ehklss = np.abs(hklss - rhklss)

        return ehklss

    ai, aj, ak = mat2euler(solution.rotation_matrix)

    params = lmfit.Parameters()
    params.add("center_x", value=solution.center_x, vary=vary_center)
    params.add("center_y", value=solution.center_y, vary=vary_center)
    params.add("ai", value=ai, vary=vary_angles)
    params.add("aj", value=aj, vary=vary_angles)
    params.add("ak", value=ak, vary=vary_angles)
    params.add("scale", value=solution.scale, vary=vary_scale, min=0.8, max=1.2)

    wavelength = get_electron_wavelength(accelarating_voltage)
    camera_length = camera_length * 1e10
    args = k_xy, lattice_recip, wavelength, camera_length

    res = lmfit.minimize(objfunc, params, args=args, method=method)

    if verbose:  # pragma: no cover
        lmfit.report_fit(res)

    p = res.params

    ai, aj, ak = p["ai"].value, p["aj"].value, p["ak"].value
    scale = p["scale"].value
    center_x = params["center_x"].value
    center_y = params["center_y"].value

    rotation_matrix = euler2mat(ai, aj, ak)

    k_xy = k_xy + np.array((center_x, center_y)) * scale
    cart = detector_to_fourier(k_xy, wavelength=wavelength, camera_length=camera_length)

    intermediate = cart.dot(rotation_matrix.T)  # Must use the transpose here
    hklss = lattice_recip.fractional(intermediate)

    rhklss = np.rint(hklss)

    error_hkls = res.residual.reshape(-1, 3)
    error_mean = np.mean(error_hkls)

    valid_peak_mask = np.max(error_hkls, axis=-1) < index_error_tol
    valid_peak_count = np.count_nonzero(valid_peak_mask, axis=-1)

    num_peaks = len(k_xy)

    match_rate = (valid_peak_count * (1 / num_peaks)) if num_peaks else 0

    orientation = OrientationResult(
        phase_index=solution.phase_index,
        rotation_matrix=rotation_matrix,
        match_rate=match_rate,
        error_hkls=error_hkls,
        total_error=error_mean,
        scale=scale,
        center_x=center_x,
        center_y=center_y,
    )

    res = np.empty(2, dtype=object)
    res[0] = orientation
    res[1] = rhklss

    return res


class VectorIndexationGenerator:
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

    def __init__(self, vectors, vector_library):
        if vectors.cartesian is None:
            raise ValueError(
                "Cartesian coordinates are required in order to index "
                "diffraction vectors. Use the calculate_cartesian_coordinates "
                "method of DiffractionVectors to obtain these."
            )
        else:
            self.vectors = vectors
            self.library = vector_library

    def index_vectors(
        self,
        mag_tol,
        angle_tol,
        index_error_tol,
        n_peaks_to_index,
        n_best,
        *args,
        **kwargs,
    ):
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

        matched = vectors.cartesian.map(
            match_vectors,
            library=library,
            mag_tol=mag_tol,
            angle_tol=np.deg2rad(angle_tol),
            index_error_tol=index_error_tol,
            n_peaks_to_index=n_peaks_to_index,
            n_best=n_best,
            inplace=False,
            *args,
            **kwargs,
        )
        indexation = matched.isig[0]
        rhkls = matched.isig[1].data

        indexation_results = VectorMatchingResults(indexation)
        indexation_results.vectors = vectors
        indexation_results.hkls = rhkls
        indexation_results = transfer_navigation_axes(
            indexation_results, vectors.cartesian
        )

        vectors.hkls = rhkls

        return indexation_results

    def refine_best_orientation(
        self,
        orientations,
        accelarating_voltage,
        camera_length,
        rank=0,
        index_error_tol=0.2,
        vary_angles=True,
        vary_center=False,
        vary_scale=False,
        method="leastsq",
    ):
        """Refines the best orientation and assigns hkl indices to diffraction vectors.

        Parameters
        ----------
        rank : int
            The rank of the solution, i.e. rank=2 refines the third best solution
        index_error_tol : float
            Max allowed error in peak indexation for classifying it as indexed,
            calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
        method : str
            Minimization algorithm to use, choose from:
            'leastsq', 'nelder', 'powell', 'cobyla', 'least-squares'.
            See `lmfit` documentation (https://lmfit.github.io/lmfit-py/fitting.html)
            for more information.
        vary_angles : bool,
            Free the euler angles (rotation matrix) during the refinement.
        vary_center : bool
            Free the center of the diffraction pattern (beam center) during the refinement.
        vary_scale : bool
            Free the scale (i.e. pixel size) of the diffraction vectors during refinement.

        Returns
        -------
        indexation_results : VectorMatchingResults
            Navigation axes of the diffraction vectors signal containing vector
            indexation results for each probe position.
        """

        return self.refine_n_best_orientations(
            orientations,
            accelarating_voltage=accelarating_voltage,
            camera_length=camera_length,
            n_best=1,
            rank=rank,
            index_error_tol=index_error_tol,
            method=method,
            vary_angles=vary_angles,
            vary_center=vary_center,
            vary_scale=vary_scale,
        )

    def refine_n_best_orientations(
        self,
        orientations,
        accelarating_voltage,
        camera_length,
        n_best=0,
        rank=0,
        index_error_tol=0.2,
        vary_angles=True,
        vary_center=False,
        vary_scale=False,
        method="leastsq",
    ):
        """Refines the best orientation and assigns hkl indices to diffraction vectors.

        Parameters
        ----------
        orientations : VectorMatchingResults
            List of orientations to refine, must be an instance of `VectorMatchingResults`.
        accelerating_voltage : float
            The acceleration voltage with which the data was acquired.
        camera_length : float
            The camera length in meters.
        n_best : int
            Refine the best `n` orientations starting from `rank`.
            With `n_best=0` (default), all orientations are refined.
        rank : int
            The rank of the solution to start from.
        index_error_tol : float
            Max allowed error in peak indexation for classifying it as indexed,
            calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
        method : str
            Minimization algorithm to use, choose from:
            'leastsq', 'nelder', 'powell', 'cobyla', 'least-squares'.
            See `lmfit` documentation (https://lmfit.github.io/lmfit-py/fitting.html)
            for more information.
        vary_angles : bool,
            Free the euler angles (rotation matrix) during the refinement.
        vary_center : bool
            Free the center of the diffraction pattern (beam center) during the refinement.
        vary_scale : bool
            Free the scale (i.e. pixel size) of the diffraction vectors during refinement.

        Returns
        -------
        indexation_results : VectorMatchingResults
            Navigation axes of the diffraction vectors signal containing vector
            indexation results for each probe position.
        """
        vectors = self.vectors
        library = self.library

        matched = orientations.map(
            _refine_best_orientations,
            vectors=vectors,
            library=library,
            accelarating_voltage=accelarating_voltage,
            camera_length=camera_length,
            n_best=n_best,
            rank=rank,
            method="leastsq",
            verbose=False,
            vary_angles=vary_angles,
            vary_center=vary_center,
            vary_scale=vary_scale,
            inplace=False,
            parallel=False,
        )

        indexation = matched.isig[0]
        rhkls = matched.isig[1].data

        indexation_results = VectorMatchingResults(indexation)
        indexation_results.vectors = vectors
        indexation_results.hkls = rhkls
        indexation_results = transfer_navigation_axes(
            indexation_results, vectors.cartesian
        )

        return indexation_results
