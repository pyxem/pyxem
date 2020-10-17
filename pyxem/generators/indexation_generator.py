# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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

from pyxem.signals.indexation_results import TemplateMatchingResults
from pyxem.signals.indexation_results import VectorMatchingResults

from pyxem.signals import transfer_navigation_axes
from pyxem.signals import select_method_from_method_dict

from pyxem.utils.indexation_utils import (
    correlate_library,
    zero_mean_normalized_correlation,
    fast_correlation,
    full_frame_correlation,
    index_magnitudes,
    match_vectors,
    OrientationResult,
    get_nth_best_solution,
    correlate_library_from_dict,
    optimal_fft_size,
)


from transforms3d.euler import mat2euler, euler2mat
from pyxem.utils.vector_utils import detector_to_fourier
from diffsims.utils.sim_utils import get_electron_wavelength

import lmfit


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
    ValueError : "use TemplateIndexationGenerator or PatternIndexationGenerator"
    """

    def __init__(self, signal, diffraction_library):
        self.signal = signal
        self.library = diffraction_library
        raise ValueError("use TemplateIndexationGenerator or PatternIndexationGenerator")


def _correlate_patterns(image, template_dict, n_largest, method, mask):
    """Correlates all simulated diffraction templates in a DiffractionLibrary
    with a particular experimental diffraction pattern (image).

    Parameters
    ----------
    image : numpy.array
        The experimental diffraction pattern of interest.
    template_dict : dict
        Dictionary containing orientations, fourier transform of templates and template norms for
        every phase.
    n_largest : int
        The number of well correlated simulations to be retained.
    method : str
        Name of method used to compute correlation between templates and diffraction patterns. Can be
         'full_frame_correlation'. (I believe angular decomposition can also fit this framework)
    mask : bool
        A mask for navigation axes. 1 indicates positions to be indexed.


    Returns
    -------
    top_matches : numpy.array
        Array of shape (<num phases>*n_largest, 3) containing the top n
        correlated simulations for the experimental pattern of interest, where
        each entry is on the form [phase index, [z, x, z], correlation].


    References
    ----------
    full_frame_correlation:
    A. Foden, D. M. Collins, A. J. Wilkinson and T. B. Britton "Indexing electron backscatter diffraction patterns with
     a refined template matching approach" doi: https://doi.org/10.1016/j.ultramic.2019.112845
    """

    top_matches = np.empty((len(template_dict), n_largest, 3), dtype="object")

    if method == "full_frame_correlation":
        size = 2 * np.array(image.shape) - 1
        fsize = [optimal_fft_size(a, real=True) for a in (size)]
        image_FT = np.fft.fftshift(np.fft.rfftn(image, fsize))
        image_norm = np.sqrt(full_frame_correlation(image_FT, 1, image_FT, 1))

    if mask == 1:
        for phase_index, library_entry in enumerate(template_dict.values()):
            orientations = library_entry["orientations"]
            patterns = library_entry["patterns"]
            pattern_norms = library_entry["pattern_norms"]

            zip_for_locals = zip(orientations, patterns, pattern_norms)

            or_saved, corr_saved = np.empty((n_largest, 3)), np.zeros((n_largest, 1))

            for (or_local, pat_local, pn_local) in zip_for_locals:

                if method == "full_frame_correlation":
                    corr_local = full_frame_correlation(
                        image_FT, image_norm, pat_local, pn_local
                    )

                if corr_local > np.min(corr_saved):
                    or_saved[np.argmin(corr_saved)] = or_local
                    corr_saved[np.argmin(corr_saved)] = corr_local

                combined_array = np.hstack((or_saved, corr_saved))
                combined_array = combined_array[
                    np.flip(combined_array[:, 3].argsort())
                ]  # see stackoverflow/2828059 for details
                top_matches[phase_index, :, 0] = phase_index
                top_matches[phase_index, :, 2] = combined_array[:, 3]  # correlation
                for i in np.arange(n_largest):
                    top_matches[phase_index, i, 1] = combined_array[
                        i, :3
                    ]  # orientation

    return top_matches.reshape(-1, 3)


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
        The number of well correlated simulations to be retained.
    method : str
        Name of method used to compute correlation between templates and diffraction patterns. Can be
        'fast_correlation', 'full_frame_correlation' or 'zero_mean_normalized_correlation'. (ADDED in pyxem 0.11.0)
    mask : bool
        A mask for navigation axes. 1 indicates positions to be indexed.


    Returns
    -------
    top_matches : numpy.array
        Array of shape (<num phases>*n_largest, 3) containing the top n
        correlated simulations for the experimental pattern of interest, where
        each entry is on the form [phase index, [z, x, z], correlation].

    See Also
    --------
    IndexationGenerator.correlate

    Notes
    -----
    Correlation results are defined as,
        phase_index : int
            Index of the phase, following the ordering of the library keys
        [z, x, z] : ndarray
            numpy array of three floats, specifying the orientation in the
            Bunge convention, in degrees.
        correlation : float
            A coefficient of correlation, only normalised to the template
            intensity. This is in contrast to the reference work.

    References
    ----------
    E. F. Rauch and L. Dupuy, “Rapid Diffraction Patterns identification through
       template matching,” vol. 50, no. 1, pp. 87–99, 2005.

    A. Nakhmani and  A. Tannenbaum, "A New Distance Measure Based on Generalized Image Normalized Cross-Correlation
    for Robust Video Tracking and Image Recognition"
    Pattern Recognit Lett. 2013 Feb 1; 34(3): 315–321; doi: 10.1016/j.patrec.2012.10.025

    Discussion on Normalized cross correlation (xcdskd):
    https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html

    """

    top_matches = np.empty((len(library), n_largest, 3), dtype="object")

    if method == "zero_mean_normalized_correlation":
        nb_pixels = image.shape[0] * image.shape[1]
        average_image_intensity = np.average(image)
        image_std = np.linalg.norm(image - average_image_intensity)

    if mask == 1:
        for phase_index, library_entry in enumerate(library.values()):
            orientations = library_entry["orientations"]
            pixel_coords = library_entry["pixel_coords"]
            intensities = library_entry["intensities"]
            # TODO: This is only applicable some of the time, probably use an if + special_local in the for
            pattern_norms = library_entry["pattern_norms"]

            zip_for_locals = zip(orientations, pixel_coords, intensities, pattern_norms)

            or_saved, corr_saved = np.empty((n_largest, 3)), np.zeros((n_largest, 1))

            for (or_local, px_local, int_local, pn_local) in zip_for_locals:
                # TODO: Factorise out the generation of corr_local to a method='mthd' section
                # Extract experimental intensities from the diffraction image
                image_intensities = image[
                    px_local[:, 1], px_local[:, 0]
                ]  # Counter intuitive indexing? Why is it not px_local[:, 0], px_local[:, 1]?

                if method == "zero_mean_normalized_correlation":
                    corr_local = zero_mean_normalized_correlation(
                        nb_pixels,
                        image_std,
                        average_image_intensity,
                        image_intensities,
                        int_local,
                    )

                elif method == "fast_correlation":
                    corr_local = fast_correlation(
                        image_intensities, int_local, pn_local
                    )

                if corr_local > np.min(corr_saved):
                    or_saved[np.argmin(corr_saved)] = or_local
                    corr_saved[np.argmin(corr_saved)] = corr_local

                combined_array = np.hstack((or_saved, corr_saved))
                combined_array = combined_array[
                    np.flip(combined_array[:, 3].argsort())
                ]  # see stackoverflow/2828059 for details
                top_matches[phase_index, :, 0] = phase_index
                top_matches[phase_index, :, 2] = combined_array[:, 3]  # correlation
                for i in np.arange(n_largest):
                    top_matches[phase_index, i, 1] = combined_array[
                        i, :3
                    ]  # orientation

    return top_matches.reshape(-1, 3)


def get_fourier_transform(template_coordinates, template_intensities, shape, fsize):
    """Returns the Fourier transform of a list of templates.

    Takes a list of template coordinates and the corresponding list of
    template intensities, and returns the Fourier transform of the template.

    Parameters
    ----------
    template_coordinates: numpy array
        Array containing coordinates for non-zero intensities in the template
    template_intensities: list
        List of intensity values for the template.
    shape: tuple
        Dimensions of the signal.
    fsize: list
        Dimensions of the Fourier transformed signal.

    Returns
    -------
    template_FT: numpy array
        Fourier transform of the template.
    template_norm: float
        Self correlation value for the template.
    """
    template = np.zeros((shape))
    template[
        template_coordinates[:, 1], template_coordinates[:, 0]
    ] = template_intensities[:]
    template_FT = np.fft.fftshift(np.fft.rfftn(template, fsize))
    template_norm = np.sqrt(full_frame_correlation(template_FT, 1, template_FT, 1))
    return template_FT, template_norm


def get_library_FT_dict(template_library, shape, fsize):
    """Takes a template library and converts it to a dictionary of Fourier transformed templates.

    Parameters
    ----------
    template_library: DiffractionLibrary
        The library of simulated diffraction patterns for indexation.
    shape: tuple
        Dimensions of the signal.
    fsize: list
        Dimensions of the Fourier transformed signal.

    Returns
    -------
    library_FT_dict: dict
        Dictionary containing the fourier transformed template library, together with the corresponding orientations and
        pattern norms.

    """
    library_FT_dict = {}
    for entry, library_entry in enumerate(template_library.values()):
        orientations = library_entry["orientations"]
        pixel_coords = library_entry["pixel_coords"]
        intensities = library_entry["intensities"]
        template_FTs = []
        pattern_norms = []
        for coord, intensity in zip(pixel_coords, intensities):
            template_FT, pattern_norm = get_fourier_transform(
                coord, intensity, shape, fsize
            )
            template_FTs.append(template_FT)
            pattern_norms.append(pattern_norm)

        library_FT_dict[entry] = {
            "orientations": orientations,
            "patterns": template_FTs,
            "pattern_norms": pattern_norms,
        }

    return library_FT_dict




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
        *args,
        **kwargs,
    ):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : int
            The n orientations with the highest correlation values are returned.
        method : str
            Name of method used to compute correlation between templates and diffraction patterns. Can be
            'fast_correlation' or 'zero_mean_normalized_correlation'.
        mask : Array
            Array with the same size as signal (in navigation) or None
        print_help : bool
            Display information about the method used.
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

        method_dict = {
            "fast_correlation": fast_correlation,
            "zero_mean_normalized_correlation": zero_mean_normalized_correlation,
        }

        if mask is None:
            # Index at all real space pixels
            mask = 1

        # tests if selected method is a valid argument, and can print help for selected method.
        chosen_function = select_method_from_method_dict(
            method, method_dict, print_help
        )
        if method in ["fast_correlation", "zero_mean_normalized_correlation"]:
            # adds a normalisation to library
            for phase in library.keys():
                norm_array = np.ones(
                    library[phase]["intensities"].shape[0]
                )  # will store the norms

                for i, intensity_array in enumerate(library[phase]["intensities"]):
                    norm_array[i] = np.linalg.norm(intensity_array)
                library[phase][
                    "pattern_norms"
                ] = norm_array  # puts this normalisation into the library

            matches = signal.map(
                correlate_library,
                library=library,
                n_largest=n_largest,
                method=method,
                mask=mask,
                inplace=False,
                **kwargs,
            )

        matching_results = TemplateMatchingResults(matches)
        matching_results = transfer_navigation_axes(matching_results, signal)

        return matching_results


class PatternIndexationGenerator:
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
        *args,
        **kwargs,
    ):
        """Correlates the library of simulated diffraction patterns with the
        electron diffraction signal.

        Parameters
        ----------
        n_largest : int
            The n orientations with the highest correlation values are returned.
        method : str
            Name of method used to compute correlation between templates and diffraction patterns. Can be
            'fast_correlation', 'full_frame_correlation' or 'zero_mean_normalized_correlation'.
        mask : Array
            Array with the same size as signal (in navigation) or None
        print_help : bool
            Display information about the method used.
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

        method_dict = {
            "fast_correlation": fast_correlation,
            "zero_mean_normalized_correlation": zero_mean_normalized_correlation,
            "full_frame_correlation": full_frame_correlation,
        }

        if mask is None:
            # Index at all real space pixels
            mask = 1

        # tests if selected method is a valid argument, and can print help for selected method.
        chosen_function = select_method_from_method_dict(
            method, method_dict, print_help
        )

        if method in ["full_frame_correlation"]:
            shape = signal.data.shape[-2:]
            size = 2 * np.array(shape) - 1
            fsize = [optimal_fft_size(a, real=True) for a in (size)]
            if not (np.asarray(size) + 1 == np.asarray(fsize)).all():
                raise ValueError(
                    "Please select input signal and templates of dimensions 2**n X 2**n"
                )

            library_FT_dict = get_library_FT_dict(library, shape, fsize)

            matches = signal.map(
                correlate_library_from_dict,
                template_dict=library_FT_dict,
                n_largest=n_largest,
                method=method,
                mask=mask,
                inplace=False,
                **kwargs,
            )

        matching_results = TemplateMatchingResults(matches)
        matching_results = transfer_navigation_axes(matching_results, signal)

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

    res = np.empty(2, dtype=np.object)
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

    res = np.empty(2, dtype=np.object)
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
        vectors = self.vectors
        library = self.library

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
