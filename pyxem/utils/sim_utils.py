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

import math

import numpy as np
from scipy.constants import h, m_e, e, c, pi
import collections
import diffpy.structure

from .atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
from .lobato_scattering_params import ATOMIC_SCATTERING_PARAMS_LOBATO
from pyxem.utils.vector_utils import get_angle_cartesian
from transforms3d.axangles import axangle2mat
from transforms3d.euler import mat2euler
from transforms3d.euler import euler2mat


def get_electron_wavelength(accelerating_voltage):
    """Calculates the (relativistic) electron wavelength in Angstroms for a
    given accelerating voltage in kV.

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage in kV.

    Returns
    -------
    wavelength : float
        The relativistic electron wavelength in Angstroms.

    """
    E = accelerating_voltage * 1e3
    wavelength = h / math.sqrt(2 * m_e * e * E *
                               (1 + (e / (2 * m_e * c * c)) * E)) * 1e10
    return wavelength


def get_interaction_constant(accelerating_voltage):
    """Calculates the interaction constant, sigma, for a given
    acelerating voltage.

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage in V.

    Returns
    -------
    sigma : float
        The relativistic electron wavelength in m.

    """
    E = accelerating_voltage
    wavelength = get_electron_wavelength(accelerating_voltage)
    sigma = (2 * pi * (m_e + e * E))

    return sigma


def get_unique_families(hkls):
    """Returns unique families of Miller indices, which must be permutations of
    each other.

    Parameters
    ----------
    hkls : list
        List of Miller indices ([h, k, l])

    Returns
    -------
    pretty_unique : dictionary
        A dict with unique hkl and multiplicity {hkl: multiplicity}.
    """
    def is_perm(hkl1, hkl2):
        h1 = np.abs(hkl1)
        h2 = np.abs(hkl2)
        return all([i == j for i, j in zip(sorted(h1), sorted(h2))])

    unique = collections.defaultdict(list)
    for hkl1 in hkls:
        found = False
        for hkl2 in unique.keys():
            if is_perm(hkl1, hkl2):
                found = True
                unique[hkl2].append(hkl1)
                break
        if not found:
            unique[hkl1].append(hkl1)

    pretty_unique = {}
    for k, v in unique.items():
        pretty_unique[sorted(v)[-1]] = len(v)

    return pretty_unique


def get_scattering_params_dict(scattering_params):
    """Get scattering parameter dictionary from name.

    Parameters
    ----------
    scattering_params : string
        Name of scattering factors. One of 'lobato', 'xtables'.

    Returns
    -------
    scattering_params_dict : dict
        Dictionary of scattering parameters mapping from element name.
    """
    if scattering_params == 'lobato':
        scattering_params_dict = ATOMIC_SCATTERING_PARAMS_LOBATO
    elif scattering_params == 'xtables':
        scattering_params_dict = ATOMIC_SCATTERING_PARAMS
    else:
        raise NotImplementedError("The scattering parameters `{}` are not implemented. "
                                  "See documentation for available "
                                  "implementations.".format(scattering_params))
    return scattering_params_dict


def get_vectorized_list_for_atomic_scattering_factors(structure,
                                                      debye_waller_factors,
                                                      scattering_params):
    """ Create a flattened array of coeffs, fcoords and occus for vectorized
    computation of atomic scattering factors.

    Note: The dimensions of the returned objects are not necessarily the same
    size as the number of atoms in the structure as each partially occupied
    specie occupies its own position in the flattened array.


    Parameters
    ----------
    structure : diffpy.structure
        The atomic structure for which scattering factors are required.
    debye_waller_factors : list
        List of Debye-Waller factors for atoms in structure.

    Returns
    -------
    coeffs : np.array()
        Coefficients of atomic scattering factor parameterization for each atom.
    fcoords : np.array()
        Fractional coordinates of each atom in structure.
    occus : np.array()
        Occupancy of each atomic site.
    dwfactors : np.array()
        Debye-Waller factors for each atom in the structure.
    """

    scattering_params_dict = get_scattering_params_dict(scattering_params)

    n_structures = len(structure)
    coeffs = np.empty((n_structures, 5, 2))
    fcoords = np.empty((n_structures, 3))
    occus = np.empty(n_structures)
    dwfactors = np.empty(n_structures)

    for i, site in enumerate(structure):
        coeffs[i] = scattering_params_dict[site.element]
        dwfactors[i] = debye_waller_factors.get(site.element, 0)
        fcoords[i] = site.xyz
        occus[i] = site.occupancy

    return coeffs, fcoords, occus, dwfactors


def get_atomic_scattering_factors(g_hkl_sq, coeffs, scattering_params):
    """Calculate atomic scattering factors for n atoms.

    Parameters
    ----------
    g_hkl_sq : ndarray
        One-dimensional array of g-vector lengths squared.
    coeffs : ndarray
        Three-dimensional array [n, 5, 2] of coefficients corresponding to the n atoms.
    scattering_params : string
        Type of scattering factor calculation to use. One of 'lobato', 'xtables'.

    Returns
    -------
    scattering_factors : ndarray
        The calculated atomic scattering parameters.
    """
    g_sq_coeff_1 = np.outer(g_hkl_sq, coeffs[:, :, 1]).reshape(g_hkl_sq.shape + coeffs[:, :, 1].shape)
    if scattering_params == 'lobato':
        f = (2 + g_sq_coeff_1) * (1 / np.square(1 + g_sq_coeff_1))
    elif scattering_params == 'xtables':
        f = np.exp(-0.25 * g_sq_coeff_1)
    return np.sum(coeffs[:, :, 0] * f, axis=-1)


def get_kinematical_intensities(structure,
                                g_indices,
                                g_hkls,
                                excitation_error,
                                maximum_excitation_error,
                                debye_waller_factors,
                                scattering_params='lobato'):
    """Calculates peak intensities.

    The peak intensity is a combination of the structure factor for a given
    peak and the position the Ewald sphere intersects the relrod. In this
    implementation, the intensity scales linearly with proximity.

    Parameters
    ----------
    structure : Structure
        The structure for which to derive the structure factors.
    indices : array-like
        The fractional coordinates of the peaks for which to calculate the
        structure factor.
    proximities : array-like
        The distances between the Ewald sphere and the peak centers.

    Returns
    -------
    peak_intensities : array-like
        The intensities of the peaks.

    """
    coeffs, fcoords, occus, dwfactors = get_vectorized_list_for_atomic_scattering_factors(
        structure=structure, debye_waller_factors=debye_waller_factors,
        scattering_params=scattering_params)

    # Store array of g_hkls^2 values since used multiple times.
    g_hkls_sq = g_hkls**2

    # Create array containing atomic scattering factors.
    fs = get_atomic_scattering_factors(g_hkls_sq, coeffs, scattering_params)

    # Change the coordinate system of fcoords to align with that of g_indices
    fcoords = np.dot(fcoords, np.linalg.inv(np.dot(structure.lattice.stdbase,
                                                   structure.lattice.recbase)))

    # Calculate structure factors for all excited g-vectors.
    f_hkls = np.sum(fs * occus * np.exp(
        2j * np.pi * np.dot(g_indices, fcoords.T) -
        0.25 * np.outer(g_hkls_sq, dwfactors)),
        axis=-1)

    # Define an intensity scaling that is linear with distance from Ewald sphere
    # along the beam direction.
    shape_factor = 1 - (excitation_error / maximum_excitation_error)

    # Calculate the peak intensities from the structure factor and excitation
    # error.
    peak_intensities = (f_hkls * f_hkls.conjugate()).real * shape_factor
    return peak_intensities


def simulate_kinematic_scattering(atomic_coordinates,
                                  element,
                                  accelerating_voltage,
                                  simulation_size=256,
                                  max_k=1.5,
                                  illumination='plane_wave',
                                  sigma=20,
                                  scattering_params='lobato'):
    """Simulate electron scattering from arrangement of atoms comprising one
    elemental species.

    Parameters
    ----------
    atomic_coordinates : array
        Array specifying atomic coordinates in structure.
    element : string
        Element symbol, e.g. "C".
    accelerating_voltage : float
        Accelerating voltage in keV.
    simulation_size : int
        Simulation size, n, specifies the n x n array size for
        the simulation calculation.
    max_k : float
        Maximum scattering vector magnitude in reciprocal angstroms.
    illumination : string
        Either 'plane_wave' or 'gaussian_probe' illumination
    sigma : float
        Gaussian probe standard deviation, used when illumination == 'gaussian_probe'
    scattering_params : string
        Type of scattering factor calculation to use. One of 'lobato', 'xtables'.

    Returns
    -------
    simulation : ElectronDiffraction
        ElectronDiffraction simulation.
    """
    # Delayed loading to prevent circular dependencies.
    from pyxem.signals.electron_diffraction import ElectronDiffraction

    # Get atomic scattering parameters for specified element.
    coeffs = np.array(get_scattering_params_dict(scattering_params)[element])

    # Calculate electron wavelength for given keV.
    wavelength = get_electron_wavelength(accelerating_voltage)

    # Define a 2D array of k-vectors at which to evaluate scattering.
    l = np.linspace(-max_k, max_k, simulation_size)
    kx, ky = np.meshgrid(l, l)

    # Convert 2D k-vectors into 3D k-vectors accounting for Ewald sphere.
    k = np.array((kx, ky, (wavelength / 2) * (kx ** 2 + ky ** 2)))

    # Calculate scattering vector squared for each k-vector.
    gs_sq = np.linalg.norm(k, axis=0)**2

    # Get the scattering factors for this element.
    fs = get_atomic_scattering_factors(gs_sq, coeffs[np.newaxis, :], scattering_params)

    # Evaluate scattering from all atoms
    scattering = np.zeros_like(gs_sq)
    if illumination == 'plane_wave':
        for r in atomic_coordinates:
            scattering = scattering + (fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    elif illumination == 'gaussian_probe':
        for r in atomic_coordinates:
            probe = (1 / (np.sqrt(2 * np.pi) * sigma)) * \
                np.exp((-np.abs(((r[0]**2) - (r[1]**2)))) / (4 * sigma**2))
            scattering = scattering + (probe * fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    else:
        raise ValueError("User specified illumination '{}' not defined.".format(illumination))

    # Calculate intensity
    intensity = (scattering * scattering.conjugate()).real

    return ElectronDiffraction(intensity)


def simulate_rotated_structure(diffraction_generator, structure, rotation_matrix, reciprocal_radius, with_direct_beam):
    """Calculate electron diffraction data for a structure after rotating it.

    Parameters
    ----------
    diffraction_generator : DiffractionGenerator
        Diffraction generator used to simulate diffraction patterns
    structure : diffpy.structure.Structure
        Structure object to simulate
    rotation_matrix : ndarray
        3x3 matrix describing the base rotation to apply to the structure
    reciprocal_radius : float
        The maximum g-vector magnitude to be included in the simulations.
    with_direct_beam : bool
        Include the direct beam peak
    """
    lattice_rotated = diffpy.structure.lattice.Lattice(
        *structure.lattice.abcABG(),
        baserot=rotation_matrix)
    # Don't change the original
    structure_rotated = diffpy.structure.Structure(structure)
    structure_rotated.placeInLattice(lattice_rotated)

    return diffraction_generator.calculate_ed_data(
        structure_rotated,
        reciprocal_radius,
        with_direct_beam)


def peaks_from_best_template(single_match_result, library):
    """ Takes a TemplateMatchingResults object and return the associated peaks,
    to be used in combination with map().

    Parameters
    ----------
    single_match_result : ndarray
        An entry in a TemplateMatchingResults.
    library : DiffractionLibrary
        Diffraction library containing the phases and rotations.

    Returns
    -------
    peaks : array
        Coordinates of peaks in the matching results object in calibrated units.
    """
    best_fit = single_match_result[np.argmax(single_match_result[:, 2])]
    phase_names = list(library.keys())
    best_index = int(best_fit[0])
    phase = phase_names[best_index]
    try:
        simulation = library.get_library_entry(
            phase=phase,
            angle=tuple(best_fit[1]))['Sim']
    except ValueError:
        structure = library.structures[best_index]
        rotation_matrix = euler2mat(*np.deg2rad(best_fit[1]), 'rzxz')
        simulation = simulate_rotated_structure(
            library.diffraction_generator,
            structure,
            rotation_matrix,
            library.reciprocal_radius,
            library.with_direct_beam)

    peaks = simulation.coordinates[:, :2]  # cut z
    return peaks


def peaks_from_best_vector_match(single_match_result, library):
    """ Takes a VectorMatchingResults object and return the associated peaks,
    to be used in combination with map().

    Parameters
    ----------
    single_match_result : ndarray
        An entry in a VectorMatchingResults
    library : DiffractionLibrary
        Diffraction library containing the phases and rotations

    Returns
    -------
    peaks : ndarray
        Coordinates of peaks in the matching results object in calibrated units.
    """
    best_fit = single_match_result[np.argmax(single_match_result[:, 2])]
    best_index = best_fit[0]

    rotation_matrix = best_fit[1].T
    # Don't change the original
    structure = library.structures[best_index]
    sim = simulate_rotated_structure(
        library.diffraction_generator,
        structure,
        rotation_matrix,
        library.reciprocal_radius,
        with_direct_beam=False)

    # Cut z
    return sim.coordinates[:, :2]


def get_points_in_sphere(reciprocal_lattice, reciprocal_radius):
    """Finds all reciprocal lattice points inside a given reciprocal sphere.
    Utilised within the DiffractionGenerator.

    Parameters
    ----------
    reciprocal_lattice : diffpy.Structure.Lattice
        The crystal lattice for the structure of interest.
        TODO: Mention that it is the reciprocal lattice. Just take the structure and calculate from there?
    reciprocal_radius  : float
        The radius of the sphere in reciprocal space (units of reciprocal
        Angstroms) within which reciprocal lattice points are returned.

    Returns
    -------
    spot_indicies : numpy.array
        Miller indices of reciprocal lattice points in sphere.
    spot_coords : numpy.array
        Cartesian coordinates of reciprocal lattice points in sphere.
    spot_distances : numpy.array
        Distance of reciprocal lattice points in sphere from the origin.
    """
    a, b, c = reciprocal_lattice.a, reciprocal_lattice.b, reciprocal_lattice.c
    h_max = np.ceil(reciprocal_radius / a)
    k_max = np.ceil(reciprocal_radius / b)
    l_max = np.ceil(reciprocal_radius / c)
    from itertools import product
    h_list = np.arange(-h_max, h_max + 1)
    k_list = np.arange(-k_max, k_max + 1)
    l_list = np.arange(-l_max, l_max + 1)
    potential_points = np.asarray(list(product(h_list, k_list, l_list)))
    in_sphere = np.abs(reciprocal_lattice.dist(potential_points, [0, 0, 0])) < reciprocal_radius
    spot_indicies = potential_points[in_sphere]
    spot_coords = reciprocal_lattice.cartesian(spot_indicies)
    spot_distances = reciprocal_lattice.dist(spot_indicies, [0, 0, 0])

    return spot_indicies, spot_coords, spot_distances


def is_lattice_hexagonal(latt):
    """Determines if a diffpy lattice is hexagonal or trigonal.

    Parameters
    ----------
    latt : diffpy.Structure.lattice
        The diffpy lattice object to be determined as hexagonal or not.

    Returns
    -------
    is_true : bool
        True if hexagonal or trigonal.
    """
    truth_list = []
    truth_list.append(latt.a == latt.b)
    truth_list.append(latt.alpha == 90)
    truth_list.append(latt.beta == 90)
    truth_list.append(latt.gamma == 120)
    return len(truth_list) == np.sum(truth_list)


def transfer_navigation_axes(new_signal, old_signal):
    """ Transfers navigation axis calibrations from an old signal to a new
    signal produced from it by a method or a generator.

    Parameters
    ----------
    new_signal : Signal
        The product signal with undefined navigation axes.
    old_signal : Signal
        The parent signal with calibrated navigation axes.

    Returns
    -------
    new_signal : Signal
        The new signal with calibrated navigation axes.
    """
    new_signal.axes_manager.set_signal_dimension(
        len(new_signal.data.shape) - old_signal.axes_manager.navigation_dimension)

    for i in range(min(new_signal.axes_manager.navigation_dimension,
                       old_signal.axes_manager.navigation_dimension)):
        ax_new = new_signal.axes_manager.navigation_axes[i]
        ax_old = old_signal.axes_manager.navigation_axes[i]
        ax_new.name = ax_old.name
        ax_new.scale = ax_old.scale
        ax_new.units = ax_old.units

    return new_signal


def transfer_navigation_axes_to_signal_axes(new_signal, old_signal):
    """ Transfers navigation axis calibrations from an old signal to the signal
    axes of a new signal produced from it by a method or a generator.

    Used from methods that generate a signal with a single value at each
    navigation position.

    Parameters
    ----------
    new_signal : Signal
        The product signal with undefined navigation axes.
    old_signal : Signal
        The parent signal with calibrated navigation axes.

    Returns
    -------
    new_signal : Signal
        The new signal with calibrated signal axes.
    """
    for i in range(min(new_signal.axes_manager.signal_dimension,
                       old_signal.axes_manager.navigation_dimension)):
        ax_new = new_signal.axes_manager.signal_axes[i]
        ax_old = old_signal.axes_manager.navigation_axes[i]
        ax_new.name = ax_old.name
        ax_new.scale = ax_old.scale
        ax_new.units = ax_old.units

    return new_signal


def uvtw_to_uvw(uvtw):
    """Convert 4-index direction to a 3-index direction.

    Parameters
    ----------
    uvtw : array-like with 4 floats

    Returns
    -------
    uvw : tuple of 4 floats
    """
    u, v, t, w = uvtw
    u, v, w = 2 * u + v, 2 * v + u, w
    common_factor = math.gcd(math.gcd(u, v), w)
    return tuple((int(x / common_factor)) for x in (u, v, w))


def rotation_list_stereographic(structure, corner_a, corner_b, corner_c,
                                inplane_rotations, resolution):
    """Generate a rotation list covering the inverse pole figure specified by
    three corners in cartesian coordinates.

    Parameters
    ----------
    structure : diffpy.structure.Structure
        Structure for which to calculate the rotation list.
    corner_a, corner_b, corner_c : tuple
        The three corners of the inverse pole figure, each given by a
        three-dimensional coordinate. The coordinate system is given by the
        structure lattice.
    resolution : float
        Angular resolution in radians of the generated rotation list.
    inplane_rotations : list
        List of angles in radians for in-plane rotation of the diffraction
        pattern. This corresponds to the third Euler angle rotation. The
        rotation list will be generated for each of these angles, and combined.
        This should be done automatically, but by including all possible
        rotations in the rotation list, it becomes too large.

        To cover all inplane rotations, use e.g. np.linspace(0, 2*np.pi, 360).

    Returns
    -------
    rotation_list : numpy.array
        Rotations covering the inverse pole figure given as an array of Euler
        angles in degrees.
    """
    # Convert the crystal directions to cartesian vectors and normalize
    if len(corner_a) == 4:
        corner_a = uvtw_to_uvw(corner_a)
    if len(corner_b) == 4:
        corner_b = uvtw_to_uvw(corner_b)
    if len(corner_c) == 4:
        corner_c = uvtw_to_uvw(corner_c)

    lattice = structure.lattice

    corner_a = np.dot(corner_a, lattice.stdbase)
    corner_b = np.dot(corner_b, lattice.stdbase)
    corner_c = np.dot(corner_c, lattice.stdbase)

    corner_a /= np.linalg.norm(corner_a)
    corner_b /= np.linalg.norm(corner_b)
    corner_c /= np.linalg.norm(corner_c)

    angle_a_to_b = get_angle_cartesian(corner_a, corner_b)
    angle_a_to_c = get_angle_cartesian(corner_a, corner_c)
    angle_b_to_c = get_angle_cartesian(corner_b, corner_c)
    axis_a_to_b = np.cross(corner_a, corner_b)
    axis_a_to_c = np.cross(corner_a, corner_c)

    # Input validation. The corners have to define a non-degenerate triangle
    if np.count_nonzero(axis_a_to_b) == 0:
        raise ValueError('Directions a and b are parallel')
    if np.count_nonzero(axis_a_to_c) == 0:
        raise ValueError('Directions a and c are parallel')

    rotations = []

    # Generate a list of theta_count evenly spaced angles theta_b in the range
    # [0, angle_a_to_b] and an equally long list of evenly spaced angles
    # theta_c in the range[0, angle_a_to_c].
    # Ensure that we keep the resolution also along the direction to the corner
    # b or c farthest away from a.
    theta_count = math.ceil(max(angle_a_to_b, angle_a_to_c) / resolution)
    for i, (theta_b, theta_c) in enumerate(
            zip(np.linspace(0, angle_a_to_b, theta_count),
                np.linspace(0, angle_a_to_c, theta_count))):
        # Define the corner local_b at a rotation theta_b from corner_a toward
        # corner_b on the circle surface. Similarly, define the corner local_c
        # at a rotation theta_c from corner_a toward corner_c.

        rotation_a_to_b = axangle2mat(axis_a_to_b, theta_b)
        rotation_a_to_c = axangle2mat(axis_a_to_c, theta_c)
        local_b = np.dot(rotation_a_to_b, corner_a)
        local_c = np.dot(rotation_a_to_c, corner_a)

        # Then define an axis and a maximum rotation to create a great cicle
        # arc between local_b and local_c. Ensure that this is not a degenerate
        # case where local_b and local_c are coincident.
        angle_local_b_to_c = get_angle_cartesian(local_b, local_c)
        axis_local_b_to_c = np.cross(local_b, local_c)
        if np.count_nonzero(axis_local_b_to_c) == 0:
            # Theta rotation ended at the same position. First position, might
            # be other cases?
            axis_local_b_to_c = corner_a
        axis_local_b_to_c /= np.linalg.norm(axis_local_b_to_c)

        # Generate points along the great circle arc with a distance defined by
        # resolution.
        phi_count_local = max(math.ceil(angle_local_b_to_c / resolution), 1)
        for j, phi in enumerate(
                np.linspace(0, angle_local_b_to_c, phi_count_local)):
            rotation_phi = axangle2mat(axis_local_b_to_c, phi)

            for k, psi in enumerate(inplane_rotations):
                # Combine the rotations. Order is important. The matrix is
                # applied from the left, and we rotate by theta first toward
                # local_b, then across the triangle toward local_c
                rotation = list(mat2euler(rotation_phi @ rotation_a_to_b, 'rzxz'))
                rotations.append(np.rad2deg([rotation[0], rotation[1], psi]))

    return np.unique(rotations, axis=0)
