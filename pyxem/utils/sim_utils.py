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

import math

import numpy as np
from scipy.constants import h, m_e, e, c, pi
import collections

from .atomic_scattering_params import ATOMIC_SCATTERING_PARAMS
from pyxem.signals.electron_diffraction import ElectronDiffraction
from transforms3d.quaternions import mat2quat, rotate_vector


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

    Evaluates

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


def get_vectorized_list_for_atomic_scattering_factors(structure,
                                                      debye_waller_factors):
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

    coeffs, fcoords, occus, dwfactors = [], [], [], []

    for site in structure:
        c = ATOMIC_SCATTERING_PARAMS[site.element]
        coeffs.append(c)
        dwfactors.append(debye_waller_factors.get(site.element, 0))
        fcoords.append(site.xyz)
        occus.append(site.occupancy)

    coeffs = np.array(coeffs)
    fcoords = np.array(fcoords)
    occus = np.array(occus)
    dwfactors = np.array(dwfactors)

    return coeffs, fcoords, occus, dwfactors


def get_kinematical_intensities(structure,
                                g_indices,
                                g_hkls,
                                excitation_error,
                                maximum_excitation_error,
                                debye_waller_factors):
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
        The distances between the Ewald sphere and the peak centres.

    Returns
    -------
    peak_intensities : array-like
        The intensities of the peaks.

    """
    coeffs, fcoords, occus, dwfactors = get_vectorized_list_for_atomic_scattering_factors(
        structure=structure, debye_waller_factors=debye_waller_factors)

    # Store array of s^2 values since used multiple times.
    s2s = (g_hkls / 2) ** 2

    # Create array containing atomic scattering factors.
    fss = []
    for s2 in s2s:
        fs = np.sum(coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1)
        fss.append(fs)
    fss = np.array(fss)

    # Change the coordinate system of fcoords to align with that of g_indices
    fcoords = np.dot(fcoords, np.linalg.inv(np.dot(structure.lattice.stdbase,
                                                   structure.lattice.recbase)))

    # Calculate structure factors for all excited g-vectors.
    f_hkls = []
    for n in np.arange(len(g_indices)):
        g = g_indices[n]
        fs = fss[n]
        dw_correction = np.exp(-dwfactors * s2s[n])
        f_hkl = np.sum(fs * occus * np.exp(2j * np.pi * np.dot(fcoords, g))
                       * dw_correction)
        f_hkls.append(f_hkl)
    f_hkls = np.array(f_hkls)

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
                                  sigma=20):
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
    illumination = string
        Either 'plane_wave' or 'gaussian_probe' illumination

    Returns
    -------
    simulation : ElectronDiffraction
        ElectronDiffraction simulation.
    """
    # Get atomic scattering parameters for specified element.
    c = np.array(ATOMIC_SCATTERING_PARAMS[element])
    # Calculate electron wavelength for given keV.
    wavelength = get_electron_wavelength(accelerating_voltage)

    # Define a 2D array of k-vectors at which to evaluate scattering.
    l = np.linspace(-max_k, max_k, simulation_size)
    kx, ky = np.meshgrid(l, l)

    # Convert 2D k-vectors into 3D k-vectors accounting for Ewald sphere.
    k = np.array((kx, ky, (wavelength / 2) * (kx ** 2 + ky ** 2)))

    # Calculate scatering angle squared for each k-vector.
    s2s = (np.linalg.norm(k, axis=0) / 2) ** 2

    # Evaluate atomic scattering factor.
    fs = np.zeros_like(s2s)
    for i in np.arange(4):
        fs = fs + (c[i][0] * np.exp(-c[i][1] * s2s))

    # Evaluate scattering from all atoms
    scattering = np.zeros_like(s2s)
    if illumination == 'plane_wave':
        for r in atomic_coordinates:
            scattering = scattering + (fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    elif illumination == 'gaussian_probe':
        for r in atomic_coordinates:
            probe = (1 / (np.sqrt(2 * np.pi) * sigma)) * \
                np.exp((-np.abs(((r[0]**2) - (r[1]**2)))) / (4 * sigma**2))
            scattering = scattering + (probe * fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    else:
        raise ValueError("User specified illumination not defined.")

    # Calculate intensity
    intensity = (scattering * scattering.conjugate()).real

    return ElectronDiffraction(intensity)


def peaks_from_best_template(single_match_result, phase, library):
    """ Takes a match_result object and return the associated peaks, to be used
    in combination with map().

    Parameters
    ----------
    single_match_result : matching_results
        An entry in a matching_results object.
    phase : list
        List of keys to library, as passed to IndexationGenerator.correlate()
    library : dictionary
        Nested dictionary containing keys of [phase][rotation]

    Returns
    -------
    peaks : array
        Coordinates of peaks in the matching results object in calibrated units.
    """
    best_fit = single_match_result[np.argmax(single_match_result[:, 4])]
    _phase = phase[int(best_fit[0])]
    pattern = library.get_library_entry(
        phase=_phase,
        angle=(
            best_fit[1],
            best_fit[2],
            best_fit[3]))['Sim']
    peaks = pattern.coordinates[:, :2]  # cut z
    return peaks


def get_points_in_sphere(reciprocal_lattice, reciprocal_radius):
    """Finds all reciprocal lattice points inside a given reciprocal sphere.
    Utilised within the DifractionGenerator.

    Parameters
    ----------
    reciprocal_lattice : diffpy.Structure.Lattice
        The crystal lattice for the structure of interest.
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


def carry_through_navigation_calibration(new_signal, old_signal):
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
    try:
        x = new_signal.axes_manager.signal_axes[0]
        x.name = 'x'
        x.scale = old_signal.axes_manager.navigation_axes[0].scale
        x.units = 'nm'
    except IndexError:
        pass
        # Set calibration to same as signal for second navigation axis if there
    try:
        y = new_signal.axes_manager.signal_axes[1]
        y.name = 'y'
        y.scale = old_signal.axes_manager.navigation_axes[1].scale
        y.units = 'nm'
    except IndexError:
        pass

    return new_signal
