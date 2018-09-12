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


def get_electron_wavelength(accelerating_voltage):
    """Calculates the (relativistic) electron wavelength in Angstroms
    for a given accelerating voltage in kV.

    Evaluates

    Parameters
    ----------
    accelerating_voltage : float
        The accelerating voltage in kV.

    Returns
    -------
    wavelength : float
        The relativistic electron wavelength in Angstroms.

    """
    E = accelerating_voltage*1e3
    wavelength = h / math.sqrt(2 * m_e * e * E *\
                               (1 + (e / (2 * m_e * c * c)) * E))*1e10
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
    """ Get unique families of Miller indices.

    Parameters
    ----------
    hkls ([h, k, l]): List of Miller indices.

    Returns
    -------
    {hkl: multiplicity}: A dict with unique hkl and multiplicity.
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

def get_vectorized_list_for_atomic_scattering_factors(structure,debye_waller_factors):
    """
    Create a flattened array of coeffs, fcoords and occus for vectorized
    computation of atomic scattering factors later. Note that these are not
    necessarily the same size as the structure as each partially occupied
    specie occupies its own position in the flattened array.

    For primarily for internal use
    """

    coeffs,fcoords,occus,dwfactors = [],[],[],[]

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

    return coeffs,fcoords,occus,dwfactors

def get_interplanar_angle(structure, hkl1, hkl2, degrees=False):
    """Calculate the angle between two crystallogrpahic planes or reciprocal
    lattice vectors.
    Parameters
    ----------
    structure : Structure
        The structure for which to calculate the interplanar angle.
    hkl1 : list
        Miller indices of first plane.
    hkl2 : list
        Miller indices of second plane.
    degrees : bool
        If True, angle returned in degrees, else angle returned in radians.
    Returns
    -------
    phi : float
        The inter-planar angle.
    """
    #define lattice in real and reciprocal space
    l = structure.lattice
    rl = l.reciprocal_lattice_crystallographic
    #calculate real space interplanar spacings
    d1 = l.d_hkl(hkl1)
    d2 = l.d_hkl(hkl2)
    #separate h, k, l for angle calculation
    h1, k1, l1 = hkl1[0], hkl1[1], hkl1[2]
    h2, k2, l2 = hkl2[0], hkl2[1], hkl2[2]
    #specify reciprocal basis vector angles in radians
    alpha = radians(rl.alpha)
    beta = radians(rl.beta)
    gamma = radians(rl.gamma)
    #evaluate two parts of angle calculation
    x = h1 * h2 * rl.a**2 + k1 * k2 * rl.b**2 + l1 * l2 * rl.c**2
    y = (k1 * l2 + l1 * k2) * rl.b * rl.c * cos(alpha) + (h1 * l2 + l1 * h2) * rl.a * rl.c * cos(beta) + (h1 * k2 + k1 * h2) * rl.a * rl.b * cos(gamma)
    cos_phi = d1 * d2 * (x + y)
    #to avoid acos failing due to numerical precision
    if np.isclose(cos_phi, 1): cos_phi = 1
    if np.isclose(cos_phi, -1): cos_phi = -1
    #calculate angle
    phi = acos(cos_phi)
    #convert answer to degrees and return
    if degrees==True:
        phi = math.degrees(phi)
    return phi

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
    coeffs,fcoords,occus,dwfactors = get_vectorized_list_for_atomic_scattering_factors(structure=structure,debye_waller_factors=debye_waller_factors)

    # Store array of s^2 values since used multiple times.
    s2s = (g_hkls / 2) ** 2

    # Create array containing atomic scattering factors.
    fss = []
    for s2 in s2s:
        fs = np.sum(coeffs[:, :, 0] * np.exp(-coeffs[:, :, 1] * s2), axis=1)
        fss.append(fs)
    fss = np.array(fss)

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
    #Get atomic scattering parameters for specified element.
    c = np.array(ATOMIC_SCATTERING_PARAMS[element])
    #Calculate electron wavelength for given keV.
    wavelength = get_electron_wavelength(accelerating_voltage)

    #Define a 2D array of k-vectors at which to evaluate scattering.
    l = np.linspace(-max_k, max_k, simulation_size)
    kx, ky = np.meshgrid(l, l)

    #Convert 2D k-vectors into 3D k-vectors accounting for Ewald sphere.
    k = np.array((kx, ky, (wavelength / 2) * (kx ** 2 + ky **2)))

    #Calculate scatering angle squared for each k-vector.
    s2s = (np.linalg.norm(k, axis=0) / 2) ** 2

    #Evaluate atomic scattering factor.
    fs = np.zeros_like(s2s)
    for i in np.arange(4):
        fs = fs + (c[i][0] * np.exp(-c[i][1] * s2s))

    #Evaluate scattering from all atoms
    scattering = np.zeros_like(s2s)
    if illumination == 'plane_wave':
        for r in atomic_coordinates:
            scattering = scattering + (fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    elif illumination == 'gaussian_probe':
        for r in atomic_coordinates:
            probe = (1 / (np.sqrt(2*np.pi)*sigma))*np.exp((-np.abs(((r[0]**2) - (r[1]**2))))/(4*sigma**2))
            scattering = scattering + (probe * fs * np.exp(np.dot(k.T, r) * np.pi * 2j))
    else:
        raise ValueError("User specified illumination not defined.")

    #Calculate intensity
    intensity  = (scattering * scattering.conjugate()).real

    return ElectronDiffraction(intensity)


def peaks_from_best_template(single_match_result,phase,library):
    """ Takes a match_result object and return the associated peaks, to be used with
    in combination with map.

    Example : peaks= match_results.map(peaks_from_best_template,phase=phase,library=library)

    phase : list
        of keys to library, should be the same as passed to IndexationGenerator.correlate()

    library : nested dictionary containing keys of [phase][rotation]
    """

    best_fit = single_match_result[np.argmax(single_match_result[:,4])]
    _phase = phase[int(best_fit[0])]
    pattern = library.get_library_entry(phase=_phase,angle=(best_fit[1],best_fit[2],best_fit[3]))['Sim']
    peaks = pattern.coordinates[:,:2] #cut z
    return peaks

def get_points_in_sphere(reciprocal_lattice,reciprocal_radius):
    """
    Finds all reciprocal lattice points inside a given reciprocal sphere. Utilised
    within the DifractionGenerator.

    Inputs:  reciprocal_lattice : Diffy Lattice Object
             reciprocal_radius  : float

    Returns: np.arrays(): spot_indicies, spot_coords, spot_distances
             Note that spot_coords are the cartesian basis.

    """

    a,b,c = reciprocal_lattice.a,reciprocal_lattice.b,reciprocal_lattice.c
    h_max = np.ceil(reciprocal_radius/a)
    k_max = np.ceil(reciprocal_radius/b)
    l_max = np.ceil(reciprocal_radius/c)
    from itertools import product
    h_list = np.arange(-h_max,h_max+1)
    k_list = np.arange(-k_max,k_max+1)
    l_list = np.arange(-l_max,l_max+1)
    potential_points = np.asarray(list(product(h_list,k_list,l_list)))
    in_sphere = np.abs(reciprocal_lattice.dist(potential_points,[0,0,0])) < reciprocal_radius
    spot_indicies = potential_points[in_sphere]
    spot_coords = reciprocal_lattice.cartesian(spot_indicies)
    spot_distances = reciprocal_lattice.dist(spot_indicies,[0,0,0])

    return spot_indicies,spot_coords,spot_distances

def is_lattice_hexagonal(latt):
    """
    Attempts to determine if a lattice belongs
    to a hexagonal crystal. Will also return true
    for trigonal systems
    """
    truth_list = []
    truth_list.append(latt.a==latt.b)
    truth_list.append(latt.alpha == 90)
    truth_list.append(latt.beta == 90)
    truth_list.append(latt.gamma == 120)
    return len(truth_list) == np.sum(truth_list)
