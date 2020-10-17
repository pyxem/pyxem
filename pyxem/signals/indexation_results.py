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

import hyperspy.api as hs
from hyperspy.signal import BaseSignal
from hyperspy.signals import Signal2D
from warnings import warn

from pyxem.signals import transfer_navigation_axes
from pyxem.signals.diffraction_vectors import generate_marker_inputs_from_peaks

from pyxem.utils.indexation_utils import peaks_from_best_template
from pyxem.utils.indexation_utils import crystal_from_template_matching
from pyxem.utils.indexation_utils import crystal_from_vector_matching

from pyxem import CrystallographicMap


def crystal_from_template_matching(z_matches):
    """Takes template matching results for a single navigation position and
    returns the best matching phase and orientation with correlation and
    reliability to define a crystallographic map.

    Parameters
    ----------
    z_matches : numpy.array
        Template matching results in an array of shape (m,3) sorted by
        correlation (descending) within each phase, with entries
        [phase, [z, x, z], correlation]

    Returns
    -------
    results_array : numpy.array
        Crystallographic mapping results in an array of shape (3) with entries
        [phase, np.array((z, x, z)), dict(metrics)]

    """
    # Create empty array for results.
    results_array = np.empty(3, dtype="object")
    # Consider single phase and multi-phase matching cases separately
    if np.unique(z_matches[:, 0]).shape[0] == 1:
        # get best matching phase (there is only one here)
        results_array[0] = z_matches[0, 0]
        # get best matching orientation Euler angles
        results_array[1] = z_matches[0, 1]
        # get template matching metrics
        metrics = dict()
        metrics["correlation"] = z_matches[0, 2]
        metrics["orientation_reliability"] = (
            100 * (1 - z_matches[1, 2] / z_matches[0, 2])
            if z_matches[0, 2] > 0
            else 100
        )
        results_array[2] = metrics
    else:
        # get best matching result
        index_best_match = np.argmax(z_matches[:, 2])
        # get best matching phase
        results_array[0] = z_matches[index_best_match, 0]
        # get best matching orientation Euler angles.
        results_array[1] = z_matches[index_best_match, 1]
        # get second highest correlation orientation for orientation_reliability
        z = z_matches[z_matches[:, 0] == results_array[0]]
        second_orientation = np.partition(z[:, 2], -2)[-2]
        # get second highest correlation phase for phase_reliability
        z = z_matches[z_matches[:, 0] != results_array[0]]
        second_phase = np.max(z[:, 2])
        # get template matching metrics
        metrics = dict()
        metrics["correlation"] = z_matches[index_best_match, 2]
        metrics["orientation_reliability"] = 100 * (
            1 - second_orientation / z_matches[index_best_match, 2]
        )
        metrics["phase_reliability"] = 100 * (
            1 - second_phase / z_matches[index_best_match, 2]
        )
        results_array[2] = metrics

    return results_array


def crystal_from_vector_matching(z_matches):
    """Takes vector matching results for a single navigation position and
    returns the best matching phase and orientation with correlation and
    reliability to define a crystallographic map.

    Parameters
    ----------
    z_matches : numpy.array
        Template matching results in an array of shape (m,5) sorted by
        total_error (ascending) within each phase, with entries
        [phase, R, match_rate, ehkls, total_error]

    Returns
    -------
    results_array : numpy.array
        Crystallographic mapping results in an array of shape (3) with entries
        [phase, np.array((z, x, z)), dict(metrics)]
    """
    if z_matches.shape == (1,):  # pragma: no cover
        z_matches = z_matches[0]

    # Create empty array for results.
    results_array = np.empty(3, dtype="object")

    # get best matching phase
    best_match = get_nth_best_solution(
        z_matches, "vector", key="total_error", descending=False
    )
    results_array[0] = best_match.phase_index

    # get best matching orientation Euler angles
    results_array[1] = np.rad2deg(mat2euler(best_match.rotation_matrix, "rzxz"))

    # get vector matching metrics
    metrics = dict()
    metrics["match_rate"] = best_match.match_rate
    metrics["ehkls"] = best_match.error_hkls
    metrics["total_error"] = best_match.total_error

    # get second highest correlation phase for phase_reliability (if present)
    other_phase_matches = [
        match for match in z_matches if match.phase_index != best_match.phase_index
    ]

    if other_phase_matches:
        second_best_phase = sorted(
            other_phase_matches, key=attrgetter("total_error"), reverse=False
        )[0]

        metrics["phase_reliability"] = 100 * (
            1 - best_match.total_error / second_best_phase.total_error
        )

        # get second best matching orientation for orientation_reliability
        same_phase_matches = [
            match for match in z_matches if match.phase_index == best_match.phase_index
        ]
        second_match = sorted(
            same_phase_matches, key=attrgetter("total_error"), reverse=False
        )[1]
    else:
        # get second best matching orientation for orientation_reliability
        second_match = get_nth_best_solution(
            z_matches, "vector", rank=1, key="total_error", descending=False
        )

    metrics["orientation_reliability"] = 100 * (
        1 - best_match.total_error / (second_match.total_error or 1.0)
    )

    results_array[2] = metrics

    return results_array


def get_phase_name_and_index(library):
    """Get a dictionary of phase names and its corresponding index value in library.keys().

    Parameters
    ----------
    library : DiffractionLibrary
        Diffraction library containing the phases and rotations

    Returns
    -------
    phase_name_index_dict : Dictionary {str : int}
    typically on the form {'phase_name 1' : 0, 'phase_name 2': 1, ...}
    """

    phase_name_index_dict = dict([(y, x) for x, y in enumerate(list(library.keys()))])
    return phase_name_index_dict


def peaks_from_best_template(single_match_result, library, rank=0):
    """Takes a TemplateMatchingResults object and return the associated peaks,
    to be used in combination with map().

    Parameters
    ----------
    single_match_result : ndarray
        An entry in a TemplateMatchingResults.
    library : DiffractionLibrary
        Diffraction library containing the phases and rotations.
    rank : int
        Get peaks from nth best orientation (default: 0, best vector match)

    Returns
    -------
    peaks : array
        Coordinates of peaks in the matching results object in calibrated units.
    """
    best_fit = get_nth_best_solution(single_match_result, "template", rank=rank)

    phase_names = list(library.keys())
    phase_index = int(best_fit[0])
    phase = phase_names[phase_index]
    simulation = library.get_library_entry(phase=phase, angle=tuple(best_fit[1]))["Sim"]

    peaks = simulation.coordinates[:, :2]  # cut z
    return peaks


def peaks_from_best_vector_match(single_match_result, library, rank=0):
    """Takes a VectorMatchingResults object and return the associated peaks,
    to be used in combination with map().

    Parameters
    ----------
    single_match_result : ndarray
        An entry in a VectorMatchingResults
    library : DiffractionLibrary
        Diffraction library containing the phases and rotations
    rank : int
        Get peaks from nth best orientation (default: 0, best vector match)

    Returns
    -------
    peaks : ndarray
        Coordinates of peaks in the matching results object in calibrated units.
    """
    best_fit = get_nth_best_solution(single_match_result, "vector", rank=rank)
    phase_index = best_fit.phase_index

    rotation_orientation = mat2euler(best_fit.rotation_matrix)
    # Don't change the original
    structure = library.structures[phase_index]
    sim = library.diffraction_generator.calculate_ed_data(
        structure,
        reciprocal_radius=library.reciprocal_radius,
        rotation=rotation_orientation,
        with_direct_beam=False,
    )

    # Cut z
    return sim.coordinates[:, :2]



def _basic_to_crystal_map():
    pass


class TemplateMatchingResults(Signal2D):
    """Template matching results containing the top n best matching crystal
    phase and orientation at each navigation position with associated metrics.
    """

    _signal_type = "template_matching"

    def plot_best_matching_results_on_signal(
        self, signal, library, permanent_markers=True, *args, **kwargs
    ):
        """Plot the best matching diffraction vectors on a signal.

        Parameters
        ----------
        signal : ElectronDiffraction2D
            The ElectronDiffraction2D signal object on which to plot the peaks.
            This signal must have the same navigation dimensions as the peaks.
        library : DiffractionLibrary
            Diffraction library containing the phases and rotations
        permanent_markers : bool
            Permanently save the peaks as markers on the signal
        *args :
            Arguments passed to signal.plot()
        **kwargs :
            Keyword arguments passed to signal.plot()
        """
        match_peaks = self.map(peaks_from_best_template, library=library, inplace=False)
        mmx, mmy = generate_marker_inputs_from_peaks(match_peaks)
        signal.plot(*args, **kwargs)
        for mx, my in zip(mmx, mmy):
            m = hs.markers.point(x=mx, y=my, color="red", marker="x")
            signal.add_marker(m, plot_marker=True, permanent=permanent_markers)

    def get_crystallographic_map(self, *args, **kwargs):
        """Obtain a crystallographic map specifying the best matching phase and
        orientation at each probe position with corresponding metrics.

        Returns
        -------
        cryst_map : CrystallographicMap
            Crystallographic mapping results containing the best matching phase
            and orientation at each navigation position with associated metrics.

            The Signal at each navigation position is an array of,

                            [phase, np.array((z,x,z)), dict(metrics)]

            which defines the phase, orientation as Euler angles in the zxz
            convention and metrics associated with the matching.

            Metrics for template matching results are
                'correlation'
                'orientation_reliability'
                'phase_reliability'

        """
        # TODO: Add alternative methods beyond highest correlation score.
        crystal_map = self.map(
            crystal_from_template_matching, inplace=False, *args, **kwargs
        )

        cryst_map = CrystallographicMap(crystal_map)
        cryst_map = transfer_navigation_axes(cryst_map, self)
        cryst_map.method = "template_matching"

        return cryst_map

class PatternMatchingResults(Signal2D):
    """Template matching results containing the top n best matching crystal
    phase and orientation at each navigation position with associated metrics.
    """

    _signal_type = "pattern_matching"

    def plot_best_matching_results_against_signal(
        self, signal, library,*args, **kwargs
    ):
        """Plot the best matching diffraction vectors on a signal.

        Parameters
        ----------
        signal : ElectronDiffraction2D
            The ElectronDiffraction2D signal object on which to plot the peaks.
            This signal must have the same navigation dimensions as the peaks.
        library : DiffractionLibrary
            Diffraction library containing the phases and rotations
        *args :
            Arguments passed to signal.plot()
        **kwargs :
            Keyword arguments passed to signal.plot()
        """
        pass

    def get_crystallographic_map(self, *args, **kwargs):
        """Obtain a crystallographic map specifying the best matching phase and
        orientation at each probe position with corresponding metrics.

        Returns
        -------
        cryst_map : CrystallographicMap
            Crystallographic mapping results containing the best matching phase
            and orientation at each navigation position with associated metrics.

            The Signal at each navigation position is an array of,

                            [phase, np.array((z,x,z)), dict(metrics)]

            which defines the phase, orientation as Euler angles in the zxz
            convention and metrics associated with the matching.

            Metrics for template matching results are
                'correlation'
                'orientation_reliability'
                'phase_reliability'

        """
        # TODO: Add alternative methods beyond highest correlation score.
        crystal_map = self.map(
            crystal_from_template_matching, inplace=False, *args, **kwargs
        )

        cryst_map = CrystallographicMap(crystal_map)
        cryst_map = transfer_navigation_axes(cryst_map, self)
        cryst_map.method = "template_matching"

        return cryst_map



class VectorMatchingResults(BaseSignal):
    """Vector matching results containing the top n best matching crystal
    phase and orientation at each navigation position with associated metrics.

    Attributes
    ----------
    vectors : DiffractionVectors
        Diffraction vectors indexed.
    hkls : BaseSignal
        Miller indices associated with each diffraction vector.
    """

    _signal_dimension = 0
    _signal_type = "vector_matching"

    def __init__(self, *args, **kwargs):
        BaseSignal.__init__(self, *args, **kwargs)
        # self.axes_manager.set_signal_dimension(2)
        self.vectors = None
        self.hkls = None

    def get_crystallographic_map(self, *args, **kwargs):
        """Obtain a crystallographic map specifying the best matching phase and
        orientation at each probe position with corresponding metrics.

        Returns
        -------
        cryst_map : CrystallographicMap
            Crystallographic mapping results containing the best matching phase
            and orientation at each navigation position with associated metrics.

            The Signal at each navigation position is an array of,

                            [phase, np.array((z,x,z)), dict(metrics)]

            which defines the phase, orientation as Euler angles in the zxz
            convention and metrics associated with the matching.

            Metrics for template matching results are
                'match_rate'
                'total_error'
                'orientation_reliability'
                'phase_reliability'
        """
        crystal_map = self.map(
            crystal_from_vector_matching, inplace=False, *args, **kwargs
        )

        cryst_map = CrystallographicMap(crystal_map)
        cryst_map = transfer_navigation_axes(cryst_map, self)
        cryst_map.method = "vector_matching"

        return cryst_map

    def get_indexed_diffraction_vectors(
        self, vectors, overwrite=False, *args, **kwargs
    ):
        """Obtain an indexed diffraction vectors object.

        Parameters
        ----------
        vectors : DiffractionVectors
            A diffraction vectors object to be indexed.

        Returns
        -------
        indexed_vectors : DiffractionVectors
            An indexed diffraction vectors object.
        """
        if overwrite is False:
            if vectors.hkls is not None:
                warn(
                    "The vectors supplied are already associated with hkls set "
                    "overwrite=True to replace these hkls."
                )
            else:
                vectors.hkls = self.hkls

        elif overwrite is True:
            vectors.hkls = self.hkls

        return vectors

    def plot_best_matching_results_on_signal(
        self, signal, permanent_markers=True, *args, **kwargs
    ):
        """Plot the best matching diffraction vectors on a signal.

        Parameters
        ----------
        signal : ElectronDiffraction2D
            The ElectronDiffraction2D signal object on which to plot the peaks.
            This signal must have the same navigation dimensions as the peaks.
        permanent_markers : bool
            Permanently save the peaks as markers on the signal. Default True.
        *args :
            Arguments passed to signal.plot()
        **kwargs :
            Keyword arguments passed to signal.plot()
        """
        match_peaks = self.vectors
        mmx, mmy = generate_marker_inputs_from_peaks(match_peaks)
        signal.plot(*args, **kwargs)
        for mx, my in zip(mmx, mmy):
            m = hs.markers.point(x=mx, y=my, color="red", marker="x")
            signal.add_marker(m, plot_marker=True, permanent=permanent_markers)
