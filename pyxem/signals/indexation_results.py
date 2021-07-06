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

from warnings import warn
import numpy as np
from transforms3d.euler import mat2euler

import hyperspy.api as hs
from hyperspy.signal import BaseSignal
from orix.quaternion import Rotation
from orix.crystal_map import CrystalMap

from pyxem.signals.diffraction_vectors import generate_marker_inputs_from_peaks
from pyxem.utils.signal import transfer_navigation_axes
from pyxem.utils.indexation_utils import get_nth_best_solution


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

    results_array[2] = metrics

    return results_array


def _peaks_from_best_template(single_match_result, library, rank=0):
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
    simulation = library.get_library_entry(phase=phase, angle=tuple(best_fit[1:4]))[
        "Sim"
    ]

    peaks = simulation.coordinates[:, :2]  # cut z
    return peaks


def _get_best_match(z):
    """Returns the match with the highest score for a given navigation pixel

    Parameters
    ----------
    z : np.array
        array with shape (5,n_matches), the 5 elements are phase, alpha, beta, gamma, score

    Returns
    -------
    z_best : np.array
        array with shape (5,)
    """
    return z[np.argmax(z[:, -1]),:]

def _get_phase_reliability(z):
    """ Returns the phase reliability (phase_alpha_best/phase_beta_best) for a given navigation pixel

    Parameters
    ----------
    z : np.array
        array with shape (5,n_matches), the 5 elements are phase, alpha, beta, gamma, score

    Returns
    -------
    phase_reliabilty : float
        np.inf if only one phase is avaliable
    """
    best_match = _get_best_match(z)
    phase_best = best_match[0]
    phase_best_score = best_match[4]

    # mask for other phases
    lower_phases = z[z[:,0] != phase_best]
    # needs a second phase, if none return np.inf
    if lower_phases.size > 0:
        phase_second = _get_best_match(lower_phases)
        phase_second_score = phase_second[4]
    else:
        return np.inf

    return phase_best_score/phase_second_score

def _get_second_best_phase(z):
    """ Returns the the second best phase for a given navigation pixel

    Parameters
    ----------
    z : np.array
        array with shape (5,n_matches), the 5 elements are phase, alpha, beta, gamma, score

    Returns
    -------
    phase_id : int
        associated with the second best phase
    """
    best_match = _get_best_match(z)
    phase_best = best_match[0]

    # mask for other phases
    lower_phases = z[z[:,0] != phase_best]

    # needs a second phase, if none return -1
    if lower_phases.size > 0:
        phase_second = _get_best_match(lower_phases)
        return phase_second[4]
    else:
        return -1

class GenericMatchingResults:
    def __init__(self, data):
        self.data = hs.signals.Signal2D(data)

    def to_crystal_map(self):
        """
        Exports an indexation result with multiple results per navigation position to
        crystal map with one result per pixel

        Returns
        -------
        orix.CrystalMap
        """
        _s = self.data.map(_get_best_match, inplace=False)

        """ Gets properties """
        phase_id = _s.isig[0].data.flatten()
        alpha = _s.isig[1].data.flatten()
        beta = _s.isig[2].data.flatten()
        gamma = _s.isig[3].data.flatten()
        score = _s.isig[4].data.flatten()

        """ Gets navigation placements """
        xy = np.indices(_s.data.shape[:2])
        x = xy[1].flatten()
        y = xy[0].flatten()

        """ Tidies up so we can put these things into CrystalMap """
        euler = np.deg2rad(np.vstack((alpha, beta, gamma)).T)
        rotations = Rotation.from_euler(
            euler, convention="bunge", direction="crystal2lab"
        )

        """ add various properties """
        phase_reliabilty = self.data.map(_get_phase_reliability,inplace=False).data.flatten()
        second_phase = self.data.map(_get_second_best_phase,inplace=False).data.flatten()
        properties = {"score": score,
                      "phase_reliabilty" : phase_reliabilty,
                      "second_phase" : second_phase}

        return CrystalMap(
            rotations=rotations, phase_id=phase_id, x=x, y=y, prop=properties
        )


class TemplateMatchingResults(GenericMatchingResults):
    """Template matching results containing the top n best matching crystal
    phase and orientation at each navigation position with associated metrics.

    Examples
    --------
    Saving the signal containing all potential matches at each pixel

    >>> TemplateMatchingResult.data.save("filename")

    Exporting the best matches to a crystal map

    >>> xmap = TemplateMatchingResult.to_crystal_map()
    """

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
        match_peaks = self.data.map(
            _peaks_from_best_template, library=library, inplace=False
        )
        mmx, mmy = generate_marker_inputs_from_peaks(match_peaks)
        signal.plot(*args, **kwargs)
        for mx, my in zip(mmx, mmy):
            m = hs.markers.point(x=mx, y=my, color="red", marker="x")
            signal.add_marker(m, plot_marker=True, permanent=permanent_markers)


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
        cryst_map : Signal2D
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

        crystal_map = transfer_navigation_axes(crystal_map, self)
        return crystal_map

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
