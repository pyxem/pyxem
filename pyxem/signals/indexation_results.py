# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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
from typing import Union, Literal, Sequence, Iterator

import hyperspy.api as hs
from hyperspy._signals.lazy import LazySignal
from hyperspy.signal import BaseSignal
import numpy as np
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation, Orientation
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL
from transforms3d.euler import mat2euler
from diffsims.crystallography._diffracting_vector import DiffractingVector
from orix.vector import Vector3d
from orix.projections import StereographicProjection
from orix.plot.inverse_pole_figure_plot import _get_ipf_axes_labels
from orix.vector.fundamental_sector import _closed_edges_in_hemisphere
import numpy as np
import hyperspy.api as hs

from pyxem.utils.indexation_utils import get_nth_best_solution
from pyxem.signals.diffraction_vectors2d import DiffractionVectors2D
from pyxem.utils._signals import _transfer_navigation_axes


def crystal_from_vector_matching(z_matches):
    """Takes vector matching results for a single navigation position and
    returns the best matching phase and orientation with correlation and
    reliability to define a crystallographic map.

    Parameters
    ----------
    z_matches : numpy.ndarray
        Template matching results in an array of shape (m,5) sorted by
        total_error (ascending) within each phase, with entries
        [phase, R, match_rate, ehkls, total_error]

    Returns
    -------
    results_array : numpy.ndarray
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


def _get_best_match(z):
    """Returns the match with the highest score for a given navigation pixel

    Parameters
    ----------
    z : numpy.ndarray
        array with shape (5,n_matches), the 5 elements are phase, alpha, beta, gamma, score

    Returns
    -------
    z_best : numpy.ndarray
        array with shape (5,)

    """
    return z[np.argmax(z[:, -1]), :]


def _get_phase_reliability(z):
    """Returns the phase reliability (phase_alpha_best/phase_beta_best) for a given navigation pixel

    Parameters
    ----------
    z : numpy.ndarray
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
    lower_phases = z[z[:, 0] != phase_best]
    # needs a second phase, if none return np.inf
    if lower_phases.size > 0:
        phase_second = _get_best_match(lower_phases)
        phase_second_score = phase_second[4]
    else:
        return np.inf

    return phase_best_score / phase_second_score


def _get_second_best_phase(z):
    """Returns the the second best phase for a given navigation pixel

    Parameters
    ----------
    z : numpy.ndarray
        array with shape (5,n_matches), the 5 elements are phase, alpha, beta, gamma, score

    Returns
    -------
    phase_id : int
        associated with the second best phase
    """
    best_match = _get_best_match(z)
    phase_best = best_match[0]

    # mask for other phases
    lower_phases = z[z[:, 0] != phase_best]

    # needs a second phase, if none return -1
    if lower_phases.size > 0:
        phase_second = _get_best_match(lower_phases)
        return phase_second[4]
    else:
        return -1


def vectors_to_coordinates(vectors):
    """
    Convert a set of diffraction vectors to coordinates. For use with the map
    function and making markers.
    """
    return np.vstack((vectors.x, vectors.y)).T


def vectors_to_intensity(vectors, scale=1):
    """
    Convert a set of diffraction vectors to coordinates. For use with the map
    function and making markers.
    """

    return (vectors.intensity / np.max(vectors.intensity)) * scale


def vectors_to_text(vectors):
    """
    Convert a set of diffraction vectors to text. For use with the map function
    and making text markers.
    """

    def add_bar(i: int) -> str:
        if i < 0:
            return f"$\\bar{{{abs(i)}}}$"
        else:
            return f"{i}"

    out = []
    for hkl in vectors.hkl:
        h, k, l = np.round(hkl).astype(np.int16)
        out.append(f"({add_bar(h)} {add_bar(k)} {add_bar(l)})")
    return out


def rotation_from_orientation_map(result, rots):
    if rots.ndim == 1:
        rots = rots[np.newaxis, :]
    index, _, rotation, mirror = result.T
    index = index.astype(int)
    ori = rots[index]
    euler = (
        Orientation(ori).to_euler(
            degrees=True,
        )
        * mirror[..., np.newaxis]
    )
    euler[:, 0] = rotation
    ori = Orientation.from_euler(euler, degrees=True).data
    return ori


def extract_vectors_from_orientation_map(result, all_vectors):
    index, _, rotation, mirror = result[n_best_index, :].T
    index = index.astype(int)
    if all_vectors.ndim == 0:
        vectors = all_vectors
    else:
        vectors = all_vectors[index]
    # Copy manually, as deepcopy adds a lot of overhead with the phase
    intensity = vectors.intensity
    vectors = DiffractingVector(vectors.phase, xyz=vectors.data.copy())
    # Flip y, as discussed in https://github.com/pyxem/pyxem/issues/925
    vectors.y = -vectors.y
    # Mirror if necessary
    vectors.y = mirror * vectors.y
    rotation = Rotation.from_euler(
        (rotation, 0, 0), degrees=True, direction="crystal2lab"
    )
    vectors = ~rotation * vectors.to_miller()
    vectors = DiffractingVector(
        vectors.phase, xyz=vectors.data.copy(), intensity=intensity
    )

    return vectors


def orientation2phase(orientations, sizes):
    o = orientations[:, 0]
    return np.searchsorted(sizes, o)


class OrientationMap(DiffractionVectors2D):
    """Signal class for orientation maps.  Note that this is a special case where
    for each navigation position, the signal contains the top n best matches in the form
    of a nx4 array with columns [index,correlation, in-plane rotation, mirror(factor)]

    The Simulation is saved in the metadata but can be accessed using the .simulation attribute.

    Parameters
    ----------
    *args
        See :class:`~hyperspy._signals.signal2d.Signal2D`.
    **kwargs
        See :class:`~hyperspy._signals.signal2d.Signal2D`
    """

    _signal_type = "orientation_map"

    @property
    def simulation(self):
        return self.metadata.get_item("simulation")

    @simulation.setter
    def simulation(self, value):
        self.metadata.set_item("simulation", value)

    def deepcopy(self):
        """Deepcopy the signal"""
        self.simulation._phase_slider = None
        self.simulation._rotation_slider = None
        return super().deepcopy()

    @property
    def num_rows(self):
        return self.axes_manager.signal_axes[1].size

    def to_rotation(self, flatten=False):
        """
        Convert the orientation map to a set of `orix.Quaternion.Rotation` objects.
        Returns
        -------
        """
        if self._lazy:
            warn("Computing the signal")
            self.compute()
        all_rotations = Rotation.stack(self.simulation.rotations).flatten()
        rotations = self.map(
            rotation_from_orientation_map,
            rots=all_rotations.data,
            inplace=False,
            lazy_output=False,
            output_signal_size=(self.num_rows, 4),
            output_dtype=float,
        )

        rots = Rotation(rotations)
        if flatten:
            shape1 = np.prod(rots.shape[:-1])
            rots = rots.reshape((shape1, rots.shape[-1]))
        return rots

    def to_phase_index(self):
        """
        Convert the orientation map to a set of phase ids

        Returns
        -------
        np.ndarray or None
            The phase ids for each pixel in the orientation map or None if the simulation
            does not have multiple phases.
        """
        if self.simulation.has_multiple_phases:
            sizes = np.cumsum([i.size for i in self.simulation.rotations])
            return self.map(orientation2phase, sizes=sizes, inplace=False).data
        else:
            return None

    def to_single_phase_orientations(self, **kwargs) -> Orientation:
        """Convert the orientation map to an `Orientation`-object,
        given a single-phase simulation.
        """
        if self.simulation.has_multiple_phases:
            raise ValueError(
                "Multiple phases found in simulation (use to_crystal_map instead)"
            )

        # Use the quaternion data from rotations to support 2D rotations,
        # i.e. unique rotations for each navigation position
        rotations = hs.signals.Signal2D(self.simulation.rotations.data)

        return Orientation(
            self.map(
                rotation_from_orientation_map,
                rots=rotations,
                inplace=False,
                output_signal_size=(self.num_rows, 4),
                output_dtype=float,
                **kwargs,
            ),
            symmetry=self.simulation.phases.point_group,
        )

    def to_single_phase_vectors(
        self, n_best_index: int = 0, **kwargs
    ) -> hs.signals.Signal1D:
        """
        Get the reciprocal lattice vectors for a single-phase simulation.

        Parameters
        ----------
        n_best_index: int
            The index into the `n_best` matchese
        """

        if self.simulation.has_multiple_phases:
            raise ValueError("Multiple phases found in simulation")

        # Use vector data as signal in case of different vectors per navigation position
        vectors_signal = hs.signals.Signal1D(self.simulation.coordinates)

        return self.map(
            extract_vectors_from_orientation_map,
            all_vectors=vectors_signal,
            inplace=False,
            output_signal_size=(),
            output_dtype=object,
            show_progressbar=False,
            **kwargs,
        )

    def to_crystal_map(self) -> CrystalMap:
        """Convert the orientation map to an `orix.CrystalMap` object"""
        if self.axes_manager.navigation_dimension != 2:
            raise ValueError(
                "Only 2D navigation supported. Please raise an issue if you are interested in "
                "support for 3+ navigation dimensions."
            )

        x, y = [ax.axis for ax in self.axes_manager.navigation_axes]
        xx, yy = np.meshgrid(x, y)
        xx = xx - np.min(xx)
        yy = yy - np.min(yy)
        scan_unit = self.axes_manager.navigation_axes[0].units
        rotations = self.to_rotation(flatten=True)
        phase_index = self.to_phase_index()

        if self.simulation.has_multiple_phases:
            phases = PhaseList(list(self.simulation.phases))
            if phase_index.ndim == 3:
                phase_index = phase_index[..., 0]
            phase_index = phase_index.flatten()
        else:
            phases = PhaseList(self.simulation.phases)

        return CrystalMap(
            rotations=rotations,
            x=xx.flatten(),
            phase_id=phase_index,
            y=yy.flatten(),
            scan_unit=scan_unit,
            phase_list=phases,
        )

    def to_ipf_markers(self):
        """Convert the orientation map to a set of inverse pole figure
        markers which visualizes the best matching orientations in the
        reduced S2 space.
        """
        if self._lazy:
            raise ValueError(
                "Cannot create markers from lazy signal. Please compute the signal first."
            )
        if self.simulation.has_multiple_phases:
            raise ValueError("Multiple phases found in simulation")

        orients = self.to_single_phase_orientations()
        sector = self.simulation.phases.point_group.fundamental_sector
        labels = _get_ipf_axes_labels(
            sector.vertices, symmetry=self.simulation.phases.point_group
        )
        s = StereographicProjection()
        vectors = orients * Vector3d.zvector()
        edges = _closed_edges_in_hemisphere(sector.edges, sector)
        vectors = vectors.in_fundamental_sector(self.simulation.phases.point_group)
        x, y = s.vector2xy(vectors)
        ex, ey = s.vector2xy(edges)
        tx, ty = s.vector2xy(sector.vertices)

        x = x.reshape(vectors.shape)
        y = y.reshape(vectors.shape)
        cor = self.data[..., 1]

        offsets = np.empty(shape=vectors.shape[:-1], dtype=object)
        correlation = np.empty(shape=vectors.shape[:-1], dtype=object)
        original_offset = np.vstack((ex, ey)).T
        texts_offset = np.vstack((tx, ty)).T

        mins, maxes = original_offset.min(axis=0), original_offset.max(axis=0)

        original_offset = (
            (original_offset - ((maxes + mins) / 2)) / (maxes - mins) * 0.2
        )
        original_offset = original_offset + 0.85

        texts_offset = (texts_offset - ((maxes + mins) / 2)) / (maxes - mins) * 0.2
        texts_offset = texts_offset + 0.85
        for i in np.ndindex(offsets.shape):
            off = np.vstack((x[i], y[i])).T
            norm_points = (off - ((maxes + mins) / 2)) / (maxes - mins) * 0.2
            norm_points = norm_points + 0.85
            offsets[i] = norm_points
            correlation[i] = cor[i] / np.max(cor[i]) * 0.5

        square = hs.plot.markers.Squares(
            offsets=[[0.85, 0.85]],
            widths=(0.3,),
            units="width",
            offset_transform="axes",
            facecolor="white",
            edgecolor="black",
        )
        polygon_sector = hs.plot.markers.Polygons(
            verts=original_offset[np.newaxis],
            transform="axes",
            alpha=1,
            facecolor="none",
        )

        best_points = hs.plot.markers.Points(
            offsets=offsets.T,
            sizes=(4,),
            offset_transform="axes",
            alpha=correlation.T,
            facecolor="green",
        )

        texts = hs.plot.markers.Texts(
            texts=labels,
            offsets=texts_offset,
            sizes=(1,),
            offset_transform="axes",
            facecolor="k",
        )
        return square, polygon_sector, best_points, texts

    def to_single_phase_markers(
        self,
        n_best: int = 1,
        annotate: bool = False,
        marker_colors: str = ("red", "blue", "green", "orange", "purple"),
        text_color: str = "black",
        lazy_output: bool = None,
        annotation_shift: Sequence[float] = None,
        text_kwargs: dict = None,
        include_intensity: bool = False,
        intesity_scale: float = 1,
        **kwargs,
    ):
        """Convert the orientation map to a set of markers for plotting.

        Parameters
        ----------
        n_best: int
            The amount of solutions to plot
        annotate : bool
            If True, the euler rotation and the correlation will be annotated on the plot using
            the `Texts` class from hyperspy.
        marker_color: str, optional
            The color of the point markers used for simulated reflections
        text_color: str, optional
            The color used for the text annotations for reflections. Does nothing if `annotate` is `False`.
        annotation_shift: List[float,float], optional
            The shift to apply to the annotation text. Default is [0,-0.1]
        include_intensity: bool
            If True, the intensity of the diffraction spot will be displayed with more intense peaks
            having a larger marker size.
        """
        if text_kwargs is None:
            text_kwargs = dict()
        if annotation_shift is None:
            annotation_shift = [0, -0.15]
        if not self._lazy:
            navigation_chunks = (5,) * self.axes_manager.navigation_dimension
        else:
            navigation_chunks = None

        for n in range(n_best):
            vectors = self.to_single_phase_vectors(
                lazy_output=lazy_output, navigation_chunks=navigation_chunks
            )
            color = marker_colors[n % len(marker_colors)]
            if include_intensity:
                intensity = vectors.map(
                    vectors_to_intensity,
                    scale=intesity_scale,
                    inplace=False,
                    ragged=True,
                    output_dtype=object,
                    output_signal_size=(),
                    navigation_chunks=navigation_chunks,
                ).data.T
                kwargs["sizes"] = intensity

            coords = vectors.map(
                vectors_to_coordinates,
                inplace=False,
                ragged=True,
                output_dtype=object,
                output_signal_size=(),
                navigation_chunks=navigation_chunks,
            )
            markers = hs.plot.markers.Points.from_signal(
                coords, facecolor="none", edgecolor=color, **kwargs
            )
            yield markers

            if annotate:
                texts = vectors.map(
                    vectors_to_text,
                    inplace=False,
                    lazy_output=lazy_output,
                    ragged=True,
                    output_dtype=object,
                    output_signal_size=(),
                )
                coords.map(lambda x: x + annotation_shift, inplace=True)
                text_markers = hs.plot.markers.Texts.from_signal(
                    coords, texts=texts.data.T, color=text_color, **text_kwargs
                )
                yield text_markers

    def to_polar_markers(self, n_best: int = 1) -> Iterator[hs.plot.markers.Markers]:
        (
            r_templates,
            theta_templates,
            intensities_templates,
        ) = self.simulation.polar_flatten_simulations()

        def marker_generator_factory(n_best_entry: int):
            def marker_generator(entry):
                index, correlation, rotation, factor = entry[n_best_entry]
                r = r_templates[int(index)]
                theta = theta_templates[int(index)]
                theta += 2 * np.pi + np.deg2rad(rotation)
                theta %= 2 * np.pi
                theta -= np.pi
                return np.array((theta, r)).T

            return marker_generator

        for n in range(n_best):
            markers_signal = self.map(
                marker_generator_factory(n),
                inplace=False,
                ragged=True,
                lazy_output=True,
            )
            markers = hs.plot.markers.Points.from_signal(markers_signal)
            yield markers

    def to_navigator(self, direction: Vector3d = Vector3d.zvector()):
        """Create a colored navigator and a legend (in the form of a marker) which can be passed as the
        navigator argument to the `plot` method of some signal.
        """
        oris = self.to_single_phase_orientations()[:, :, 0]
        ipfcolorkey = IPFColorKeyTSL(oris.symmetry, direction)

        float_rgb = ipfcolorkey.orientation2color(oris)
        int_rgb = (float_rgb * 255).astype(np.uint8)

        s = hs.signals.Signal1D(int_rgb)
        s.change_dtype("rgb8")

        return s

    def plot_over_signal(self, signal, annotate=False, **plot_kwargs):
        """Convenience method to plot the orientation map and the n-best matches over the signal.

        Parameters
        ----------
        signal : BaseSignal
            The signal to plot the orientation map over.
        annotate: bool
            If True, the euler rotation and the correlation will be annotated on the plot using
            the `Texts` class from hyperspy.

        Notes
        -----
        The kwargs are passed to the `signal.plot` function call
        """
        nav = self.to_navigator()
        signal.plot(navigator=nav, **plot_kwargs)
        signal.add_marker(self.to_single_phase_markers(1, annotate=annotate))

    def plot_inplane_rotation(self, **kwargs):
        """Plot the in-plane rotation of the orientation map as a 2D map."""
        pass


class GenericMatchingResults:
    def __init__(self, data):
        self.data = hs.signals.Signal2D(data)

    def to_crystal_map(self):
        """
        Exports an indexation result with multiple results per navigation position to
        crystal map with one result per pixel

        Returns
        -------
        :class:`~orix.crystal_map.CrystalMap`

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
        phase_reliabilty = self.data.map(
            _get_phase_reliability, inplace=False
        ).data.flatten()
        second_phase = self.data.map(
            _get_second_best_phase, inplace=False
        ).data.flatten()
        properties = {
            "score": score,
            "phase_reliabilty": phase_reliabilty,
            "second_phase": second_phase,
        }

        return CrystalMap(
            rotations=rotations, phase_id=phase_id, x=x, y=y, prop=properties
        )


class LazyOrientationMap(LazySignal, OrientationMap):
    pass


class VectorMatchingResults(BaseSignal):
    """Vector matching results containing the top n best matching crystal
    phase and orientation at each navigation position with associated metrics.

    Attributes
    ----------
    vectors : pyxem.signals.DiffractionVectors
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

        crystal_map = _transfer_navigation_axes(crystal_map, self)
        return crystal_map

    def get_indexed_diffraction_vectors(
        self, vectors, overwrite=False, *args, **kwargs
    ):
        """Obtain an indexed diffraction vectors object.

        Parameters
        ----------
        vectors : pyxem.signals.DiffractionVectors
            A diffraction vectors object to be indexed.

        Returns
        -------
        indexed_vectors : pyxem.signals.DiffractionVectors
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
