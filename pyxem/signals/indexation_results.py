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
from typing import Sequence, Iterator
from traits.api import Undefined

from hyperspy._signals.lazy import LazySignal
from hyperspy.signal import BaseSignal
from hyperspy.axes import BaseDataAxis
from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion import Rotation, Orientation

from transforms3d.euler import mat2euler
from diffsims.crystallography._diffracting_vector import DiffractingVector

from orix.vector import Vector3d
from orix.projections import StereographicProjection
from orix.plot.inverse_pole_figure_plot import _get_ipf_axes_labels
from orix.vector.fundamental_sector import _closed_edges_in_hemisphere
from orix.plot import IPFColorKeyTSL, DirectionColorKeyTSL

from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
from scipy.spatial import Delaunay

import numpy as np
import hyperspy.api as hs

from pyxem.utils.indexation_utils import get_nth_best_solution, dict2phase, phase2dict
from pyxem.signals.diffraction_vectors2d import DiffractionVectors2D
from pyxem.utils._signals import _transfer_navigation_axes
from pyxem.utils.signal import compute_markers
from pyxem.utils._deprecated import deprecated


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


def get_ipf_outline(
    phase: Phase,
    include_labels: bool = True,
    offset_x: float = 0.85,
    offset_y: float = 0.85,
    scale: float = 0.2,
):
    """Get the outline of the IPF for the orientation map as a marker in the
    upper right hand corner including labels if desired.

    Parameters
    ----------
    phase: Phase
        The phase to use, defining the symmerty reduced zone
    include_labels : bool
        If True, the labels for the axes will be included.
    offset : float
        The offset of the markers from the lower left of the plot (as a fraction of the axis).
    scale : float
        The scale (as a fraction of the axis) for the markers.

    Returns
    -------
    polygon_sector : hs.plot.markers.Polygons
        The outline of the IPF as a marker
    texts : hs.plot.markers.Texts
        The text labels for the IPF axes
    maxes : np.ndarray
        The maximum values for the axes
    mins : np.ndarray
        The minimum values for the axes
    """
    offset = np.array([offset_x, offset_y])
    # Creating Lines around QuadMesh
    sector = phase.point_group.fundamental_sector
    s = StereographicProjection()

    edges = _closed_edges_in_hemisphere(sector.edges, sector)
    ex, ey = s.vector2xy(edges)
    original_offset = np.vstack((ex, ey)).T
    mins, maxes = original_offset.min(axis=0), original_offset.max(axis=0)
    original_offset = (
        (original_offset - ((maxes + mins) / 2)) / (maxes.max() - mins.min()) * scale
    )
    original_offset = original_offset + offset
    polygon_sector = hs.plot.markers.Polygons(
        verts=original_offset[np.newaxis],
        transform="axes",
        alpha=1,
        facecolor="none",
    )
    if include_labels:
        labels = _get_ipf_axes_labels(sector.vertices, symmetry=phase.point_group)
        tx, ty = s.vector2xy(sector.vertices)
        texts_offset = np.vstack((tx, ty)).T
        texts_offset = (
            (texts_offset - ((maxes + mins) / 2)) / (maxes.max() - mins.min()) * scale
        )
        texts_offset = texts_offset + offset
        texts = hs.plot.markers.Texts(
            texts=labels,
            offsets=texts_offset,
            sizes=(1,),
            offset_transform="axes",
            facecolor="k",
        )
        return polygon_sector, texts, maxes, mins


def get_ipf_annotation_markers(phase: Phase, offset: float = 0.85, scale: float = 0.2):
    """Get the outline of the IPF for the orientation map as a marker in the
    upper right hand corner including labels if desired. As well as the color
    mesh for the IPF.

    Parameters
    ----------
    offset : float
        The offset of the markers from the lower left of the plot (as a fraction of the axis).
    scale : float
        The scale (as a fraction of the axis) for the markers.

    Returns
    -------
    polygon_sector : hs.plot.markers.Polygons
        The outline of the IPF as a marker
    texts : hs.plot.markers.Texts
        The text labels for the IPF axes
    mesh : hs.plot.markers.Markers
        The color mesh for the IPF (using :class:`matplotlib.collections.QuadMesh`)
    """

    polygon_sector, texts, _, _ = get_ipf_outline(
        phase, offset_x=offset, offset_y=offset, scale=scale
    )

    # Create Color Mesh
    color_key = DirectionColorKeyTSL(symmetry=phase.point_group)
    g, ext = color_key._create_rgba_grid(return_extent=True)

    max_x = np.max(ext[1])
    min_x = np.min(ext[1])

    max_y = np.max(ext[0])
    min_y = np.min(ext[0])

    # center extent:
    y = np.linspace(ext[0][0], ext[0][1], g.shape[1] + 1) - ((max_y + min_y) / 2)

    y = y / (max_y - min_y) * scale + offset

    x = np.linspace(ext[1][1], ext[1][0], g.shape[0] + 1) - ((max_x + min_x) / 2)

    x = x / (max_x - min_x) * scale + offset
    xx, yy = np.meshgrid(y, x)

    mesh = hs.plot.markers.Markers(
        collection=QuadMesh,
        coordinates=np.stack((xx, yy), axis=-1),
        array=g,
        transform="axes",
        offset_transform="display",
        offsets=[[0, 0]],
    )
    return polygon_sector, mesh, texts


def vectors_to_single_phase_ipf_markers(
    vectors: Vector3d,
    phase: Phase,
    normalized_correlation: np.ndarray,
    offset_x: float = 0.85,
    offset_y: float = 0.85,
    scale: float = 0.2,
    color: str = "green",
    cmap: str = None,
):
    """Convert the orientation map to a set of inverse pole figure
    markers which visualizes the best matching orientations in the
    reduced S2 space.

    Parameters
    ----------
    vectors : Vector3d
        The vectors to display. These will be projected into the fundamental sector
    phase : Phase
        The phase for the fundamental sector
    normalized_correrlation : np.ndarray
        Correlation scores, normalized to lie between 0 and 1.
        Used for alpha for single-color plotting, and for the color map
    offset : float
        The offset of the markers from the center of the plot
    scale : float
        The scale (as a fraction of the axis) for the markers.
    """
    offset = np.array([offset_x, offset_y])

    markers = []

    vectors = vectors.in_fundamental_sector(phase.point_group)
    polygon_sector, texts, maxes, mins = get_ipf_outline(
        phase, offset_x=offset_x, offset_y=offset_y, scale=scale
    )
    markers.append(polygon_sector)
    markers.append(texts)

    s = StereographicProjection()
    x, y = s.vector2xy(vectors)
    x = x.reshape(vectors.shape)
    y = y.reshape(vectors.shape)
    offsets = np.empty(shape=vectors.shape[:-1], dtype=object)
    alpha = np.ones(shape=vectors.shape[:-1], dtype=object)

    # For each set of n_best, we reverse the order (using [::-1])
    # to ensure the best spot is displayed on top
    for i in np.ndindex(offsets.shape):
        off = np.vstack((x[i], y[i])).T
        norm_points = (off - ((maxes + mins) / 2)) / (maxes.max() - mins.min()) * scale
        norm_points = norm_points + offset
        offsets[i] = norm_points[::-1]
        alpha[i] = normalized_correlation[i][::-1]

    square = hs.plot.markers.Squares(
        offsets=[offset],
        widths=(scale + scale / 2,),
        units="width",
        offset_transform="axes",
        facecolor="white",
        edgecolor="black",
    )
    # Square needs to be on the bottom
    markers.insert(0, square)

    best_points = hs.plot.markers.Points(
        offsets=offsets.T,
        sizes=(4,),
        offset_transform="axes",
        alpha=alpha.T,
        facecolor=color,
        cmap=cmap,
        array=None if cmap is None else alpha.T,
        norm=Normalize(0, 1),
    )
    markers.append(best_points)

    return (*markers,)


def vectors_to_coordinates(vectors):
    """
    Convert a set of diffraction vectors to coordinates. For use with the map
    function and making markers.
    """
    return np.vstack((vectors[:, 0], vectors[:, 1])).T


def vectors_to_intensity(vectors, scale=1):
    """
    Convert a set of diffraction vectors to coordinates. For use with the map
    function and making markers.
    """

    return (vectors[:, 3] / np.max(vectors[:, 3])) * scale


def vectors_to_text(vectors, fast=True):
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
    for hkl in vectors[:, 4:]:
        h, k, l = np.round(hkl).astype(np.int16)
        if fast:
            out.append(f"{h} {k} {l}")
        else:
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


@deprecated(
    since="0.20",
    removal="1.0.0",
    alternative="pyxem.signals.indexation_results.vectors_from_orientation_map",
)
def extract_vectors_from_orientation_map(**kwargs):
    return vectors_from_orientation_map(**kwargs)


def vectors_from_orientation_map(
    result,
    vectors,
    phases,
    phase_index,
    hkl,
    intensities,
    n_best_index=0,
    return_object=True,
):
    """
    Note that any function which passes a large number of python objects back/forth to dask causes
    lots of overhead.  We are going to try to stick with just passing numpy arrays... Also,
    the Orix Rotation class has a really high overhead which is unfortunate. That is something to
    consider optimizing in the future.

    Parameters
    ----------
    result : np.ndarray
        The result array from the orientation map
    vectors : DiffractingVector
        The vectors from the simulation
    phases : List of orix.crystal_map.Phase
        The phase array from the simulation
    phase_index : np.ndarray
        The index of the phase for each vector
    intensities : np.ndarray
        The intensities of the vectors
    n_best_index : int
        The index of the best match to use
    return_object : bool
        If True, return a DiffractingVector object, otherwise return a numpy array with a stack of
        [x, y, z, intensity, h, k, l]
    """
    index, _, rotation, mirror = result[n_best_index, :].T
    index = index.astype(int)
    if vectors.ndim == 0:
        data = vectors
        phase = phases
        hkl = hkl
    else:
        data = vectors[index]
        phase = phases[phase_index[index]]
        intensities = intensities[index]
        hkl = hkl[index]
    phase = dict2phase(**phase)
    # Copy manually, as deepcopy adds a lot of overhead with the phase
    vectors = DiffractingVector(phase, xyz=data.copy())
    # Flip y, as discussed in https://github.com/pyxem/pyxem/issues/925
    vectors.y = -vectors.y

    rotation = Rotation.from_euler(
        (mirror * rotation, 0, 0), degrees=True, direction="crystal2lab"
    )
    coordinate_format = vectors.coordinate_format
    vectors = ~rotation * vectors.to_miller()
    vectors = DiffractingVector(
        vectors.phase,
        xyz=-vectors.data,  # Negating for proper alignment - is this flipping z direction?
        intensity=intensities,
    )
    vectors.coordinate_format = coordinate_format
    vectors.original_hkl = hkl

    # angle = np.deg2rad(mirror * rotation)
    # rot_coj = np.array([np.cos(0.5 * angle), 0, 0, -np.sin(0.5 * angle)])
    # data = qu_rotate_vec(rot_coj, vectors.data)
    # vectors = DiffractingVector(vectors.phase, xyz=data, intensity=intensities)
    # vectors.original_hkl = hkl

    # Mirror if necessary.
    # Mirroring, in this case, means casting (r, theta) to (r, -theta).
    # This is equivalent to (x, y) -> (x, -y)
    vectors.y = mirror * vectors.y

    if return_object:
        return vectors
    else:
        return np.hstack((vectors.data, vectors.intensity[:, np.newaxis], hkl))


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
        """Simulation object used to generate the orientation map.  This is stored in the metadata
        but can be accessed using the ``.simulation`` attribute. The simulation object is a
        :class:`diffsims.simulation.Simulation2D` object."""
        return self.metadata.get_item("simulation")

    @simulation.setter
    def simulation(self, value):
        self.metadata.set_item("simulation", value)

    def deepcopy(self, deepcopy_simulation=False):
        """Deepcopy the signal. Deepcopying the simulation is optional and is computationally
        expensive. If the simulation is not deepcopied, the simulation attribute of the
        copied signal is passed along"""

        if not deepcopy_simulation:
            simulation = self.simulation
            simulation._phase_slider = None
            simulation._rotation_slider = None
            self.simulation = None
        new = super().deepcopy()
        if not deepcopy_simulation:
            new.simulation = simulation
            self.simulation = simulation
        return new

    @property
    def num_rows(self):
        return self.axes_manager.signal_axes[1].size

    def to_rotation(self, flatten=False):
        """
        Convert the orientation map to a set of `orix.Quaternion.Rotation` objects.

        Parameters
        ----------
        flatten : bool
            If True, the rotations will be flattened to a 2D array. This is useful for passing
            to a `CrystalMap` object.
        Returns
        -------
        orix.Quaternion.Rotation
        """
        if self._lazy:
            raise ValueError(
                "Cannot create rotation from lazy signal. Please compute the signal first."
            )
        if isinstance(self.simulation.rotations, (list, np.ndarray)):
            quats = np.vstack([r.data for r in self.simulation.rotations])
            all_rotations = Rotation(data=quats)
        else:
            all_rotations = self.simulation.rotations
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

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the :meth:`hyperspy.api.signals.BaseSignal.map` function.
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

    @deprecated(
        since="0.20",
        removal="1.0.0",
        alternative="pyxem.signals.OrientationMap.to_vectors",
    )
    def to_single_phase_vectors(
        self, n_best_index: int = 0, **kwargs
    ) -> hs.signals.Signal1D:
        return self.to_vectors(n_best_index=n_best_index, **kwargs)

    def to_vectors(
        self, n_best_index: int = 0, return_object: bool = True, **kwargs
    ) -> hs.signals.Signal1D:
        """Get the reciprocal lattice vectors for each navigation position.

        Parameters
        ----------
        n_best_index: int
            The index into the `n_best` matchese
        **kwargs
            Additional keyword arguments to pass to the :meth:`hyperspy.api.signals.BaseSignal.map` function.
        """

        if self.simulation.has_multiple_phases:
            # With different phases, we cannot have a signal anymore, as the first dimension is now the phase
            vectors_signal = np.array([sim for sim in self.simulation])
        else:
            # Use vector data as signal in case of different vectors per navigation position
            vectors_signal = hs.signals.Signal1D(self.simulation.coordinates)

        data, intensities, phases, phase_indices, hkl = self.get_simulation_arrays()

        phases_dicts = [phase2dict(p) for p in phases]
        v = self.map(
            vectors_from_orientation_map,
            vectors=data,
            phases=phases_dicts,
            hkl=hkl,
            phase_index=phase_indices,
            intensities=intensities,
            inplace=False,
            output_signal_size=(),
            output_dtype=object,
            show_progressbar=False,
            n_best_index=n_best_index,
            return_object=return_object,
            **kwargs,
        )
        v.metadata.simulation = None
        return v

    def get_simulation_arrays(self):
        data = np.array([sim.data for sim in self.simulation], dtype=object)
        hkl = np.array([sim.hkl for sim in self.simulation], dtype=object)
        intensities = np.array([sim.intensity for sim in self.simulation], dtype=object)
        if self.simulation.has_multiple_phases:
            phases = self.simulation.phases
            csum = np.append(
                [
                    0,
                ],
                np.cumsum([r.size for r in self.simulation.rotations]),
            )
            phase_indices = np.zeros(csum[-1], dtype=int)
            for c in range(len(self.simulation.rotations)):
                phase_indices[csum[c] : csum[c + 1]] = c
        else:
            phases = [
                self.simulation.phases,
            ]
            phase_indices = np.zeros(self.simulation.rotations.size, dtype=int)
        return data, intensities, phases, phase_indices, hkl

    def to_crystal_map(self) -> CrystalMap:
        """Convert the orientation map to an :class:`orix.crystal_map.CrystalMap` object

        Returns
        -------
        orix.crystal_map.CrystalMap
            A crystal map object with the phase id as the highest matching phase.
        """
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
        if scan_unit is Undefined:
            scan_unit = "px"

        return CrystalMap(
            rotations=rotations,
            x=xx.flatten(),
            phase_id=phase_index,
            y=yy.flatten(),
            scan_unit=scan_unit,
            phase_list=phases,
        )

    def to_ipf_markers(
        self,
        offset_x: float = 0.85,
        offset_y: float = 0.85,
        scale: float = 0.2,
        color: str = "green",
        cmap: str = None,
    ):
        """Convert the orientation map to a set of inverse pole figure
        markers which visualizes the best matching orientations in the
        reduced S2 space.

        Parameters
        ----------
        offset : float
            The offset of the markers from the center of the plot
        scale : float
            The scale (as a fraction of the axis) for the markers.
        color : str
            The color of the markers. Overridden by cmap, if not None
        cmap : str
            Use a color map to show correlation scores.
            Takes priority over the `color` parameter.

        Notes
        -----
        This function can be slow with a large n_keep.
        Use `to_ipf_correlation_heatmap_markers` instead.
        """
        rots = self.to_rotation()
        vecs = rots * Vector3d.zvector()
        # Normalize the correlation scores
        cors = self.data[..., 1] / np.max(self.data[..., 1], axis=-1)[..., np.newaxis]
        if not self.simulation.has_multiple_phases:
            return vectors_to_single_phase_ipf_markers(
                vecs,
                self.simulation.phases,
                cors,
                offset_x=offset_x,
                offset_y=offset_y,
                scale=scale,
                color=color,
                cmap=cmap,
            )
        # Multiple phases, we make a few different IPFs
        markers = []
        phase_idxs = self.to_phase_index()
        for phase_idx, phase in enumerate(self.simulation.phases):
            # As a workaround, set alpha to 0 for the wrong phase
            phase_cors = cors * (phase_idxs == phase_idx)
            markers += list(
                vectors_to_single_phase_ipf_markers(
                    vecs,
                    phase,
                    phase_cors,
                    offset_x=offset_x,
                    offset_y=offset_y - 1.5 * scale * phase_idx,
                    scale=scale,
                    color=color,
                    cmap=cmap,
                )
            )
        return markers

    def to_ipf_correlation_heatmap_markers(
        self,
        offset_x: float = 0.85,
        offset_y: float = 0.85,
        scale: float = 0.2,
        cmap: str = "inferno",
    ):
        """Get the outline of the IPF for the orientation map as a marker in the
        upper right hand corner including labels if desired. As well as the color
        mesh for the IPF.

        Parameters
        ----------
        offset : float
            The offset of the markers from the lower left of the plot (as a fraction of the axis).
        scale : float
            The scale (as a fraction of the axis) for the markers.

        Returns
        -------
        polygon_sector : hs.plot.markers.Polygons
            The outline of the IPF as a marker
        texts : hs.plot.markers.Texts
            The text labels for the IPF axes
        mesh : hs.plot.markers.Markers
            The color mesh for the IPF (using :class:`matplotlib.collections.QuadMesh`)

        Notes
        -----
        It is recommended to have a large `n_keep` for this to look nice, e.g. 75% of the simulation bank.

        This will not look good if the rotations used in the simulation(s) consists of
        multiple different regions in the IPF.

        """
        phase_idx_signal = hs.signals.Signal1D(self.to_phase_index())
        phases = self.simulation.phases
        rotations = self.simulation.rotations
        if not self.simulation.has_multiple_phases:
            phases = [phases]
            rotations = [rotations]
            phase_idx_signal = 0

        s = StereographicProjection()
        rotation_sizes = np.cumsum([0] + [r.size for r in rotations])

        markers = []

        for phase_idx, phase in enumerate(phases):
            offset = np.array([offset_x, offset_y])
            polygon_sector, texts, maxs, mins = get_ipf_outline(
                phase, offset_x=offset_x, offset_y=offset_y, scale=scale
            )
            # We make a mesh of the correlation scores,
            # But we mask out the <n_best rotations.
            # Fetch all possible zone axes for the phase to speed up calculations
            rots = rotations[phase_idx]
            vecs = rots * Vector3d.zvector()

            x, y = s.vector2xy(vecs)
            xy = np.vstack((x, y)).T

            new_x = np.linspace(mins[0], maxs[0], 100 + 1)
            new_y = np.linspace(mins[1], maxs[1], 100 + 1)
            xx, yy = np.meshgrid(new_x, new_y)
            out_xy = np.stack((xx, yy), axis=-1)
            new_xy = np.vstack((xx[:-1, :-1].flatten(), yy[:-1, :-1].flatten())).T

            # Interpolate the values to a grid.
            # This is usually slow, e.g. using scipy.interpolate.griddata.
            # Since the data coordinates and interpolated coordinates
            # are always the same, and since we use linear interpolation,
            # we can pre-calculate the barycentric weights for Delaunay triangles
            # https://stackoverflow.com/a/20930910
            tri = Delaunay(xy, incremental=False)
            simplex = tri.find_simplex(new_xy)

            # points outside the convex hull gets assigned the index -1.
            # We later mask these out
            outside = simplex < 0

            vertices = np.take(tri.simplices, simplex, axis=0)
            temp = np.take(tri.transform, simplex, axis=0)
            delta = new_xy - temp[:, -1]
            bary = np.einsum("njk,nk->nj", temp[:, :-1, :], delta)
            weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

            def interpolate(values):
                return np.einsum("nj,nj->n", np.take(values, vertices), weights)

            def interpolate_and_mask_correlations(result, phase_idxs):
                indices, correlation, _, _ = result[phase_idxs == phase_idx].T
                max_cor = result[:, 1].max()
                indices = indices.astype(int) - rotation_sizes[phase_idx] - 1
                all_correlations = np.full(x.shape, np.nan)
                all_correlations[indices] = correlation / max_cor
                out = interpolate(all_correlations)
                out[outside] = np.nan
                return out

            g = self.map(
                interpolate_and_mask_correlations,
                phase_idxs=phase_idx_signal,
                inplace=False,
                ragged=True,
            ).data

            # Scale the coordinates
            out_xy = (out_xy - ((maxs + mins) / 2)) / (maxs.max() - mins.min()) * scale
            out_xy += offset

            mesh = hs.plot.markers.Markers(
                collection=QuadMesh,
                coordinates=out_xy,
                array=g.T,
                transform="axes",
                offset_transform="display",
                offsets=[[0, 0]],
                norm=Normalize(0, 1),
                cmap=cmap,
            )
            square = hs.plot.markers.Squares(
                offsets=[offset],
                widths=(scale + scale / 2,),
                units="width",
                offset_transform="axes",
                facecolor="white",
                edgecolor="black",
            )

            markers.append(square)
            markers.append(polygon_sector)
            markers.append(mesh)
            markers.append(texts)

            offset_y -= 1.5 * scale
        return (*markers,)

    def to_markers(
        self,
        n_best: int = 1,
        annotate: bool = False,
        marker_colors: str = ("red", "blue", "green", "orange", "purple"),
        text_color: str = "black",
        lazy_output: bool = None,
        annotation_shift: Sequence[float] = None,
        text_kwargs: dict = None,
        include_intensity: bool = False,
        intensity_scale: float = 1,
        fast: bool = True,
        **kwargs,
    ) -> Sequence[hs.plot.markers.Markers]:
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
        lazy_output: bool
            If True, the output will be a lazy signal. If None, the output will be lazy if the input is lazy.
        fast: bool
            If True the annotations will print as -h rather than \\bar{h}

        Returns
        -------
        all_markers : Sequence[hs.plot.markers.Markers]
            A list of markers for each of the n_best solutions

        """
        if lazy_output is None:
            lazy_output = self._lazy
        if text_kwargs is None:
            text_kwargs = dict()
        if annotation_shift is None:
            annotation_shift = [0, -0.15]
        if not self._lazy:
            navigation_chunks = (5,) * self.axes_manager.navigation_dimension
        else:
            navigation_chunks = None
        all_markers = []
        for n in range(n_best):
            vectors = self.to_vectors(
                n_best_index=n,
                return_object=False,
                lazy_output=True,
                navigation_chunks=navigation_chunks,
            )
            color = marker_colors[n % len(marker_colors)]
            if include_intensity:
                intensity = vectors.map(
                    vectors_to_intensity,
                    scale=intensity_scale,
                    inplace=False,
                    ragged=True,
                    output_dtype=object,
                    output_signal_size=(),
                    navigation_chunks=navigation_chunks,
                    lazy_output=True,
                ).data.T
                kwargs["sizes"] = intensity

            coords = vectors.map(
                vectors_to_coordinates,
                inplace=False,
                ragged=True,
                output_dtype=object,
                output_signal_size=(),
                navigation_chunks=navigation_chunks,
                lazy_output=True,
            )
            markers = hs.plot.markers.Points.from_signal(
                coords, facecolor="none", edgecolor=color, **kwargs
            )
            all_markers.append(markers)

            if annotate:
                texts = vectors.map(
                    vectors_to_text,
                    inplace=False,
                    lazy_output=True,
                    ragged=True,
                    output_dtype=object,
                    output_signal_size=(),
                    fast=fast,
                )
                # New signal for offset coordinates, as using inplace=True shifts the point markers too
                text_coords = coords.map(
                    lambda x: x + annotation_shift,
                    inplace=False,
                    lazy_output=True,
                )
                text_markers = hs.plot.markers.Texts.from_signal(
                    text_coords, texts=texts.data.T, color=text_color, **text_kwargs
                )
                all_markers.append(text_markers)
        if lazy_output is False:  # Compute all at once (as it is faster)
            all_markers = compute_markers(all_markers)
        return all_markers

    @deprecated(
        since="0.20.0",
        alternative="pyxem.signals.OrientationMap.to_polar_markers",
        removal="1.0.0",
    )
    def to_single_phase_polar_markers(
        self,
        signal_axes: Sequence[BaseDataAxis],
        n_best: int = 1,
        marker_colors: str = ("red", "blue", "green", "orange", "purple"),
        lazy_output: bool = None,
        **kwargs,
    ) -> Iterator[hs.plot.markers.Markers]:
        return self.to_polar_markers(
            n_best=n_best,
            marker_colors=marker_colors,
            lazy_output=lazy_output,
            **kwargs,
        )

    def to_polar_markers(
        self,
        n_best: int = 1,
        marker_colors: str = ("red", "blue", "green", "orange", "purple"),
        lazy_output: bool = None,
        **kwargs,
    ) -> Iterator[hs.plot.markers.Markers]:
        """
        Convert the orientation map to a set of markers for plotting in polar coordinates.

        Parameters
        ----------
        signal_axes: Sequence[BaseDataAxis]
            The signal axes for the orientation map. The first axis should be the azimuthal axis
            and the second axis should be the radial axis.
        n_best: int
            The number of best fit solutions to return as markers
        marker_colors:
            The colors to use for the markers. If there are more than 5 solutions, the colors will repeat.
        lazy_output:
            If True, the output will be a set of lazy markers. If False the output will be a set of computed markers.
            If None, the output will be lazy if the input is lazy or not lazy if the input is not lazy.
        kwargs:
            Additional keyword arguments to pass to the hyperspy.plot.markers.Points.from_signal function.
        Returns
        -------
        all_markers : Sequence[hs.plot.markers.Markers]
            An list of markers for each of the n_best solutions

        """
        if lazy_output is None:
            lazy_output = self._lazy
        if not lazy_output:
            navigation_chunks = (5,) * self.axes_manager.navigation_dimension
        else:
            navigation_chunks = None

        def vec2polar(vector):
            # Bug with dtype of python float
            vector = vector.astype(np.float64)
            r = np.linalg.norm(vector[:, :2], axis=1)
            theta = np.arctan2(
                vector[:, 1],
                vector[:, 0],
            )
            # flip y
            theta = -theta
            return np.vstack([theta, r]).T

        all_markers = []
        for n in range(n_best):
            color = marker_colors[n % len(marker_colors)]

            vecs = self.to_vectors(
                n_best_index=n,
                return_object=False,
                lazy_output=True,
                navigation_chunks=navigation_chunks,
            )
            markers_signal = vecs.map(vec2polar, inplace=False, ragged=True)

            if "sizes" not in kwargs:
                kwargs["sizes"] = 15
            markers = hs.plot.markers.Points.from_signal(
                markers_signal, facecolor="none", edgecolor=color, **kwargs
            )

            all_markers.append(markers)
        if lazy_output is False:  # Compute all at once (as it is faster)
            all_markers = compute_markers(all_markers)
        return all_markers

    def to_ipf_colormap(
        self,
        direction: Vector3d = Vector3d.zvector(),
        add_markers: bool = True,
    ):
        """Create a colored navigator and a legend (in the form of a marker) which can be passed as the
        navigator argument to the `plot` method of some signal.

        Parameters
        ----------
        direction : Vector3d
            The direction to plot the IPF in
        add_markers : bool
            If True, the markers for the IPF will be added to the navigator as permanent markers.

        Returns
        -------
        hs.signals.BaseSignal
        """
        oris = self.to_single_phase_orientations()[:, :, 0]
        ipfcolorkey = IPFColorKeyTSL(oris.symmetry, direction)

        float_rgb = ipfcolorkey.orientation2color(oris)
        int_rgb = (float_rgb * 255).astype(np.uint8)

        s = hs.signals.Signal1D(int_rgb)
        s.change_dtype("rgb8")
        s = s.T

        if add_markers:
            annotations = get_ipf_annotation_markers(self.simulation.phases)
            s.add_marker(
                annotations,
                permanent=True,
                plot_signal=False,
                plot_marker=False,
                plot_on_signal=False,
            )
        return s

    def to_phase_map(self):
        """Create a colored navigator which can be passed as the
        navigator argument to the `plot` method of some signal.

        Returns
        -------
        hs.signals.BaseSignal
        """
        if not self.simulation.has_multiple_phases:
            raise ValueError("Only a single phase present in simulation")

        phase_idxs = self.to_phase_index()
        # in case n_best = 1
        phase_idxs = phase_idxs.reshape(
            *self.axes_manager._navigation_shape_in_array, -1
        )
        colors = [p.color_rgb for p in self.simulation.phases]

        float_rgb = np.take(colors, phase_idxs[..., 0], axis=0)
        int_rgb = (float_rgb * 255).astype(np.uint8)

        s = hs.signals.Signal1D(int_rgb)
        s.change_dtype("rgb8")
        s = s.T
        return s

    def plot_over_signal(
        self,
        signal,
        add_vector_markers=True,
        add_ipf_markers=True,
        add_ipf_correlation_heatmap=False,
        add_ipf_colorkey=True,
        vector_kwargs=None,
        **kwargs,
    ):
        """Convenience method to plot the orientation map and the n-best matches over the signal.

        Parameters
        ----------
        signal : BaseSignal
            The signal to plot the orientation map over.
        add_vector_markers : bool
            If True, the vector markers will be added to the signal.
        add_ipf_markers : bool
            If True, the IPF best fit will be added to the signal in an overlay
        add_ipf_correlation_heatmap : bool
            If True, a correlation score heatmap as an IPF will be added to the signal in an overlay.
            This overrides the `add_ipf_markers` parameter.
        add_ipf_colorkey : bool
            If True, the IPF colorkey will be added to the signal
        vector_kwargs : dict
            Additional keyword arguments to pass to the `to_single_phase_markers` method
        kwargs
            Additional keyword arguments to pass to the
            :meth:`hyperspy.api.signals.Signal2D.plot` method
        """
        if self.simulation.has_multiple_phases:
            nav = self.to_phase_map()
            add_ipf_colorkey = False
        else:
            nav = self.to_ipf_colormap(add_markers=False)
        if vector_kwargs is None:
            vector_kwargs = dict()
        signal.plot(navigator=nav, **kwargs)
        if add_vector_markers:
            signal.add_marker(self.to_markers(1, **vector_kwargs))
        if add_ipf_markers and not add_ipf_correlation_heatmap:
            ipf_markers = self.to_ipf_markers()
            signal.add_marker(ipf_markers)
        if add_ipf_correlation_heatmap:
            heatmap_markers = self.to_ipf_correlation_heatmap_markers()
            signal.add_marker(heatmap_markers)
        if add_ipf_colorkey:
            signal.add_marker(
                get_ipf_annotation_markers(self.simulation.phases), plot_on_signal=False
            )


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


class LazyOrientationMap(OrientationMap, LazySignal):
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
