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

import numpy as np
import pytest
from transforms3d.euler import euler2mat
import dask.array as da

import sys

from diffsims.generators.simulation_generator import SimulationGenerator
from orix.sampling import get_sample_reduced_fundamental
from orix.quaternion import Rotation, Orientation
from orix.crystal_map import CrystalMap
from orix.vector import Vector3d

from pyxem.signals import VectorMatchingResults, DiffractionVectors, OrientationMap
from pyxem.utils.indexation_utils import OrientationResult
from pyxem.data import (
    si_grains,
    si_phase,
    si_tilt,
    si_grains_simple,
    fe_multi_phase_grains,
    fe_bcc_phase,
    fe_fcc_phase,
)

from pyxem.signals.indexation_results import vectors_from_orientation_map, phase2dict
import hyperspy.api as hs


@pytest.fixture
def sp_vector_match_result():
    # We require (total_error of row_1 > correlation row_2)
    res = np.empty(2, dtype="object")
    res[0] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.5,
        np.array([0.1, 0.05, 0.2]),
        0.1,
        1.0,
        0,
        0,
    )
    res[1] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.6,
        np.array([0.1, 0.10, 0.2]),
        0.2,
        1.0,
        0,
        0,
    )
    return VectorMatchingResults(res)


@pytest.fixture
def dp_vector_match_result():
    res = np.empty(4, dtype="object")
    res = res.reshape(2, 2)
    res[0, 0] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([90, 0, 0]), "rzxz"),
        0.6,
        np.array([0.1, 0.10, 0.2]),
        0.3,
        1.0,
        0,
        0,
    )
    res[0, 1] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 10, 20]), "rzxz"),
        0.5,
        np.array([0.1, 0.05, 0.2]),
        0.4,
        1.0,
        0,
        0,
    )
    res[1, 0] = OrientationResult(
        1,
        euler2mat(*np.deg2rad([0, 45, 45]), "rzxz"),
        0.8,
        np.array([0.1, 0.30, 0.2]),
        0.1,
        1.0,
        0,
        0,
    )
    res[1, 1] = OrientationResult(
        1,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.7,
        np.array([0.1, 0.05, 0.1]),
        0.2,
        1.0,
        0,
        0,
    )
    return VectorMatchingResults(res)


@pytest.mark.skip(reason="This functionality is under limited support as of 0.14.0")
def test_single_vector_get_crystallographic_map(sp_vector_match_result):
    _ = sp_vector_match_result.get_crystallographic_map()


@pytest.mark.skip(reason="This functionality is under limited support as of 0.14.0")
def test_double_vector_get_crystallographic_map(dp_vector_match_result):
    _ = dp_vector_match_result.get_crystallographic_map()


@pytest.mark.parametrize(
    "overwrite, result_hkl, current_hkl, expected_hkl",
    [
        (True, [0, 0, 1], None, [0, 0, 1]),
        (False, [0, 0, 1], None, [0, 0, 1]),
    ],
)
def test_vector_get_indexed_diffraction_vectors(
    overwrite, result_hkl, current_hkl, expected_hkl
):
    match_results = VectorMatchingResults(np.array([[1], [2]]))
    match_results.hkls = result_hkl
    vectors = DiffractionVectors(np.array([[1], [2]]))
    vectors.hkls = current_hkl
    match_results.get_indexed_diffraction_vectors(vectors, overwrite)
    np.testing.assert_allclose(vectors.hkls, expected_hkl)


def test_vector_get_indexed_diffraction_vectors_warn():
    match_results = VectorMatchingResults(np.array([[1], [2]]))
    match_results.hkls = [0, 0, 1]
    vectors = DiffractionVectors(np.array([[1], [2]]))
    vectors.hkls = [0, 0, 0]
    with pytest.warns(Warning):
        match_results.get_indexed_diffraction_vectors(vectors)
    np.testing.assert_allclose(vectors.hkls, [0, 0, 0])


class TestOrientationResult:
    """Testing the OrientationMap class for valid outputs. These tests are based on the
    examples provided in the documentation.
    """

    @pytest.fixture(scope="class")
    def single_rot_orientation_result(self):
        s = si_tilt()
        s.calibration.center = None
        polar_si_tilt = s.get_azimuthal_integral2d(
            npt=100, npt_azim=360, inplace=False, mean=True
        )
        phase = si_phase()
        generator = SimulationGenerator(200)
        sim = generator.calculate_diffraction2d(
            phase,
            rotation=Rotation.from_euler(
                [0, 0, 0],
                degrees=True,
            ),
            max_excitation_error=0.1,
            reciprocal_radius=1.5,
            with_direct_beam=False,
        )
        orientation_map = polar_si_tilt.get_orientation(sim)
        return orientation_map

    @pytest.fixture(scope="class")
    def multi_rot_orientation_result(self):
        s, r = si_grains(return_rotations=True)
        s.calibration.center = None
        polar = s.get_azimuthal_integral2d(
            npt=100, npt_azim=180, inplace=False, mean=True
        )
        phase = si_phase()
        generator = SimulationGenerator(200, minimum_intensity=0.05)
        rotations = get_sample_reduced_fundamental(
            resolution=1, point_group=phase.point_group
        )
        sims = generator.calculate_diffraction2d(
            phase,
            rotation=rotations,
            max_excitation_error=0.1,
            reciprocal_radius=2,
            with_direct_beam=False,
        )
        orientations = polar.get_orientation(sims)

        return orientations, r

    @pytest.fixture(scope="class")
    def simple_multi_rot_orientation_result(self):
        s, r = si_grains_simple(return_rotations=True)
        s.calibration.center = None
        polar = s.get_azimuthal_integral2d(
            npt=100, npt_azim=180, inplace=False, mean=True
        )
        phase = si_phase()
        generator = SimulationGenerator(200, minimum_intensity=0.05)
        rotations = get_sample_reduced_fundamental(
            resolution=1, point_group=phase.point_group
        )
        sims = generator.calculate_diffraction2d(
            phase,
            rotation=rotations,
            max_excitation_error=0.1,
            reciprocal_radius=2,
            with_direct_beam=True,
        )
        polar = polar**1
        orientations = polar.get_orientation(sims)
        return orientations, r, s

    @pytest.fixture(scope="class")
    def multi_phase_orientation_result(self):
        s = fe_multi_phase_grains()
        s.calibration.center = None
        polar = s.get_azimuthal_integral2d(
            npt=100, npt_azim=180, inplace=False, mean=True
        )
        phase = fe_fcc_phase()
        phase2 = fe_bcc_phase()

        generator = SimulationGenerator(200, minimum_intensity=0.05)
        rotations = get_sample_reduced_fundamental(
            resolution=2, point_group=phase.point_group
        )
        rotations2 = get_sample_reduced_fundamental(
            resolution=1, point_group=phase2.point_group
        )
        sims = generator.calculate_diffraction2d(
            [phase, phase2],
            rotation=[rotations, rotations2],
            max_excitation_error=0.1,
            reciprocal_radius=2,
        )
        orientations = polar.get_orientation(sims, n_best=3)
        return orientations

    def test_tilt_orientation_result(self, single_rot_orientation_result):
        assert isinstance(single_rot_orientation_result, OrientationMap)
        orients = single_rot_orientation_result.to_single_phase_orientations()
        # Check that the orientations are within 1 degree of the expected value
        degrees_between = orients.angle_with(
            Orientation.from_euler([0, 0, 0]), degrees=True
        )
        assert np.all(
            degrees_between[:, :2] <= 1
        )  # off by 1 degree (due to pixelation?)
        degrees_between = orients.angle_with(
            Orientation.from_euler([10, 0, 0], degrees=True), degrees=True
        )
        assert np.all(degrees_between[:, 2:] <= 1)

    def test_grain_orientation_result(self, simple_multi_rot_orientation_result):
        orientations, rotations, s = simple_multi_rot_orientation_result
        assert isinstance(rotations, Orientation)
        assert isinstance(orientations, OrientationMap)
        orients = orientations.to_single_phase_orientations()

        v1 = (orients * Vector3d.zvector()).in_fundamental_sector(orients.symmetry)
        v2 = (rotations * Vector3d.zvector()).in_fundamental_sector(rotations.symmetry)

        # Check that the orientations are within 2 degrees of the expected value.
        # Use 2 degrees since that is the angular resolution of the polar dataset
        degrees_between = v1.angle_with(v2, degrees=True)
        min_deg = np.min(degrees_between, axis=2)
        np.testing.assert_allclose(min_deg, 0, atol=3)

    def test_to_crystal_map(self, simple_multi_rot_orientation_result):
        orientations, rotations, s = simple_multi_rot_orientation_result
        crystal_map = orientations.to_crystal_map()
        assert isinstance(crystal_map, CrystalMap)
        assert np.all(crystal_map.phase_id == 0)

    def test_to_crystal_map_multi_phase(self, multi_phase_orientation_result):
        crystal_map = multi_phase_orientation_result.to_crystal_map()
        assert isinstance(crystal_map, CrystalMap)
        assert np.all(crystal_map.phase_id < 2)

    @pytest.mark.parametrize("annotate", [True, False])
    @pytest.mark.parametrize("lazy_output", [True, False])
    @pytest.mark.parametrize("add_intensity", [True, False])
    def test_to_markers(
        self,
        simple_multi_rot_orientation_result,
        annotate,
        lazy_output,
        add_intensity,
    ):
        fast = True
        orientations, rotations, s = simple_multi_rot_orientation_result
        markers = orientations.to_markers(
            lazy_output=lazy_output,
            annotate=annotate,
            include_intensity=add_intensity,
            fast=fast,
        )
        assert isinstance(markers[0], hs.plot.markers.Markers)

    def test_to_markers_lazy(self, simple_multi_rot_orientation_result):
        orientations, rotations, s = simple_multi_rot_orientation_result
        orientations = orientations.as_lazy()
        markers = orientations.to_markers()
        assert isinstance(markers[0].kwargs["offsets"], da.Array)

    def test_to_single_phase_markers_polar(self, simple_multi_rot_orientation_result):
        orientations, rotations, s = simple_multi_rot_orientation_result
        polar = s.get_azimuthal_integral2d(
            npt=100, npt_azim=180, inplace=False, mean=True
        )
        from pyxem.common import VisibleDeprecationWarning

        with pytest.warns(VisibleDeprecationWarning):
            markers = orientations.to_single_phase_polar_markers(
                polar.axes_manager.signal_axes
            )
        assert isinstance(markers[0], hs.plot.markers.Markers)

    def test_to_markers_polar(self, simple_multi_rot_orientation_result):
        orientations, rotations, s = simple_multi_rot_orientation_result
        markers = orientations.to_polar_markers()
        assert isinstance(markers[0], hs.plot.markers.Markers)

    def test_to_ipf_markers(self, simple_multi_rot_orientation_result):
        orientations, rotations, s = simple_multi_rot_orientation_result
        markers = orientations.to_ipf_markers()
        assert isinstance(markers[0], hs.plot.markers.Markers)

    @pytest.mark.parametrize("add_markers", [True, False])
    def test_to_ipf_map(self, simple_multi_rot_orientation_result, add_markers):
        orientations, rotations, s = simple_multi_rot_orientation_result
        navigator = orientations.to_ipf_colormap(add_markers=add_markers)
        assert isinstance(navigator, hs.signals.BaseSignal)
        if add_markers:
            assert len(navigator.metadata.Markers) == 3

    def test_to_phasemap(self, multi_phase_orientation_result):
        navigator = multi_phase_orientation_result.to_phase_map()
        assert isinstance(navigator, hs.signals.BaseSignal)

    def test_multi_phase_errors(self, multi_phase_orientation_result):
        with pytest.raises(ValueError):
            multi_phase_orientation_result.to_ipf_colormap()
        with pytest.raises(ValueError):
            multi_phase_orientation_result.to_single_phase_orientations()

    def test_lazy_error(self, simple_multi_rot_orientation_result):
        orientations, rotations, s = simple_multi_rot_orientation_result
        orientations = orientations.as_lazy()
        with pytest.raises(ValueError):
            rotations = orientations.to_rotation()
        with pytest.raises(ValueError):
            rotations = orientations.to_ipf_markers()

    def test_to_crystal_map_error(self, simple_multi_rot_orientation_result):
        orientations, rotations, s = simple_multi_rot_orientation_result
        orientations = hs.stack((orientations, orientations))
        with pytest.raises(ValueError):
            rotations = orientations.to_crystal_map()

    @pytest.mark.parametrize("add_vector_markers", [False, True])
    @pytest.mark.parametrize("add_ipf_markers", [False, True])
    @pytest.mark.parametrize("add_ipf_correlation_heatmap", [False, True])
    @pytest.mark.parametrize("add_ipf_colorkey", [False, True])
    @pytest.mark.parametrize(
        "vector_kwargs", [None, {"annotate": False}, {"annotate": True}]
    )
    def test_plot_over_single_phase_signal(
        self,
        simple_multi_rot_orientation_result,
        add_vector_markers,
        add_ipf_markers,
        add_ipf_correlation_heatmap,
        add_ipf_colorkey,
        vector_kwargs,
    ):
        orientations, rotations, s = simple_multi_rot_orientation_result
        orientations.plot_over_signal(
            s,
            add_vector_markers=add_vector_markers,
            add_ipf_markers=add_ipf_markers,
            add_ipf_correlation_heatmap=add_ipf_correlation_heatmap,
            add_ipf_colorkey=add_ipf_colorkey,
            vector_kwargs=vector_kwargs,
        )

    @pytest.mark.parametrize("add_vector_markers", [False, True])
    @pytest.mark.parametrize("add_ipf_correlation_heatmap", [False, True])
    @pytest.mark.parametrize("add_ipf_colorkey", [False, True])
    @pytest.mark.parametrize(
        "vector_kwargs", [None, {"annotate": False}, {"annotate": True}]
    )
    def test_plot_over_multi_phase_signal(
        self,
        multi_phase_orientation_result,
        add_vector_markers,
        add_ipf_correlation_heatmap,
        add_ipf_colorkey,
        vector_kwargs,
    ):
        add_ipf_markers = True
        # Mock signal
        s = hs.signals.Signal2D(
            np.zeros(multi_phase_orientation_result.data.shape[:-2])[
                ..., np.newaxis, np.newaxis
            ]
        )
        multi_phase_orientation_result.plot_over_signal(
            s,
            add_vector_markers=add_vector_markers,
            add_ipf_markers=add_ipf_markers,
            add_ipf_correlation_heatmap=add_ipf_correlation_heatmap,
            add_ipf_colorkey=add_ipf_colorkey,
            vector_kwargs=vector_kwargs,
        )

    def test_to_ipf_correlation_heatmap_markers_single_phase(
        self, simple_multi_rot_orientation_result
    ):
        orientations, rotations, s = simple_multi_rot_orientation_result
        markers = orientations.to_ipf_correlation_heatmap_markers()
        assert all(isinstance(m, hs.plot.markers.Markers) for m in markers)

    def test_to_ipf_correlation_heatmap_markers_multi_phase(
        self, multi_phase_orientation_result
    ):
        markers = multi_phase_orientation_result.to_ipf_correlation_heatmap_markers()
        assert all(isinstance(m, hs.plot.markers.Markers) for m in markers)

    def test_vector_markers_correctness(self):
        """
        Check if the markers are plotted correctly by performing orientation mapping
        on a non-centrosymmetric signal, where only one quadrant is non-zero,
        using a template with only one diffraction spot.
        This spot should then be plotted in the correct quadrant.
        """
        from pyxem.signals import Diffraction2D

        # Simple signal: 1 in first quadrant, 0 elsewhere
        signal = Diffraction2D(np.array([[[[0, 1], [0, 0]]]]))
        signal.calibration(center=None)
        polar = signal.get_azimuthal_integral2d(npt=10, npt_azim=36)

        from diffpy.structure import Lattice, Atom, Structure
        from orix.crystal_map import Phase

        # Primitive cubic structure, any will do
        l = Lattice(5, 5, 5, 90, 90, 90)
        a = [Atom("Au", xyz=[0, 0, 0], lattice=l)]
        s = Structure(a, l)
        p = Phase(space_group=221, structure=s)
        gen = SimulationGenerator()
        sim = gen.calculate_diffraction2d(p, with_direct_beam=False)

        # Set intensities to 0 for all vectors except one
        _, _, v = sim.get_simulation(0)
        v.intensity[:-1] = 0
        v.intensity[-1] = 1

        # Perform matching
        res = polar.get_orientation(sim)

        # Check cartesian
        x, y = res.to_markers()[0].kwargs["offsets"][0, 0][-1]
        assert x > 0
        # When plotting, the y-axis points downwards, but pyxem defines it as upwards.
        # Therefore, the first quadrant, as pyxem defines it, has negative y when plotting.
        assert y < 0

        # Check polar
        r, t = res.to_polar_markers()[0].kwargs["offsets"][0, 0][-1]
        assert r > 0
        assert 0 < t < np.pi / 2

    def test_vectors_from_orientation_map(self, single_rot_orientation_result):
        data, intensities, phases, phase_indices, hkl = (
            single_rot_orientation_result.get_simulation_arrays()
        )
        phases_dicts = [phase2dict(p) for p in phases]
        original_data = data.copy()
        v = vectors_from_orientation_map(
            single_rot_orientation_result.data[0, 0],
            vectors=data,
            phases=phases_dicts,
            hkl=hkl,
            phase_index=phase_indices,
            intensities=intensities,
            n_best_index=0,
        )

        np.testing.assert_almost_equal(
            v.original_hkl, single_rot_orientation_result.simulation.coordinates.hkl
        )
        np.testing.assert_equal(original_data, data)
