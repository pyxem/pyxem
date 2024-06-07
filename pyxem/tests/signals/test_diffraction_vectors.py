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

import pytest
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.draw import disk
from skimage.morphology import disk as disk2

from hyperspy.signals import Signal2D
from hyperspy.signal import BaseSignal


from pyxem.signals import DiffractionVectors, DiffractionVectors2D
from pyxem.utils._subpixel_finding import (
    _center_of_mass_hs,
    _get_experimental_square,
    _conventional_xc,
)
from pyxem.data import tilt_boundary_data
from hyperspy.axes import UniformDataAxis

# DiffractionVectors correspond to a single list of vectors, a map of vectors
# all of equal length, and the ragged case. A fixture is defined for each of
# these cases and all methods tested for it.


@pytest.fixture(
    params=[
        np.array(
            [
                [
                    np.array(
                        [
                            [0.089685, 0.292971],
                            [0.017937, 0.277027],
                            [-0.069755, 0.257097],
                            [-0.165419, 0.241153],
                            [0.049825, 0.149475],
                            [-0.037867, 0.129545],
                            [-0.117587, 0.113601],
                        ]
                    ),
                    np.array(
                        [
                            [0.089685, 0.292971],
                            [0.017937, 0.277027],
                            [-0.069755, 0.257097],
                            [-0.165419, 0.241153],
                            [0.049825, 0.149475],
                            [-0.037867, 0.129545],
                            [-0.117587, 0.113601],
                            [0.149475, 0.065769],
                            [0.229195, 0.045839],
                            [0.141503, 0.025909],
                            [0.073741, 0.013951],
                            [0.001993, 0.001993],
                            [-0.069755, -0.009965],
                        ]
                    ),
                ],
                [
                    np.array(
                        [
                            [0.089685, 0.292971],
                            [0.017937, 0.277027],
                            [-0.069755, 0.257097],
                            [-0.165419, 0.241153],
                            [0.049825, 0.149475],
                            [-0.037867, 0.129545],
                            [-0.117587, 0.113601],
                            [0.149475, 0.065769],
                            [0.229195, 0.045839],
                            [0.141503, 0.025909],
                            [0.073741, 0.013951],
                        ]
                    ),
                    np.array([[0.001993, 0.001993]]),
                ],
            ],
            dtype=object,
        )
    ]
)
def diffraction_vectors_map(request):
    dvm = DiffractionVectors(request.param)
    dvm.axes_manager[0].name = "x"
    dvm.axes_manager[1].name = "y"
    return dvm


class TestVectorPlotting:
    def test_plot_diffraction_vectors(self, diffraction_vectors_map):
        with pytest.warns(UserWarning, match="distance_threshold=0 was given"):
            diffraction_vectors_map.plot_diffraction_vectors(
                xlim=1.0, ylim=1.0, distance_threshold=0
            )

    def test_plot_diffraction_vectors_on_signal(
        self, diffraction_vectors_map, diffraction_pattern
    ):
        diffraction_vectors_map.plot_diffraction_vectors_on_signal(diffraction_pattern)


def test_get_cartesian_coordinates(diffraction_vectors_map):
    accelerating_voltage = 200
    camera_length = 0.2
    diffraction_vectors_map.calculate_cartesian_coordinates(
        accelerating_voltage, camera_length
    )
    # Coordinate conversion is tested in vector_utils. Just test that the
    # result is stored correctly
    assert diffraction_vectors_map.cartesian is not None
    assert (
        diffraction_vectors_map.axes_manager[0].name
        == diffraction_vectors_map.cartesian.axes_manager[0].name
    )


class TestInitVectors:
    @pytest.fixture()
    def peaks(self):
        vectors = np.empty((2, 2), dtype=object)
        vectors[0, 0] = np.random.randint(0, 100, (5, 2))
        vectors[0, 1] = np.random.randint(0, 100, (6, 2))
        vectors[1, 0] = np.random.randint(0, 100, (7, 2))
        vectors[1, 1] = np.random.randint(0, 100, (8, 2))

        peaks = BaseSignal(vectors, ragged=True)
        return peaks

    @pytest.fixture()
    def peaks_w_intensity(self):
        vectors = np.empty((2, 2), dtype=object)
        vectors[0, 0] = np.random.randint(0, 100, (5, 3))
        vectors[0, 1] = np.random.randint(0, 100, (6, 3))
        vectors[1, 0] = np.random.randint(0, 100, (7, 3))
        vectors[1, 1] = np.random.randint(0, 100, (8, 3))

        peaks = BaseSignal(vectors, ragged=True)
        return peaks

    @pytest.mark.parametrize("column_names", (None, ["x", "y"]))
    @pytest.mark.parametrize("units", (None, ["nm", "nm"]))
    def test_from_peaks(self, peaks, column_names, units):
        peaks.axes_manager.navigation_axes[0].name = "x"
        peaks.axes_manager.navigation_axes[1].name = "y"
        dv = DiffractionVectors.from_peaks(
            peaks,
            center=(50, 50),
            calibration=0.1,
            column_names=column_names,
            units=units,
        )

        for i in np.ndindex((2, 2)):
            np.testing.assert_array_equal((peaks.data[i] - 50) * 0.1, dv.data[i])
        assert dv.scales == [0.1, 0.1]
        assert dv.axes_manager[0].name == "x"
        assert dv.axes_manager[1].name == "y"
        assert dv.column_names == ["x", "y"]
        if units is not None:
            assert dv.units == ["nm", "nm"]

    def test_from_peaks_lazy(self, peaks):
        peaks = peaks.as_lazy()
        dv = DiffractionVectors.from_peaks(
            peaks,
            center=(50, 50),
            calibration=0.1,
        )
        assert dv._lazy

        for i in np.ndindex((2, 2)):
            np.testing.assert_array_equal((peaks.data[i] - 50) * 0.1, dv.data[i])
        assert dv.scales == [0.1, 0.1]

    def test_from_peaks_calibration(self, peaks):
        peaks.metadata.add_node("Peaks.signal_axes")
        peaks.metadata.Peaks.signal_axes = (
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
        )
        dv = DiffractionVectors.from_peaks(
            peaks,
            center=None,
            calibration=None,
        )

        for i in np.ndindex((2, 2)):
            np.testing.assert_almost_equal((peaks.data[i]) * 0.1 - 5.0, dv.data[i])
        assert dv.scales == [0.1, 0.1]

    def test_from_peaks_calibration_to_markers(self, peaks):
        peaks.metadata.add_node("Peaks.signal_axes")
        peaks.metadata.Peaks.signal_axes = (
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
        )
        dv = DiffractionVectors.from_peaks(
            peaks,
            center=None,
            calibration=None,
        )
        points = dv.to_markers()
        for i in np.ndindex((2, 2)):
            np.testing.assert_almost_equal(
                ((peaks.data[i]) * 0.1 - 5.0)[:, 0],
                points.kwargs["offsets"][i[::-1]][:, 1],
            )

    def test_from_peaks_calibration_to_markers_with_intensity(self, peaks_w_intensity):
        peaks_w_intensity.metadata.add_node("Peaks.signal_axes")
        peaks_w_intensity.metadata.Peaks.signal_axes = (
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
        )
        dv = DiffractionVectors.from_peaks(
            peaks_w_intensity,
            center=None,
            calibration=None,
        )
        points = dv.to_markers()
        for i in np.ndindex((2, 2)):
            np.testing.assert_almost_equal(
                ((peaks_w_intensity.data[i]) * 0.1 - 5.0)[:, 0],
                points.kwargs["offsets"][i[::-1]][:, 1],
            )
        s = Signal2D(np.ones((2, 2, 10, 10)))
        s.add_marker(points)

    def test_from_peaks_calibration_error(self, peaks):
        with pytest.raises(ValueError):
            dv = DiffractionVectors.from_peaks(
                peaks,
                center=None,
                calibration=None,
            )
        with pytest.raises(ValueError):
            dv = DiffractionVectors.from_peaks(
                peaks,
                center=(1.0, 1.0),
                calibration=None,
            )
        with pytest.raises(ValueError):
            dv = DiffractionVectors.from_peaks(
                peaks,
                center=None,
                calibration=(1.0, 1.0),
            )

    def test_from_peaks_calibration_2_peaks(self, peaks):
        peaks.metadata.add_node("Peaks.signal_axes")
        peaks.metadata.Peaks.signal_axes = (
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
        )
        dv = DiffractionVectors.from_peaks(
            peaks,
            center=None,
            calibration=None,
        )
        pixels = dv.pixel_vectors

        for i in np.ndindex((2, 2)):
            np.testing.assert_almost_equal(peaks.data[i], pixels.data[i])

    def test_peaks_with_intensity(self, peaks_w_intensity):
        peaks_w_intensity.metadata.add_node("Peaks.signal_axes")
        peaks_w_intensity.metadata.Peaks.signal_axes = (
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
            UniformDataAxis(scale=0.1, offset=-5.0, units="nm"),
        )
        dv = DiffractionVectors.from_peaks(
            peaks_w_intensity,
            center=None,
            calibration=None,
        )
        pixels = dv.pixel_vectors
        for i in np.ndindex((2, 2)):
            np.testing.assert_almost_equal(peaks_w_intensity.data[i], pixels.data[i])

    def test_initial_metadata(self, diffraction_vectors_map):
        assert diffraction_vectors_map.scales is None
        assert diffraction_vectors_map.metadata.VectorMetadata["scales"] == None

        assert diffraction_vectors_map.offsets is None
        assert diffraction_vectors_map.metadata.VectorMetadata["offsets"] == None

        assert diffraction_vectors_map.detector_shape is None
        assert diffraction_vectors_map.metadata.VectorMetadata["detector_shape"] == None

    def test_set_scales(self, diffraction_vectors_map):
        diffraction_vectors_map.scales = 0.1
        assert diffraction_vectors_map.scales == [0.1, 0.1]
        assert (
            diffraction_vectors_map.scales == diffraction_vectors_map.pixel_calibration
        )

    def test_scales_error(self, diffraction_vectors_map):
        with pytest.raises(ValueError):
            diffraction_vectors_map.scales = [1, 2, 3]

    def test_set_column_names_error(self, diffraction_vectors_map):
        with pytest.raises(ValueError):
            diffraction_vectors_map.column_names = ["x", "y", "z"]

    def test_set_units_error(self, diffraction_vectors_map):
        with pytest.raises(ValueError):
            diffraction_vectors_map.units = ["x", "y", "z"]

    def test_set_units_singleton(self, diffraction_vectors_map):
        diffraction_vectors_map.units = "nm"
        assert diffraction_vectors_map.units == ["nm", "nm"]

    def test_set_column_names(self, diffraction_vectors_map):
        diffraction_vectors_map.column_names = ["x", "y"]
        assert diffraction_vectors_map.column_names == ["x", "y"]

    def test_set_column_names_none(self, diffraction_vectors_map):
        diffraction_vectors_map.column_names = None
        assert diffraction_vectors_map.column_names == ["column_0", "column_1"]

    def test_num_rows(self, diffraction_vectors_map):
        assert diffraction_vectors_map.num_rows is None

    def test_set_offsets_error(self, diffraction_vectors_map):
        with pytest.raises(ValueError):
            diffraction_vectors_map.offsets = [1, 2, 3]

    def test_setting_metadat(self, diffraction_vectors_map):
        diffraction_vectors_map.scales = 0.1
        assert diffraction_vectors_map.metadata.VectorMetadata["scales"] == [0.1, 0.1]

        diffraction_vectors_map.offsets = 1
        assert diffraction_vectors_map.metadata.VectorMetadata["offsets"] == [1, 1]

        diffraction_vectors_map.detector_shape = [100, 100]
        assert diffraction_vectors_map.metadata.VectorMetadata["detector_shape"] == [
            100,
            100,
        ]


class TestSubpixelRefinement:
    def test_com_accuracy(self):
        xx, yy = disk(center=(45, 50), radius=10)
        cir = np.zeros((100, 100))
        cir[xx, yy] = 1
        sq = _get_experimental_square(cir, (47, 50), 30)
        cen = _center_of_mass_hs(sq)
        assert np.allclose(np.array(cen) - 15, (-2.0, 0.0))

    def test_conventional_xc_accuracy(self):
        xx, yy = disk(center=(45, 50), radius=10)
        kernel = disk2(radius=10)
        cir = np.random.random((100, 100))

        cir[xx, yy] += 10
        sq = _get_experimental_square(cir, (47, 50), 30)
        cen = _conventional_xc(sq, kernel, upsample_factor=1)
        assert np.allclose(np.array(cen), (-2.0, 0.0))

    @pytest.mark.parametrize("intensity", (True, False))
    def test_subpixel_refinement_com(self, intensity):
        import pyxem.data.dummy_data.make_diffraction_test_data as mdtd
        import pyxem as pxm
        import numpy as np

        di = mdtd.DiffractionTestImage(intensity_noise=False)
        di.add_disk(x=128, y=128, intensity=10.0)  # Add a zero beam disk at the center
        di.add_cubic_disks(vx=20, vy=20, intensity=2.0, n=5)
        di.add_background_lorentz()
        dtd = mdtd.DiffractionTestDataset(10, 10, 256, 256)
        position_array = np.ones((10, 10), dtype=bool)
        position_array[:5] = False
        dtd.add_diffraction_image(di)
        s = dtd.get_signal()
        temp = s.template_match_disk(disk_r=5, subtract_min=False)
        dv = temp.get_diffraction_vectors(
            threshold_abs=0.4, min_distance=5, get_intensity=intensity
        )
        dv.subpixel_refine(s, method="center-of-mass")
        if intensity:
            assert dv.data[0, 0].shape[1] == 3
        else:
            assert dv.data[0, 0].shape[1] == 2

    @pytest.mark.parametrize("intensity", (True, False))
    def test_subpixel_refinement_xc(self, intensity):
        import pyxem.data.dummy_data.make_diffraction_test_data as mdtd
        import pyxem as pxm
        import numpy as np

        di = mdtd.DiffractionTestImage(intensity_noise=False)
        di.add_disk(x=128, y=128, intensity=10.0)  # Add a zero beam disk at the center
        di.add_cubic_disks(vx=20, vy=20, intensity=2.0, n=5)
        di.add_background_lorentz()
        dtd = mdtd.DiffractionTestDataset(10, 10, 256, 256)
        position_array = np.ones((10, 10), dtype=bool)
        position_array[:5] = False
        dtd.add_diffraction_image(di)
        s = dtd.get_signal()
        temp = s.template_match_disk(disk_r=5, subtract_min=False)
        dv = temp.get_diffraction_vectors(
            threshold_abs=0.4, min_distance=5, get_intensity=intensity
        )
        dv.subpixel_refine(s, method="cross-correlation", upsample_factor=2, disk_r=5)
        if intensity:
            assert dv.data[0, 0].shape[1] == 3
        else:
            assert dv.data[0, 0].shape[1] == 2


class TestConvertVectors:
    @pytest.mark.parametrize("real_units", (True, False))
    def test_flatten_vectors(self, diffraction_vectors_map, real_units):
        vectors = diffraction_vectors_map.flatten_diffraction_vectors(
            real_units=real_units
        )
        assert isinstance(vectors, DiffractionVectors2D)
        assert vectors.data.shape == (32, 4)
        assert vectors.ragged == False

    def test_flatten_vectors1d(self, diffraction_vectors_map):
        oned_vectors = diffraction_vectors_map.ivec[0]
        vectors = oned_vectors.flatten_diffraction_vectors()

    def test_get_navigation_positions(self):
        test_data = np.empty((2, 3), dtype=object)
        for i in np.ndindex(test_data.shape):
            test_data[i] = np.random.random((np.random.randint(2, 6), 2))
        dv = DiffractionVectors(test_data)
        dv.axes_manager[0].name = "x"
        dv.axes_manager[1].name = "y"
        dv.axes_manager[0].scale = 1
        dv.axes_manager[1].scale = 2
        dv.axes_manager[0].offset = 1
        flat = dv._get_navigation_positions(flatten=False, real_units=True)
        np.testing.assert_array_equal(flat[0, 0, 0], 0)
        np.testing.assert_array_equal(flat[0, 0, 1], 1)
        np.testing.assert_array_equal(flat[0, 1, 1], 2)
        np.testing.assert_array_equal(flat[0, 1, 0], 0)

    def test_flatten_vectors_with_scales(self):
        test_data = np.empty((2, 3, 4), dtype=object)
        for i in np.ndindex(test_data.shape):
            test_data[i] = np.random.random((np.random.randint(2, 6), 2))
        dv = DiffractionVectors(test_data)
        dv.axes_manager[0].name = "x"
        dv.axes_manager[1].name = "y"
        dv.axes_manager[2].name = "time"
        dv.axes_manager[0].scale = 1
        dv.axes_manager[1].scale = 2
        dv.axes_manager[2].scale = 3
        dv.axes_manager[0].offset = 1
        dv.axes_manager[1].offset = 2
        dv.axes_manager[2].offset = 3
        flat = dv.flatten_diffraction_vectors(real_units=True)
        assert flat.column_names[0] == "x"
        assert flat.column_names[1] == "y"
        assert flat.column_names[2] == "time"
        assert (
            np.max(flat.data[:, 0])
            == dv.axes_manager[0].size * dv.axes_manager[0].scale
        )
        assert (
            np.max(flat.data[:, 1])
            == dv.axes_manager[1].size * dv.axes_manager[1].scale
        )
        assert (
            np.max(flat.data[:, 2])
            == dv.axes_manager[2].size * dv.axes_manager[2].scale
        )
        assert np.min(flat.data[:, 0]) == dv.axes_manager[0].offset
        assert np.min(flat.data[:, 1]) == dv.axes_manager[1].offset
        assert np.min(flat.data[:, 2]) == dv.axes_manager[2].offset

    def test_flatten_vectors_with_set_metadata(self, diffraction_vectors_map):
        diffraction_vectors_map.scales = [0.1, 0.1]
        diffraction_vectors_map.offsets = [1, 1]
        vectors = diffraction_vectors_map.flatten_diffraction_vectors(real_units=True)

        assert isinstance(vectors, DiffractionVectors2D)
        assert vectors.data.shape == (32, 4)


class TestMagnitudes:
    def test_get_magnitudes_map(self, diffraction_vectors_map):
        diffraction_vectors_map.get_magnitudes()

    @pytest.mark.filterwarnings("ignore::FutureWarning")  # deemed "safe enough"
    def test_get_magnitude_histogram_map(self, diffraction_vectors_map):
        diffraction_vectors_map.get_magnitude_histogram(bins=np.arange(0, 0.5, 0.1))


class TestUniqueVectors:
    def test_get_unique_vectors_map_type(self, diffraction_vectors_map):
        unique_vectors = diffraction_vectors_map.get_unique_vectors()
        assert isinstance(unique_vectors, DiffractionVectors2D)

    @pytest.mark.parametrize(
        "distance_threshold, answer",
        [
            (
                0.01,
                np.array(
                    [
                        [-0.165419, 0.241153],
                        [-0.117587, 0.113601],
                        [-0.069755, -0.009965],
                        [-0.069755, 0.257097],
                        [-0.037867, 0.129545],
                        [0.001993, 0.001993],
                        [0.017937, 0.277027],
                        [0.049825, 0.149475],
                        [0.073741, 0.013951],
                        [0.089685, 0.292971],
                        [0.141503, 0.025909],
                        [0.149475, 0.065769],
                        [0.229195, 0.045839],
                    ]
                ),
            ),
            (
                0.1,
                np.array(
                    [
                        [-0.117587, 0.249125],
                        [-0.077727, 0.121573],
                        [-0.021923, -0.001993],
                        [0.053811, 0.284999],
                        [0.049825, 0.149475],
                        [0.121573, 0.03520967],
                        [0.229195, 0.045839],
                    ]
                ),
            ),
        ],
    )
    def test_get_unique_vectors_map_values(
        self, diffraction_vectors_map, distance_threshold, answer
    ):
        unique_vectors = diffraction_vectors_map.get_unique_vectors(
            distance_threshold=distance_threshold
        )
        np.testing.assert_almost_equal(unique_vectors.data, answer)

    def test_get_unique_vectors_map_dbscan(self, diffraction_vectors_map):
        unique_dbscan = diffraction_vectors_map.get_unique_vectors(
            method="DBSCAN", return_clusters=True
        )
        assert isinstance(unique_dbscan[0], DiffractionVectors2D)
        assert isinstance(unique_dbscan[1], DBSCAN)

    @pytest.mark.parametrize(
        "distance_threshold, answer",
        [
            (
                0.01,
                np.array(
                    [
                        [-0.165419, 0.241153],
                        [-0.117587, 0.113601],
                        [-0.069755, -0.009965],
                        [-0.069755, 0.257097],
                        [-0.037867, 0.129545],
                        [0.001993, 0.001993],
                        [0.017937, 0.277027],
                        [0.049825, 0.149475],
                        [0.073741, 0.013951],
                        [0.089685, 0.292971],
                        [0.141503, 0.025909],
                        [0.149475, 0.065769],
                        [0.229195, 0.045839],
                    ]
                ),
            ),
            (
                0.1,
                np.array(
                    [
                        [-0.031888, 0.267062],
                        [-0.03520967, 0.13087367],
                        [0.10200536, 0.02699609],
                    ]
                ),
            ),
        ],
    )
    def test_get_unique_vectors_map_values_dbscan(
        self, diffraction_vectors_map, distance_threshold, answer
    ):
        unique_vectors = diffraction_vectors_map.get_unique_vectors(
            distance_threshold=distance_threshold, method="DBSCAN"
        )
        np.testing.assert_almost_equal(unique_vectors.data, answer)


class TestFilterVectors:
    def test_filter_magnitude_map_type(self, diffraction_vectors_map):
        filtered_vectors = diffraction_vectors_map.filter_magnitude(0.1, 1.0)
        assert isinstance(filtered_vectors, DiffractionVectors)

    def test_filter_magnitude_map(self, diffraction_vectors_map):
        filtered_vectors = diffraction_vectors_map.filter_magnitude(0.1, 1.0)
        ans = np.array(
            [
                [0.089685, 0.292971],
                [0.017937, 0.277027],
                [-0.069755, 0.257097],
                [-0.165419, 0.241153],
                [0.049825, 0.149475],
                [-0.037867, 0.129545],
                [-0.117587, 0.113601],
                [0.149475, 0.065769],
                [0.229195, 0.045839],
                [0.141503, 0.025909],
            ]
        )
        np.testing.assert_almost_equal(filtered_vectors.data[0][1], ans)

    def test_filter_detector_edge_map_type(self, diffraction_vectors_map):
        diffraction_vectors_map.detector_shape = (260, 240)
        diffraction_vectors_map.pixel_calibration = 0.001
        filtered_vectors = diffraction_vectors_map.filter_detector_edge(exclude_width=2)
        assert isinstance(filtered_vectors, DiffractionVectors)

    def test_filter_detector_edge_map(self, diffraction_vectors_map):
        vectors = diffraction_vectors_map.deepcopy()
        vectors.detector_shape = (260, 240)
        vectors.pixel_calibration = 0.001

        filtered_vectors = vectors.filter_detector_edge(exclude_width=2)
        ans = np.array([[-0.117587, 0.113601]])
        np.testing.assert_almost_equal(filtered_vectors.data[0, 0], ans)

    def test_filter_basis(self):
        basis = np.array(
            [
                [0.089685, 0.292971],
                [0.017937, 0.277027],
                [-0.069755, 0.257097],
                [-0.165419, 0.241153],
                [0.049825, 0.149475],
                [-0.037867, 0.129545],
                [-0.117587, 0.113601],
            ]
        )
        arr = np.empty(2, dtype=object)
        arr[0] = basis
        arr[1] = basis[:-1]
        vect = DiffractionVectors(arr)
        filtered = vect.filter_basis(basis=basis, distance=0.1)
        assert isinstance(filtered, DiffractionVectors2D)
        np.testing.assert_almost_equal(filtered.data[0], basis)

    def test_filter_basis_ragged(self, diffraction_vectors_map):
        basis = diffraction_vectors_map.deepcopy()
        filtered = diffraction_vectors_map.filter_basis(basis=basis, distance=0.1)
        assert isinstance(filtered, DiffractionVectors)
        np.testing.assert_almost_equal(
            filtered.data[0, 0], diffraction_vectors_map.data[0, 0]
        )


class TestDiffractingPixelsMap:
    def test_get_dpm_values(self, diffraction_vectors_map):
        answer = np.array([[7.0, 13.0], [11.0, 1.0]])
        xim = diffraction_vectors_map.get_diffracting_pixels_map()
        assert np.allclose(xim, answer)

    def test_get_dpm_type(self, diffraction_vectors_map):
        xim = diffraction_vectors_map.get_diffracting_pixels_map()
        assert isinstance(xim, Signal2D)

    def test_get_dpm_title(self, diffraction_vectors_map):
        xim = diffraction_vectors_map.get_diffracting_pixels_map()
        assert xim.metadata.General.title == "Diffracting Pixels Map"

    def test_get_dpm_in_range(self, diffraction_vectors_map):
        answer = np.array([[0.0, 3.0], [1.0, 1.0]])
        xim = diffraction_vectors_map.get_diffracting_pixels_map(in_range=(0, 0.1))
        assert np.allclose(xim, answer)

    def test_get_dpm_binary(self, diffraction_vectors_map):
        answer = np.array([[1.0, 1.0], [1.0, 1.0]])
        xim = diffraction_vectors_map.get_diffracting_pixels_map(binary=True)
        assert np.allclose(xim, answer)


class TestSlicingVectors:
    @pytest.fixture()
    def vectors(self):
        vectors = np.empty((2, 2), dtype=object)
        vectors[0, 0] = np.random.randint(-100, 100, (20, 2))
        vectors[0, 1] = np.random.randint(-100, 100, (6, 2))
        vectors[1, 0] = np.random.randint(-100, 100, (7, 2))
        vectors[1, 1] = np.random.randint(-100, 100, (1, 2))
        v = DiffractionVectors(
            vectors,
            scales=[0.1, 0.2],
            offsets=[10, 20],
            column_names=["x_axis", "y_axis"],
        )

        return v

    def test_repr(self, vectors):
        repr = vectors._repr_html_()
        assert isinstance(repr, str)

    def test_center(self, vectors):
        np.testing.assert_almost_equal(vectors.center, (100, 100))

    @pytest.mark.parametrize("index", (0, "x_axis", ("x_axis",)))
    def test_column(self, vectors, index):
        slic = vectors.ivec[index]
        for i in np.ndindex((2, 2)):
            np.testing.assert_almost_equal(slic.data[i][:], vectors.data[i][:, 0])

    def test_column_error(self, vectors):
        with pytest.raises(ValueError):
            vectors.ivec[5.5]

    @pytest.mark.parametrize("index", ([0, 1], ["x_axis", "y_axis"]))
    def test_column_slicing2(self, vectors, index):
        slic = vectors.ivec[index]
        for i in np.ndindex((2, 2)):
            np.testing.assert_almost_equal(slic.data[i][:, [0, 1]], vectors.data[i])

    @pytest.mark.parametrize("item", [0, "x_axis"])
    def test_slice(self, vectors, item):
        sliced = vectors.ivec[item]
        assert sliced.column_names == [
            "x_axis",
        ]

    def test_row_lt(self, vectors):
        col = vectors.ivec[0] < 0.5
        slic = vectors.ivec[:, col]
        for i in np.ndindex((2, 2)):
            assert np.all(slic.data[i][:, 0] < 0.5)

    def test_row_gt(self, vectors):
        col = vectors.ivec[0] > 0.5
        slic = vectors.ivec[:, col]
        for i in np.ndindex((2, 2)):
            assert np.all(slic.data[i][:, 0] > 0.5)

    def test_row_gt_none(self, vectors):
        vectors = vectors.as_lazy()
        col = vectors.ivec[0] > 101
        assert vectors.data[0, 0].compute().shape == (20, 2)
        slic = vectors.ivec[:, col]
        for i in np.ndindex((2, 2)):
            assert slic.data[i].compute().shape == (0, 2)

    def test_row_gte(self, vectors):
        col = vectors.ivec[0] >= 0.5
        assert col.data[0, 0].shape == (20,)
        slic = vectors.ivec[:, col]
        for i in np.ndindex((2, 2)):
            assert np.all(slic.data[i][:, 0] >= 0.5)

    def test_row_lte(self, vectors):
        col = vectors.ivec[0] <= 0.5
        slic = vectors.ivec[:, col]
        for i in np.ndindex((2, 2)):
            assert np.all(slic.data[i][:, 0] <= 0.5)

    def test_vector_slicing_error(self, vectors):
        with pytest.raises(ValueError):
            vectors.ivec[0, 0, 0]

    def test_slicing(self, vectors):
        sliced = vectors.ivec[:, vectors.ivec[0] > 101]
        assert sliced.data[0, 0].shape == (0, 2)

    def test_slicing_first(self, vectors):
        with pytest.raises(ValueError):
            sliced = vectors.ivec[:, 0]

    def test_slicing_lazy(self, vectors):
        vectors = vectors.as_lazy()
        sliced = vectors.ivec[:, vectors.ivec[0] > 101]
        sliced = sliced.ivec[:, sliced.ivec[0] > 101]
        sliced.compute()
        assert sliced.data[0, 0].shape == (0, 2)
        sliced.flatten_diffraction_vectors()

    def test_ragged_vectors(self, vectors):
        vectors = vectors.as_lazy()
        assert vectors.ragged == True

    def test_inav_slicing(self, vectors):
        slic = vectors.inav[0, 0]
        assert isinstance(slic, DiffractionVectors)
        assert slic.data.shape == (1,)

    def test_num_columns(self, vectors):
        assert vectors.num_columns == 2
        lazy_vectors = vectors.as_lazy()
        assert lazy_vectors.num_columns == 2


class TestMask:
    def test_remove_peaks(self):
        s = tilt_boundary_data()
        s.calibration.center = None
        temp = s.template_match_disk(disk_r=5, subtract_min=False)  # Just right
        vectors = temp.get_diffraction_vectors(threshold_abs=0.4, min_distance=5)
        mask = vectors.to_mask(disk_r=12)
        masked_s = s * ~mask
        assert np.max(masked_s.data) < 2
