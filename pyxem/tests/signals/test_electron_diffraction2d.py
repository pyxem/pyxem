# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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
import dask.array as da

from hyperspy.signals import Signal2D

from pyxem.signals import (
    ElectronDiffraction1D,
    ElectronDiffraction2D,
    PolarDiffraction2D,
    LazyElectronDiffraction2D,
)


def test_init():
    z = np.zeros((2, 2, 2, 2))
    _ = ElectronDiffraction2D(
        z, metadata={"Acquisition_instrument": {"SEM": "Expensive-SEM"}}
    )


class TestSimpleMaps:
    # Confirms that maps run without error.

    def test_center_direct_beam_cross_correlate(self, diffraction_pattern):
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)
        diffraction_pattern.center_direct_beam(
            method="cross_correlate", radius_start=1, radius_finish=3
        )
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)

    @pytest.mark.skip(reason="Broken downstream in scikit image")
    def test_center_direct_beam_xc_return_shifts(self, diffraction_pattern):
        shifts = diffraction_pattern.center_direct_beam(
            method="cross_correlate",
            radius_start=1,
            radius_finish=3,
            return_shifts=True,
        )
        ans = np.array([[[0.45, 0.45], [-0.57, -0.57]], [[0.45, 0.45], [-0.52, -0.52]]])
        np.testing.assert_almost_equal(shifts, ans)

    def test_center_direct_beam_blur_return_shifts(self, diffraction_pattern):
        shifts = diffraction_pattern.center_direct_beam(
            method="blur",
            sigma=5,
            half_square_width=3,
            return_shifts=True,
            subpixel=False,
        )
        ans = np.array([[[1.0, 1.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])
        np.testing.assert_almost_equal(shifts, ans)

    def test_center_direct_beam_in_small_region(self, diffraction_pattern):
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)
        diffraction_pattern.center_direct_beam(
            method="blur", sigma=int(5), half_square_width=3
        )
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)

    def test_apply_affine_transformation(self, diffraction_pattern):
        diffraction_pattern.apply_affine_transformation(
            D=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)

    def test_apply_affine_transformation_upsample(self, diffraction_pattern):
        old_data = np.copy(diffraction_pattern.data)
        diffraction_pattern.apply_affine_transformation(
            D=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            output_shape=(8, 8),
        )
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)
        assert diffraction_pattern.axes_manager.signal_shape == (8, 8)
        np.testing.assert_almost_equal(old_data, diffraction_pattern.data)

    def test_apply_affine_transforms_paths(self, diffraction_pattern):
        D = np.array([[1.0, 0.9, 0.0], [1.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
        s = Signal2D(np.asarray([[D, D], [D, D]]))
        static = diffraction_pattern.apply_affine_transformation(D, inplace=False)
        dynamic = diffraction_pattern.apply_affine_transformation(s, inplace=False)
        assert np.allclose(static.data, dynamic.data, atol=1e-3)

    def test_apply_affine_transformation_with_casting(self, diffraction_pattern):
        diffraction_pattern.change_dtype("uint8")

        with pytest.warns(
            UserWarning,
            match="Bi-quadratic interpolation behavior has changed due to a bug in the implementation of scikit-image",
        ):
            transformed_dp = ElectronDiffraction2D(
                diffraction_pattern
            ).apply_affine_transformation(
                D=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.2]]),
                order=2,
                keep_dtype=True,
                inplace=False,
            )

        assert transformed_dp.data.dtype == "uint8"


class TestSimpleHyperspy:
    # Tests functions that assign to hyperspy metadata

    def test_set_experimental_parameters(self, diffraction_pattern):
        diffraction_pattern.set_experimental_parameters(
            beam_energy=3.0,
            camera_length=3.0,
            scan_rotation=1.0,
            convergence_angle=1.0,
            rocking_angle=1.0,
            rocking_frequency=1.0,
            exposure_time=1.0,
        )
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)

    def test_set_scan_calibration(self, diffraction_pattern):
        diffraction_pattern.set_scan_calibration(19)
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)

    @pytest.mark.parametrize(
        "calibration, center, pixels",
        [
            (1, (4, 4), True),
            (0.5, (2, 2), False),
            (0.5, None, False),
            (0.5, None, True),
        ],
    )
    def test_set_diffraction_calibration(
        self, diffraction_pattern, calibration, center, pixels
    ):
        pattern_center = np.array(diffraction_pattern.axes_manager.signal_shape) // 2
        diffraction_pattern.set_diffraction_calibration(
            calibration, center=center, pixels=pixels
        )
        dx, dy = diffraction_pattern.axes_manager.signal_axes

        # Check on axes manager values
        assert dx.scale == calibration and dy.scale == calibration

        # Check on data values as well
        if center is not None and pixels:
            assert np.all(
                diffraction_pattern.isig[0.0, 0.0].data
                == diffraction_pattern.isig[center[0], center[1]].data
            )
            assert (
                dx.offset == -center[0] * calibration
                and dy.offset == -center[1] * calibration
            )  # Check that axes manager offset is set correctly
        elif center is not None and not pixels:
            assert np.all(
                diffraction_pattern.isig[0.0, 0.0].data
                == diffraction_pattern.isig[
                    int(center[0] / calibration), int(center[1] / calibration)
                ].data
            )
            assert (
                dx.offset == -center[0] and dy.offset == -center[1]
            )  # Check that axes manager offset is set correctly
        else:
            assert np.all(
                diffraction_pattern.isig[0.0, 0.0].data
                == diffraction_pattern.isig[pattern_center[0], pattern_center[1]].data
            )
            assert (
                dx.offset == -pattern_center[0] * calibration
                and dy.offset == -pattern_center[1] * calibration
            )


class TestGainNormalisation:
    @pytest.mark.parametrize(
        "dark_reference, bright_reference",
        [
            (-1, 1),
            (0, 1),
            (0, 256),
        ],
    )
    def test_apply_gain_normalisation(
        self, diffraction_pattern, dark_reference, bright_reference
    ):
        dpr = diffraction_pattern.apply_gain_normalisation(
            dark_reference=dark_reference,
            bright_reference=bright_reference,
            inplace=False,
        )
        assert dpr.max() == bright_reference
        assert dpr.min() == dark_reference


class TestDirectBeamMethods:
    @pytest.mark.parametrize(
        "mask_expected",
        [
            (
                np.array(
                    [
                        [False, False, False, False, False, False, False, False],
                        [False, False, False, False, False, False, False, False],
                        [False, False, False, True, True, False, False, False],
                        [False, False, True, True, True, True, False, False],
                        [False, False, True, True, True, True, False, False],
                        [False, False, False, True, True, False, False, False],
                        [False, False, False, False, False, False, False, False],
                        [False, False, False, False, False, False, False, False],
                    ]
                ),
            ),
        ],
    )
    def test_get_direct_beam_mask(self, diffraction_pattern, mask_expected):
        mask_calculated = diffraction_pattern.get_direct_beam_mask(2)
        assert isinstance(mask_calculated, Signal2D)
        assert np.equal(mask_calculated, mask_expected)


class TestComputeAndAsLazyElectron2D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyElectronDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert isinstance(s, ElectronDiffraction2D)
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyElectronDiffraction2D(dask_array)
        s.compute()
        assert s.__class__ == ElectronDiffraction2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = ElectronDiffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyElectronDiffraction2D)
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = ElectronDiffraction2D(data)
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyElectronDiffraction2D)
        assert data.shape == s_lazy.data.shape


class TestDecomposition:
    def test_decomposition_class_assignment(self, diffraction_pattern):
        diffraction_pattern.decomposition()
        assert isinstance(diffraction_pattern, ElectronDiffraction2D)


class TestIntegration:
    @pytest.fixture
    def ones(self):
        ones_diff = ElectronDiffraction2D(data=np.ones(shape=(5, 5)))
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_rad"
        return ones_diff

    @pytest.mark.parametrize("energy", [None, 200])
    def test_1d_azimuthal_integration(self, ones, energy):
        ones.beam_energy = energy
        integration = ones.get_azimuthal_integral1d(npt=10)
        assert isinstance(integration, ElectronDiffraction1D)

    @pytest.mark.parametrize("energy", [None, 200])
    def test_2d_azimuthal_integration(self, ones, energy):
        ones.beam_energy = energy
        integration = ones.get_azimuthal_integral2d(npt=10)
        assert isinstance(integration, PolarDiffraction2D)

    def test_set_scan_calibration(self):
        ones = ElectronDiffraction2D(data=np.ones((3, 3, 3, 3)))
        ones.scan_calibration = 0.9
        assert ones.axes_manager.navigation_axes[0].scale == 0.9
        assert ones.axes_manager.navigation_axes[1].scale == 0.9
        assert ones.scan_calibration == 0.9

    def test_set_diffraction_calibration(self):
        ones = ElectronDiffraction2D(data=np.ones((3, 3, 3, 3)))
        ones.diffraction_calibration = 0.9
        assert ones.axes_manager.signal_axes[0].scale == 0.9
        assert ones.axes_manager.signal_axes[1].scale == 0.9
        assert ones.diffraction_calibration == 0.9

    def test_set_camera_length(self, ones):
        assert ones.camera_length is None
        ones.camera_length = 1.5
        assert ones.metadata.Acquisition_instrument.TEM["camera_length"] == 1.5
        assert ones.camera_length == 1.5

    def test_set_beam_energy(self, ones):
        assert ones.beam_energy is None
        ones.beam_energy = 1.5
        assert ones.metadata.Acquisition_instrument.TEM["beam_energy"] == 1.5
        assert ones.beam_energy == 1.5
