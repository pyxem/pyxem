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

import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest

from pyxem import TORCH_INSTALLED
from pyxem.signals import (
    Diffraction2D,
    DiffractionVectors,
    LazyDiffraction2D,
)
from pyxem.signals.diffraction_vectors import LazyDiffractionVectors
from pyxem.utils import _spot_finding as spots

skip_torch = pytest.mark.skipif(not TORCH_INSTALLED, reason="torch is required")

PATTERN_SIZE = 64
LATTICE = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * 16.0


def spot_positions(row, column):
    """Spots that drift by a subpixel amount from one scan position to the next."""
    return PATTERN_SIZE / 2 + LATTICE + [0.13 * row + 0.21, 0.17 * column - 0.31]


def gaussian_pattern(centres, sigma=1.5, amplitude=500.0, background=5.0, rng=None):
    rows, columns = np.mgrid[:PATTERN_SIZE, :PATTERN_SIZE]
    pattern = np.full((PATTERN_SIZE, PATTERN_SIZE), background)
    for row, column in centres:
        pattern += amplitude * np.exp(
            -((rows - row) ** 2 + (columns - column) ** 2) / (2 * sigma**2)
        )
    if rng is not None:
        pattern = rng.poisson(pattern)
    return pattern.astype(np.float32)


def disk_pattern(centres, radius=4.0, amplitude=300.0, rng=None):
    rows, columns = np.mgrid[:PATTERN_SIZE, :PATTERN_SIZE]
    pattern = np.full((PATTERN_SIZE, PATTERN_SIZE), 5.0)
    for row, column in centres:
        pattern += amplitude * (
            (rows - row) ** 2 + (columns - column) ** 2 <= radius**2
        )
    if rng is not None:
        pattern = rng.poisson(pattern)
    return pattern.astype(np.float32)


def make_dataset(pattern_function, navigation_shape=(4, 5), noise=False):
    rng = np.random.default_rng(0) if noise else None
    data = np.empty(navigation_shape + (PATTERN_SIZE, PATTERN_SIZE), np.float32)
    for position in np.ndindex(navigation_shape):
        data[position] = pattern_function(spot_positions(*position), rng=rng)
    return Diffraction2D(data)


def nearest_distances(found, truth):
    distances = np.hypot(*(found[:, None, :2] - truth[None]).transpose(2, 0, 1))
    return distances.min(axis=1), distances.min(axis=0)


def assert_vectors_equal(first, second):
    assert first.data.shape == second.data.shape
    for position in np.ndindex(first.data.shape):
        np.testing.assert_allclose(
            first.data[position], second.data[position], atol=1e-4
        )


@pytest.fixture
def gaussian_signal():
    return make_dataset(gaussian_pattern)


@pytest.fixture
def disk_signal():
    return make_dataset(disk_pattern)


class TestSingleFrameDetectors:
    def test_dog_subpixel_accuracy(self):
        truth = spot_positions(1, 2)
        response, peaks = spots.find_spots_dog(gaussian_pattern(truth))
        assert response.shape == (PATTERN_SIZE, PATTERN_SIZE)
        to_truth, from_truth = nearest_distances(peaks, truth)
        assert len(peaks) == len(truth)
        assert from_truth.max() < 0.1

    def test_nxcorr_subpixel_accuracy(self):
        truth = spot_positions(3, 1)
        _, peaks = spots.find_spots_nxcorr(disk_pattern(truth), disk_radius=4)
        to_truth, from_truth = nearest_distances(peaks, truth)
        assert len(peaks) == len(truth)
        assert from_truth.max() < 0.15

    def test_subpixel_off_returns_integer_positions(self):
        _, peaks = spots.find_spots_dog(
            gaussian_pattern(spot_positions(0, 0)), subpixel=False
        )
        np.testing.assert_array_equal(peaks[:, :2], np.round(peaks[:, :2]))

    def test_intensity_is_the_raw_disk_mean(self):
        pattern = gaussian_pattern(spot_positions(0, 0))
        _, peaks = spots.find_spots_dog(pattern)
        expected = spots.disk_mean_intensity(pattern, peaks[:, 0], peaks[:, 1], 2)
        np.testing.assert_allclose(peaks[:, 2], expected)
        assert np.all(peaks[:, 2] > 5.0)

    @pytest.mark.parametrize("method", ["dog", "nxcorr"])
    def test_mask_rejects_spots_without_editing_the_pattern(self, method):
        truth = spot_positions(0, 0)
        pattern = gaussian_pattern(truth)
        mask = np.zeros(pattern.shape, bool)
        mask[:24, :24] = True
        if method == "dog":
            _, peaks = spots.find_spots_dog(pattern, mask=mask)
        else:
            _, peaks = spots.find_spots_nxcorr(pattern, disk_radius=2, mask=mask)
        assert len(peaks) == len(truth) - 1
        assert not np.any((peaks[:, 0] < 24) & (peaks[:, 1] < 24))

    def test_blank_pattern_finds_nothing(self):
        blank = np.full((PATTERN_SIZE, PATTERN_SIZE), 7.0, np.float32)
        assert len(spots.find_spots_dog(blank)[1]) == 0
        assert len(spots.find_spots_nxcorr(blank)[1]) == 0

    def test_detect_beam_stop(self):
        mean_pattern = np.full((PATTERN_SIZE, PATTERN_SIZE), 100.0)
        mean_pattern[28:36, 28:36] = 1.0
        mask = spots.detect_beam_stop(mean_pattern, dilation=2)
        assert mask[32, 32] and mask[26, 26] and not mask[24, 24]
        assert spots.detect_beam_stop(np.full((8, 8), 3.0)) is None

    def test_estimate_disk_radius(self):
        radius = spots.estimate_disk_radius(
            disk_pattern(spot_positions(0, 0), radius=5)
        )
        assert 3 <= radius <= 8


class TestFindVectors:
    @pytest.mark.parametrize(
        "method, signal_fixture, kwargs, tolerance",
        [
            ("dog", "gaussian_signal", {}, 0.1),
            # A hard-edged disk on a pixel grid biases the parabola slightly.
            ("nxcorr", "disk_signal", {"disk_radius": 4, "threshold": 0.7}, 0.25),
        ],
    )
    def test_recovers_every_spot(
        self, method, signal_fixture, kwargs, tolerance, request
    ):
        signal = request.getfixturevalue(signal_fixture)
        signal.axes_manager.signal_axes[0].offset = -PATTERN_SIZE / 2
        signal.axes_manager.signal_axes[1].offset = -PATTERN_SIZE / 2
        vectors = signal.find_vectors(method=method, **kwargs)
        assert isinstance(vectors, DiffractionVectors)
        assert vectors.data.shape == (4, 5)
        assert vectors.column_names[-1] == "intensity"
        for position in np.ndindex(4, 5):
            found = vectors.data[position][:, :2] + PATTERN_SIZE / 2
            _, from_truth = nearest_distances(found, spot_positions(*position))
            assert from_truth.max() < tolerance

    def test_calibration_applies_to_vectors(self, gaussian_signal):
        for axis in gaussian_signal.axes_manager.signal_axes:
            axis.scale = 0.02
            axis.offset = -PATTERN_SIZE / 2 * 0.02
            axis.units = "1/nm"
        vectors = gaussian_signal.find_vectors()
        pixels = gaussian_signal.find_vectors(center=(0, 0), calibration=(1, 1)).data[
            0, 0
        ]
        np.testing.assert_allclose(
            vectors.data[0, 0][:, :2],
            (pixels[:, :2] - PATTERN_SIZE / 2) * 0.02,
            atol=1e-5,
        )
        assert vectors.units[:2] == ["1/nm", "1/nm"]

    def test_parameters_are_recorded(self, gaussian_signal):
        vectors = gaussian_signal.find_vectors(method="dog", threshold=8)
        assert vectors.metadata.Peaks.parameters.method == "dog"
        assert vectors.metadata.Peaks.parameters.threshold == 8

    def test_equal_peak_counts_stay_ragged(self, gaussian_signal):
        # Every position has the same number of spots (pyxem#802): the result
        # must remain an object array, not collapse into a regular array.
        vectors = gaussian_signal.find_vectors()
        counts = {len(peaks) for peaks in vectors.data.ravel()}
        assert counts == {len(LATTICE)}
        assert vectors.data.dtype == object
        assert vectors.data.shape == (4, 5)
        assert vectors.data[0, 0].shape == (len(LATTICE), 3)

    def test_one_navigation_dimension(self):
        data = np.stack([gaussian_pattern(spot_positions(0, i)) for i in range(6)])
        vectors = Diffraction2D(data).find_vectors()
        assert vectors.data.shape == (6,)
        assert all(len(peaks) == len(LATTICE) for peaks in vectors.data)

    @pytest.mark.parametrize("method", ["dog", "nxcorr"])
    @pytest.mark.parametrize("navigation_sigma", [0.0, 1.0])
    def test_lazy_matches_eager(self, gaussian_signal, method, navigation_sigma):
        eager = gaussian_signal.find_vectors(
            method=method, navigation_sigma=navigation_sigma
        )
        # Small chunks so the navigation blur has to reach across seams.
        lazy_signal = LazyDiffraction2D(
            da.from_array(gaussian_signal.data, chunks=(3, 3, -1, -1))
        )
        lazy = lazy_signal.find_vectors(
            method=method, navigation_sigma=navigation_sigma
        )
        assert isinstance(lazy, LazyDiffractionVectors)
        lazy.compute()
        assert_vectors_equal(eager, lazy)

    def test_navigation_blur_matches_a_global_blur(self, gaussian_signal):
        from scipy.ndimage import gaussian_filter

        noisy = make_dataset(gaussian_pattern, noise=True)
        blurred = Diffraction2D(
            gaussian_filter(noisy.data, (1.0, 1.0, 0, 0), mode="reflect", truncate=3.0)
        )
        assert_vectors_equal(
            noisy.find_vectors(navigation_sigma=1.0), blurred.find_vectors()
        )

    def test_lazy_never_computes_the_full_dataset(self, gaussian_signal):
        full_shape = gaussian_signal.data.shape
        original = da.Array.compute

        def guarded(array, *args, **kwargs):
            if array.shape == full_shape:
                raise AssertionError("the full dataset was computed")
            return original(array, *args, **kwargs)

        lazy_signal = LazyDiffraction2D(
            da.from_array(gaussian_signal.data, chunks=(2, 2, -1, -1))
        )
        with mock.patch.object(da.Array, "compute", guarded):
            vectors = lazy_signal.find_vectors(navigation_sigma=1.0, mask="auto")
            vectors.compute()
        assert vectors.data.shape == (4, 5)

    def test_signal_axes_split_across_chunks_are_merged(self, gaussian_signal):
        lazy_signal = LazyDiffraction2D(
            da.from_array(gaussian_signal.data, chunks=(2, 2, 16, 16))
        )
        vectors = lazy_signal.find_vectors()
        vectors.compute()
        assert_vectors_equal(gaussian_signal.find_vectors(), vectors)

    def test_mask_array(self, gaussian_signal):
        mask = np.zeros((PATTERN_SIZE, PATTERN_SIZE), bool)
        mask[:, : PATTERN_SIZE // 2 - 8] = True
        vectors = gaussian_signal.find_vectors(mask=mask)
        assert all(len(peaks) == 6 for peaks in vectors.data.ravel())

    def test_automatic_mask_finds_a_beam_stop(self):
        signal = make_dataset(gaussian_pattern)
        signal.data[..., 26:38, 26:38] = 0.0
        vectors = signal.find_vectors(mask="auto", mask_dilation=2)
        for peaks in vectors.data.ravel():
            assert len(peaks) == len(LATTICE) - 1

    def test_bad_arguments(self, gaussian_signal):
        with pytest.raises(ValueError, match="method"):
            gaussian_signal.find_vectors(method="magic")
        with pytest.raises(ValueError, match="mask"):
            gaussian_signal.find_vectors(mask="sometimes")
        with pytest.raises(ValueError, match="mask shape"):
            gaussian_signal.find_vectors(mask=np.zeros((3, 3), bool))
        with pytest.raises(ValueError, match="neighbour_refine"):
            gaussian_signal.find_vectors(method="dog", neighbour_refine=True)

    def test_automatic_disk_radius(self, disk_signal):
        vectors = disk_signal.find_vectors(
            method="nxcorr", disk_radius="auto", threshold=0.7
        )
        assert 3 <= vectors.metadata.Peaks.parameters.disk_radius <= 8


@pytest.fixture
def random_checkpoint(tmp_path):
    import torch

    from pyxem.utils._neural_spots import clear_model_cache
    from pyxem.utils._neural_spots.unet import SpotUNet

    torch.manual_seed(0)
    model = SpotUNet(in_ch=1, base=4, levels=2)
    # Make the untrained heat map fire on bright pixels so there is output.
    torch.nn.init.constant_(model.head_hm.bias, 0.0)
    path = tmp_path / "spotunet.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {"base": 4, "in_ch": 1, "levels": 2},
        },
        path,
    )
    clear_model_cache()
    yield str(path)
    clear_model_cache()


@skip_torch
class TestNeural:
    def test_runs_eager_and_lazy_alike(self, gaussian_signal, random_checkpoint):
        settings = dict(method="neural", model_path=random_checkpoint, device="cpu")
        eager = gaussian_signal.find_vectors(**settings)
        assert isinstance(eager, DiffractionVectors)
        lazy = LazyDiffraction2D(
            da.from_array(gaussian_signal.data, chunks=(2, 3, -1, -1))
        ).find_vectors(**settings)
        lazy.compute()
        assert_vectors_equal(eager, lazy)
        assert eager.metadata.Peaks.parameters.background_sigma is not None

    def test_flat_checkpoint_layout(self, gaussian_signal, tmp_path):
        import torch

        from pyxem.utils._neural_spots import clear_model_cache
        from pyxem.utils._neural_spots.unet import SpotUNet

        model = SpotUNet(in_ch=1, base=4, levels=1)
        path = tmp_path / "flat.pt"
        torch.save(
            {"state_dict": model.state_dict(), "base": 4, "in_ch": 1, "levels": 1}, path
        )
        clear_model_cache()
        vectors = gaussian_signal.find_vectors(
            method="neural", model_path=str(path), background_sigma=None
        )
        assert vectors.data.shape == (4, 5)

    def test_batch_matches_single_pattern(self, random_checkpoint):
        from pyxem.utils._neural_spots import get_model
        from pyxem.utils._neural_spots.inference import detect, detect_batch

        model, device = get_model(random_checkpoint, device="cpu")
        patterns = np.stack([gaussian_pattern(spot_positions(i, 0)) for i in range(3)])
        batch = detect_batch(model, patterns, device)
        for pattern, peaks in zip(patterns, batch):
            np.testing.assert_allclose(detect(model, pattern, device), peaks)

    def test_neighbour_refine(self, gaussian_signal, random_checkpoint):
        vectors = gaussian_signal.find_vectors(
            method="neural", model_path=random_checkpoint, neighbour_refine=True
        )
        assert vectors.data.shape == (4, 5)

    def test_missing_model_explains_how_to_get_one(self, gaussian_signal):
        import emdatabase.data

        with mock.patch.object(
            emdatabase.data, "SpotUNetDiffraction", None, create=True
        ):
            with pytest.raises(ImportError, match="model_path"):
                gaussian_signal.find_vectors(method="neural")
        with pytest.raises(FileNotFoundError):
            gaussian_signal.find_vectors(method="neural", model_path="missing.pt")


@skip_torch
@pytest.mark.skipif(
    not os.environ.get("PYXEM_SPOTUNET_WEIGHTS"),
    reason="set PYXEM_SPOTUNET_WEIGHTS to a trained SpotUNet checkpoint",
)
def test_trained_weights_localise_spots():
    """With real weights every planted spot is found to a fraction of a pixel."""
    truth = PATTERN_SIZE / 2 + LATTICE + [0.37, -0.21]
    signal = Diffraction2D(
        np.stack(
            [gaussian_pattern(truth, rng=np.random.default_rng(i)) for i in range(3)]
        )
    )
    vectors = signal.find_vectors(
        method="neural",
        model_path=os.environ["PYXEM_SPOTUNET_WEIGHTS"],
        center=(0, 0),
        calibration=(1, 1),
    )
    for peaks in vectors.data:
        _, from_truth = nearest_distances(peaks, truth)
        assert from_truth.max() < 0.75


class TestInteractive:
    @pytest.fixture
    def explorer(self, gaussian_signal):
        pytest.importorskip("anyplotlib")
        pytest.importorskip("ipywidgets")
        return gaussian_signal.find_vectors(interactive=True)

    def test_circles_follow_the_crosshair(self, explorer):
        assert explorer.position == (2, 2)
        assert len(explorer.peaks) == len(LATTICE)
        explorer.crosshair.set(cx=4, cy=1)
        assert explorer.position == (1, 4)
        _, from_truth = nearest_distances(explorer.peaks, spot_positions(1, 4))
        assert from_truth.max() < 0.15
        offsets = np.asarray(explorer.spot_markers._data["offsets"])
        np.testing.assert_allclose(offsets, explorer.peaks[:, [1, 0]], atol=1e-4)

    def test_method_switch_resets_threshold(self, explorer):
        explorer.method_control.value = "nxcorr"
        assert explorer.finder.method == "nxcorr"
        assert explorer.threshold_control.value == 0.5
        assert explorer.disk_radius_control.layout.display is None
        assert explorer.sigma1_control.layout.display == "none"
        explorer.method_control.value = "dog"
        assert explorer.threshold_control.value == 10.0

    def test_controls_update_the_finder(self, explorer):
        explorer.threshold_control.value = 1000.0 / 100
        explorer.sigma2_control.value = 3.0
        assert explorer.finder.sigma2 == 3.0
        explorer.response_control.value = True
        assert explorer.response_layer.visible

    def test_preview_matches_the_batch(self, explorer):
        explorer.navigation_sigma_control.value = 1.0
        explorer.crosshair.set(cx=0, cy=3)
        vectors = explorer.compute()
        batch_peaks = vectors.data[3, 0]
        preview = explorer.signal.find_vectors(
            navigation_sigma=1.0, center=(0, 0), calibration=(1, 1)
        ).data[3, 0]
        np.testing.assert_allclose(explorer.peaks, preview, atol=1e-4)
        assert len(batch_peaks) == len(preview)

    def test_mask_toggle_and_parameters(self, explorer):
        explorer.signal.data[..., 26:38, 26:38] = 0.0
        explorer.set_position(explorer.position)
        explorer.mask_control.value = True
        assert explorer.mask is not None
        assert len(explorer.peaks) == len(LATTICE) - 1
        parameters = explorer.parameters
        assert parameters["mask"] == "auto"
        assert parameters["method"] == "dog"

    def test_one_navigation_dimension(self):
        pytest.importorskip("anyplotlib")
        pytest.importorskip("ipywidgets")
        data = np.stack([gaussian_pattern(spot_positions(0, i)) for i in range(6)])
        explorer = Diffraction2D(data).find_vectors(interactive=True)
        explorer.navigator_line.set(x=1)
        assert explorer.position == (1,)
        assert isinstance(explorer.compute(), DiffractionVectors)
