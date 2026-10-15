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

"""get_diffraction_vectors(method="neural"), its interactive GUI, and the
pass-through of every other method to hyperspy's find_peaks."""

import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from hyperspy.signals import Signal2D

from pyxem import TORCH_INSTALLED
from pyxem.signals import Diffraction2D, DiffractionVectors, LazyDiffraction2D
from pyxem.signals.diffraction_vectors import LazyDiffractionVectors
from pyxem.utils import _find_vectors

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


def make_signal(navigation_shape=(4, 5), noise=False):
    rng = np.random.default_rng(0) if noise else None
    data = np.empty(navigation_shape + (PATTERN_SIZE, PATTERN_SIZE), np.float32)
    for position in np.ndindex(navigation_shape):
        data[position] = gaussian_pattern(spot_positions(*position), rng=rng)
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
def signal():
    return make_signal()


@pytest.fixture
def random_checkpoint(tmp_path):
    """An untrained SpotUNet whose heat map fires on bright pixels, so the
    whole pipeline runs without downloading weights."""
    import torch

    from pyxem.utils._neural_spots import clear_model_cache
    from pyxem.utils._neural_spots.unet import SpotUNet

    torch.manual_seed(0)
    model = SpotUNet(in_ch=1, base=4, levels=2)
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


class TestHyperspyMethodsPassThrough:
    @pytest.mark.parametrize(
        "method, kwargs",
        [
            ("local_max", {}),
            ("local_max", {"threshold_abs": 100, "min_distance": 3}),
            ("difference_of_gaussian", {"threshold": 0.1}),
            ("template_matching", {"template": np.ones((5, 5)), "threshold": 0.4}),
        ],
    )
    def test_same_as_find_peaks(self, signal, method, kwargs):
        expected = DiffractionVectors.from_peaks(
            Signal2D.find_peaks(
                signal, method=method, interactive=False, get_intensity=True, **kwargs
            )
        )
        found = signal.get_diffraction_vectors(method=method, **kwargs)
        assert_vectors_equal(expected, found)

    def test_default_is_local_max(self, signal):
        assert_vectors_equal(
            signal.get_diffraction_vectors(),
            signal.get_diffraction_vectors(method="local_max"),
        )

    def test_interactive_needs_the_neural_method(self, signal):
        with pytest.raises(ValueError, match="find_peaks"):
            signal.get_diffraction_vectors(method="local_max", interactive=True)


@skip_torch
class TestNeural:
    def test_runs_eager_and_lazy_alike(self, signal, random_checkpoint):
        eager = signal.get_diffraction_vectors(
            method="neural", model_path=random_checkpoint, device="cpu"
        )
        assert isinstance(eager, DiffractionVectors)
        assert eager.data.shape == (4, 5)
        assert eager.column_names[-1] == "intensity"
        assert eager.metadata.Peaks.parameters.method == "neural"
        lazy = LazyDiffraction2D(
            da.from_array(signal.data, chunks=(2, 3, -1, -1))
        ).get_diffraction_vectors(
            method="neural", model_path=random_checkpoint, device="cpu"
        )
        assert isinstance(lazy, LazyDiffractionVectors)
        lazy.compute()
        assert_vectors_equal(eager, lazy)

    def test_intensity_is_the_raw_disk_mean(self, signal, random_checkpoint):
        vectors = signal.get_diffraction_vectors(
            method="neural",
            model_path=random_checkpoint,
            center=(0, 0),
            calibration=(1, 1),
        )
        peaks = vectors.data[1, 1]
        expected = _find_vectors.disk_mean_intensity(
            signal.data[1, 1], peaks[:, 0], peaks[:, 1], _find_vectors.INTENSITY_RADIUS
        )
        np.testing.assert_allclose(peaks[:, 2], expected, rtol=1e-6)

    def test_get_intensity_false_drops_the_column(self, signal, random_checkpoint):
        vectors = signal.get_diffraction_vectors(
            method="neural", model_path=random_checkpoint, get_intensity=False
        )
        assert all(peaks.shape[1] == 2 for peaks in vectors.data.ravel())

    def test_lazy_never_computes_the_full_dataset(self, signal, random_checkpoint):
        full_shape = signal.data.shape
        original = da.Array.compute

        def guarded(array, *args, **kwargs):
            if array.shape == full_shape:
                raise AssertionError("the full dataset was computed")
            return original(array, *args, **kwargs)

        lazy_signal = LazyDiffraction2D(
            da.from_array(signal.data, chunks=(2, 2, -1, -1))
        )
        with mock.patch.object(da.Array, "compute", guarded):
            vectors = lazy_signal.get_diffraction_vectors(
                method="neural",
                model_path=random_checkpoint,
                navigation_sigma=1.0,
                mask="auto",
            )
            vectors.compute()
        assert vectors.data.shape == (4, 5)

    def test_navigation_blur_matches_a_global_blur(self, random_checkpoint):
        from scipy.ndimage import gaussian_filter

        noisy = make_signal(noise=True)
        blurred = Diffraction2D(
            gaussian_filter(noisy.data, (1.0, 1.0, 0, 0), mode="reflect", truncate=3.0)
        )
        settings = dict(
            method="neural", model_path=random_checkpoint, background_sigma=12.0
        )
        assert_vectors_equal(
            noisy.get_diffraction_vectors(navigation_sigma=1.0, **settings),
            blurred.get_diffraction_vectors(**settings),
        )

    def test_signal_axes_split_across_chunks_are_merged(
        self, signal, random_checkpoint
    ):
        settings = dict(method="neural", model_path=random_checkpoint)
        lazy = LazyDiffraction2D(
            da.from_array(signal.data, chunks=(2, 2, 16, 16))
        ).get_diffraction_vectors(**settings)
        lazy.compute()
        assert_vectors_equal(signal.get_diffraction_vectors(**settings), lazy)

    def test_equal_peak_counts_stay_ragged(self, random_checkpoint):
        # pyxem#802: equal counts at every position must not collapse the
        # object array into a regular one.
        data = np.stack([gaussian_pattern(spot_positions(0, 0))] * 6).reshape(
            2, 3, PATTERN_SIZE, PATTERN_SIZE
        )
        vectors = Diffraction2D(data).get_diffraction_vectors(
            method="neural", model_path=random_checkpoint
        )
        assert len({len(peaks) for peaks in vectors.data.ravel()}) == 1
        assert vectors.data.dtype == object
        assert vectors.data.shape == (2, 3)

    def test_masks(self, random_checkpoint):
        signal = make_signal()
        signal.data[..., 24:40, 24:40] = 0.0
        settings = dict(method="neural", model_path=random_checkpoint)
        mask = np.zeros((PATTERN_SIZE, PATTERN_SIZE), bool)
        mask[24:40, 24:40] = True
        for vectors in (
            signal.get_diffraction_vectors(mask=mask, **settings),
            signal.get_diffraction_vectors(mask="auto", mask_dilation=2, **settings),
        ):
            # The signal axes have no offset, so the vectors are pixel positions.
            pixels = np.rint(vectors.data[0, 0][:, :2]).astype(int)
            assert len(pixels)
            assert not mask[pixels[:, 0], pixels[:, 1]].any()

    def test_one_navigation_dimension(self, random_checkpoint):
        data = np.stack([gaussian_pattern(spot_positions(0, i)) for i in range(6)])
        vectors = Diffraction2D(data).get_diffraction_vectors(
            method="neural", model_path=random_checkpoint
        )
        assert vectors.data.shape == (6,)

    def test_neighbour_refine(self, signal, random_checkpoint):
        vectors = signal.get_diffraction_vectors(
            method="neural", model_path=random_checkpoint, neighbour_refine=True
        )
        assert vectors.data.shape == (4, 5)

    def test_batch_matches_single_pattern(self, random_checkpoint):
        from pyxem.utils._neural_spots import get_model
        from pyxem.utils._neural_spots.inference import detect, detect_batch

        model, device = get_model(random_checkpoint, device="cpu")
        patterns = np.stack([gaussian_pattern(spot_positions(i, 0)) for i in range(3)])
        for pattern, peaks in zip(patterns, detect_batch(model, patterns, device)):
            np.testing.assert_allclose(detect(model, pattern, device), peaks)

    def test_flat_checkpoint_layout(self, signal, tmp_path):
        import torch

        from pyxem.utils._neural_spots import clear_model_cache
        from pyxem.utils._neural_spots.unet import SpotUNet

        model = SpotUNet(in_ch=1, base=4, levels=1)
        path = tmp_path / "flat.pt"
        torch.save(
            {"state_dict": model.state_dict(), "base": 4, "in_ch": 1, "levels": 1},
            path,
        )
        clear_model_cache()
        vectors = signal.get_diffraction_vectors(
            method="neural", model_path=str(path), background_sigma=None
        )
        assert vectors.data.shape == (4, 5)

    def test_bad_arguments(self, signal, random_checkpoint):
        settings = dict(method="neural", model_path=random_checkpoint)
        with pytest.raises(TypeError, match="threshold_abs"):
            signal.get_diffraction_vectors(threshold_abs=3, **settings)
        with pytest.raises(ValueError, match="mask"):
            signal.get_diffraction_vectors(mask="sometimes", **settings)
        with pytest.raises(ValueError, match="mask shape"):
            signal.get_diffraction_vectors(mask=np.zeros((3, 3), bool), **settings)
        with pytest.raises(ValueError, match="neighbour_refine"):
            Diffraction2D(signal.data[0]).get_diffraction_vectors(
                neighbour_refine=True, **settings
            )

    def test_missing_model_explains_how_to_get_one(self, signal):
        import emdatabase.data

        with mock.patch.object(
            emdatabase.data, "SpotUNetDiffraction", None, create=True
        ):
            with pytest.raises(ImportError, match="model_path"):
                signal.get_diffraction_vectors(method="neural")
        with pytest.raises(FileNotFoundError):
            signal.get_diffraction_vectors(method="neural", model_path="missing.pt")


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
    vectors = signal.get_diffraction_vectors(
        method="neural",
        model_path=os.environ["PYXEM_SPOTUNET_WEIGHTS"],
        center=(0, 0),
        calibration=(1, 1),
    )
    for peaks in vectors.data:
        _, from_truth = nearest_distances(peaks, truth)
        assert from_truth.max() < 0.75


@pytest.fixture
def explorer(signal, random_checkpoint):
    pytest.importorskip("anyplotlib")
    pytest.importorskip("hyperspy_gui_anywidget")
    return signal.get_diffraction_vectors(
        method="neural",
        model_path=random_checkpoint,
        interactive=True,
        display=False,
    )


@skip_torch
class TestInteractive:
    def test_registered_with_hyperspy(self, explorer):
        from hyperspy.ui_registry import UI_REGISTRY

        specs = UI_REGISTRY["pyxem.Diffraction2D.get_diffraction_vectors"]
        assert specs["anywidget"]["function"] == "get_diffraction_vectors_aw"
        assert set(explorer.widgets) == {"anywidget"}
        assert {"widget", "wdict"} <= set(explorer.widgets["anywidget"])

    def test_crosshair_moves_the_pattern(self, explorer, signal):
        wdict = explorer.widgets["anywidget"]["wdict"]
        assert explorer.position == (2, 2)
        wdict["crosshair"].set(cx=4, cy=1)
        assert explorer.position == (1, 4)
        np.testing.assert_array_equal(explorer.pattern, signal.data[1, 4])
        np.testing.assert_array_equal(wdict["pattern_plot"].data, signal.data[1, 4])
        offsets = np.asarray(wdict["spot_markers"]._data["offsets"]).reshape(-1, 2)
        np.testing.assert_allclose(offsets, explorer.peaks[:, [1, 0]], atol=1e-4)

    def test_controls_are_linked_to_the_tool(self, explorer):
        wdict = explorer.widgets["anywidget"]["wdict"]
        wdict["threshold"].value = 0.99
        assert explorer.threshold == 0.99
        assert explorer.finder.threshold == 0.99
        explorer.min_distance = 7
        assert wdict["min_distance"].value == 7
        wdict["show_response"].value = True
        assert wdict["response_layer"].visible

    def test_preview_matches_the_full_search(self, explorer, signal):
        explorer.navigation_sigma = 1.0
        explorer.position = (3, 0)
        vectors = explorer.compute()
        assert explorer.vectors is vectors
        full = signal.get_diffraction_vectors(
            method="neural",
            model_path=explorer.finder.model_path,
            background_sigma=explorer.finder.background_sigma,
            navigation_sigma=1.0,
            center=(0, 0),
            calibration=(1, 1),
        )
        np.testing.assert_allclose(explorer.peaks, full.data[3, 0], atol=1e-4)

    def test_mask_toggle_and_parameters(self, explorer):
        explorer.signal.data[..., 24:40, 24:40] = 0.0
        explorer.position = (0, 0)
        explorer.mask_beam_stop = True
        assert explorer.mask is not None and explorer.mask[32, 32]
        parameters = explorer.parameters
        assert parameters["mask"] == "auto"
        assert parameters["method"] == "neural"
        explorer.mask_dilation = 0
        assert not explorer.mask[23, 23]

    def test_one_navigation_dimension(self, random_checkpoint):
        pytest.importorskip("anyplotlib")
        pytest.importorskip("hyperspy_gui_anywidget")
        data = np.stack([gaussian_pattern(spot_positions(0, i)) for i in range(6)])
        tool = Diffraction2D(data).get_diffraction_vectors(
            method="neural",
            model_path=random_checkpoint,
            interactive=True,
            display=False,
        )
        tool.widgets["anywidget"]["wdict"]["navigator_line"].set(x=1)
        assert tool.position == (1,)
        assert isinstance(tool.compute(), DiffractionVectors)

    def test_old_hyperspy_is_reported(self, signal, random_checkpoint):
        import hyperspy

        with mock.patch.object(hyperspy, "__version__", "2.4.0"):
            with pytest.raises(ImportError, match="hyperspy>=2.5.0"):
                signal.get_diffraction_vectors(
                    method="neural",
                    model_path=random_checkpoint,
                    interactive=True,
                    display=False,
                )
