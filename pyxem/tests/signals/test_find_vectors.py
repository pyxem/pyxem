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

import importlib.util
import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from hyperspy.signals import Signal2D

from pyxem import ONNXRUNTIME_INSTALLED
from pyxem.signals import Diffraction2D, DiffractionVectors, LazyDiffraction2D
from pyxem.signals.diffraction_vectors import LazyDiffractionVectors
from pyxem.utils import _find_vectors

skip_onnx = pytest.mark.skipif(
    not ONNXRUNTIME_INSTALLED or importlib.util.find_spec("onnx") is None,
    reason="onnxruntime and onnx are required",
)

PATTERN_SIZE = 64
LATTICE = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * 16.0


def spot_positions(row, column):
    """Spots that drift by a subpixel amount from one scan position to the next."""
    return PATTERN_SIZE / 2 + LATTICE + [0.13 * row + 0.21, 0.17 * column - 0.31]


def gaussian_pattern(centers, sigma=1.5, amplitude=500.0, background=5.0, rng=None):
    rows, columns = np.mgrid[:PATTERN_SIZE, :PATTERN_SIZE]
    pattern = np.full((PATTERN_SIZE, PATTERN_SIZE), background)
    for row, column in centers:
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


def _save_graph(path, nodes, outputs, inputs_shape, metadata, initializers=()):
    """A tiny ONNX model built directly with onnx.helper (no torch needed)."""
    from onnx import TensorProto, helper, save

    graph = helper.make_graph(
        nodes,
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, inputs_shape)],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            for name in outputs
        ],
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    save(model, str(path))
    return str(path)


def _constant(name, value):
    from onnx import numpy_helper

    return numpy_helper.from_array(np.asarray(value, np.float32), name)


def save_random_detector(path, mask_head=False):
    """A stand-in detector: heat-map logits ``x - 20`` of the normalized
    pattern, so it fires only on the bright spots; zero offsets; with a mask
    head, mask logits ``x - 10``."""
    from onnx import helper

    nodes = [
        helper.make_node("Sub", ["input", "twenty"], ["heatmap_logits"]),
        helper.make_node("Mul", ["input", "zero"], ["nothing"]),
        helper.make_node("Concat", ["nothing", "nothing"], ["offset"], axis=1),
    ]
    outputs = ["heatmap_logits", "offset"]
    if mask_head:
        nodes.append(helper.make_node("Sub", ["input", "ten"], ["mask_logits"]))
        outputs.append("mask_logits")
    return _save_graph(
        path,
        nodes,
        outputs,
        ["batch", 1, "height", "width"],
        {"levels": 2, "mask_head": int(mask_head), "mad_floor": 0.0, "threshold": 0.3},
        [_constant("twenty", 20.0), _constant("zero", 0.0)]
        + ([_constant("ten", 10.0)] if mask_head else []),
    )


def save_random_center_networks(directory):
    """Stand-in refine (one channel) and Friedel-partner (two channel) center
    networks, under the file names the weights module asks for: coverage
    logits ``8 x - 4`` of the window's first channel and a constant sigma."""
    from onnx import helper

    from pyxem.utils._neural_spots.weights import REFINERS

    for (_, file_name), channels in zip(REFINERS.values(), (1, 2)):
        nodes = [
            helper.make_node(
                "Slice", ["input", "start", "end", "axis"], ["first_channel"]
            ),
            helper.make_node("Mul", ["first_channel", "eight"], ["scaled"]),
            helper.make_node("Sub", ["scaled", "four"], ["coverage_logits"]),
            helper.make_node(
                "ReduceMean", ["input"], ["mean"], axes=[1, 2, 3], keepdims=0
            ),
            helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
            helper.make_node("Add", ["nothing", "log_sigma_value"], ["log_sigma"]),
        ]
        from onnx import numpy_helper

        initializers = [
            numpy_helper.from_array(np.array([0], np.int64), "start"),
            numpy_helper.from_array(np.array([1], np.int64), "end"),
            numpy_helper.from_array(np.array([1], np.int64), "axis"),
            _constant("eight", 8.0),
            _constant("four", 4.0),
            _constant("zero", 0.0),
            _constant("log_sigma_value", np.log(0.05)),
        ]
        _save_graph(
            directory / file_name,
            nodes,
            ["coverage_logits", "log_sigma"],
            ["batch", channels, 36, 36],
            {
                "channels": channels,
                "levels": 2,
                "crop_radius": 10.0,
                "crop_half": 16,
                "sigma_scale": 1.0,
                # wide enough that the steps run on the test disks
                "min_spot_radius": 1.0,
                "max_spot_radius": 64.0,
            },
            initializers,
        )


@pytest.fixture
def random_checkpoint(tmp_path, monkeypatch):
    """Stand-in detector and center networks, so the whole pipeline runs
    without downloading weights. Returns the detector's path."""
    from pyxem.utils._neural_spots import clear_model_cache
    from pyxem.utils._neural_spots.weights import LOCAL_DIRECTORY_VARIABLE

    save_random_center_networks(tmp_path)
    monkeypatch.setenv(LOCAL_DIRECTORY_VARIABLE, str(tmp_path))
    clear_model_cache()
    yield save_random_detector(tmp_path / "spotnet.onnx")
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


def test_gui_registration_adds_no_empty_toolkit():
    """pyxem must not register the anywidget toolkit on HyperSpy's behalf:
    without hyperspy_gui_anywidget, HyperSpy would route its own GUIs (the
    navigation sliders) to it and fail."""
    from hyperspy.ui_registry import TOOLKIT_REGISTRY, UI_REGISTRY

    import pyxem.signal_tools  # noqa: F401  registers the toolkey

    installed = importlib.util.find_spec("hyperspy_gui_anywidget") is not None
    assert ("anywidget" in TOOLKIT_REGISTRY) == installed
    toolkits = UI_REGISTRY["pyxem.Diffraction2D.get_diffraction_vectors"]
    assert ("anywidget" in toolkits) == installed


@skip_onnx
class TestNeural:
    def test_runs_eager_and_lazy_alike(self, signal, random_checkpoint):
        eager = signal.get_diffraction_vectors(
            method="neural", model_path=random_checkpoint, device="cpu"
        )
        assert isinstance(eager, DiffractionVectors)
        assert eager.data.shape == (4, 5)
        assert eager.column_names[2:] == ["intensity", "confidence", "sigma"]
        assert eager.metadata.Peaks.parameters.refine_centers
        assert eager.metadata.Peaks.parameters.friedel_partner
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
        assert all(peaks.shape[1] == 4 for peaks in vectors.data.ravel())
        assert vectors.column_names[2:] == ["confidence", "sigma"]

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

    def test_neighbor_refine(self, signal, random_checkpoint):
        vectors = signal.get_diffraction_vectors(
            method="neural", model_path=random_checkpoint, neighbor_refine=True
        )
        assert vectors.data.shape == (4, 5)

    def test_preview_of_one_pattern_matches_its_stack(self, random_checkpoint):
        """A pattern's result does not depend on the other patterns computed
        with it: the rescale comes from the reference, the center steps cut
        each disk's own window."""
        from pyxem.utils._find_vectors import SpotFinder

        finder = SpotFinder(model_path=random_checkpoint, device="cpu")
        patterns = np.stack([gaussian_pattern(spot_positions(i, 0)) for i in range(3)])
        finder.reference = patterns[1]
        stack = finder.detect_stack(patterns)
        for pattern, peaks in zip(patterns, stack):
            np.testing.assert_allclose(finder.detect(pattern)[1], peaks, rtol=1e-6)

    def test_columns_and_calibration(self, signal, random_checkpoint):
        for axis in signal.axes_manager.signal_axes:
            axis.scale = 0.02
            axis.offset = -PATTERN_SIZE / 2 * 0.02
            axis.units = "1/nm"
        settings = dict(method="neural", model_path=random_checkpoint)
        calibrated = signal.get_diffraction_vectors(**settings).data[1, 2]
        pixels = signal.get_diffraction_vectors(
            center=(0, 0), calibration=(1, 1), **settings
        ).data[1, 2]
        assert calibrated.shape[1] == 5
        np.testing.assert_allclose(
            calibrated[:, :2], (pixels[:, :2] - PATTERN_SIZE / 2) * 0.02, atol=1e-5
        )
        # intensity and confidence are not calibrated; sigma is a length
        np.testing.assert_allclose(calibrated[:, 2:4], pixels[:, 2:4])
        np.testing.assert_allclose(calibrated[:, 4], pixels[:, 4] * 0.02, rtol=1e-5)
        assert np.all((pixels[:, 3] > 0) & (pixels[:, 3] <= 1))

    def test_center_steps(self, signal, random_checkpoint):
        """The center steps keep every detection and move none of them by
        half a spot radius or more."""
        settings = dict(
            method="neural",
            model_path=random_checkpoint,
            center=(0, 0),
            calibration=(1, 1),
        )
        plain = signal.get_diffraction_vectors(refine_centers=False, **settings)
        refined = signal.get_diffraction_vectors(friedel_partner=False, **settings)
        both = signal.get_diffraction_vectors(**settings)
        for position in np.ndindex(4, 5):
            assert len(plain.data[position]) == len(refined.data[position])
            assert len(plain.data[position]) == len(both.data[position])
            for result in (refined, both):
                shift = np.hypot(
                    *(result.data[position][:, :2] - plain.data[position][:, :2]).T
                )
                assert np.all(shift < 32)

    def test_mask_head_detector(self, signal, random_checkpoint, tmp_path):
        path = save_random_detector(tmp_path / "masked.onnx", mask_head=True)
        vectors = signal.get_diffraction_vectors(method="neural", model_path=path)
        assert vectors.data.shape == (4, 5)

    def test_bad_arguments(self, signal, random_checkpoint):
        settings = dict(method="neural", model_path=random_checkpoint)
        with pytest.raises(TypeError, match="threshold_abs"):
            signal.get_diffraction_vectors(threshold_abs=3, **settings)
        with pytest.raises(ValueError, match="mask"):
            signal.get_diffraction_vectors(mask="sometimes", **settings)
        with pytest.raises(ValueError, match="mask shape"):
            signal.get_diffraction_vectors(mask=np.zeros((3, 3), bool), **settings)
        with pytest.raises(ValueError, match="neighbor_refine"):
            Diffraction2D(signal.data[0]).get_diffraction_vectors(
                neighbor_refine=True, **settings
            )

    def test_missing_model_explains_how_to_get_one(self, signal, monkeypatch):
        import emdatabase.data

        from pyxem.utils._neural_spots import clear_model_cache
        from pyxem.utils._neural_spots.weights import LOCAL_DIRECTORY_VARIABLE

        monkeypatch.delenv(LOCAL_DIRECTORY_VARIABLE, raising=False)
        clear_model_cache()

        with mock.patch.object(emdatabase.data, "DE_SpotNet", None, create=True):
            with pytest.raises(ImportError, match="PYXEM_SPOTNET_WEIGHTS_DIR"):
                signal.get_diffraction_vectors(method="neural")
        with pytest.raises(ValueError, match="model must be one of"):
            signal.get_diffraction_vectors(method="neural", model="nonsense")
        with pytest.raises(FileNotFoundError):
            signal.get_diffraction_vectors(method="neural", model_path="missing.pt")


@skip_onnx
@pytest.mark.skipif(
    not os.environ.get("PYXEM_SPOTNET_WEIGHTS_DIR"),
    reason="set PYXEM_SPOTNET_WEIGHTS_DIR to a directory of the trained weights",
)
@pytest.mark.parametrize("model", ["spotnet", "dynamical"])
def test_trained_weights_localize_disks(model):
    """With the trained detector and center networks every planted disk
    (radius 8, inside the range both center steps work on) is found to a
    small fraction of a pixel, confidently, with a small sigma."""
    from pyxem.utils._neural_spots import clear_model_cache

    clear_model_cache()
    size = 128
    rows, columns = np.mgrid[:size, :size]
    truth = (
        size / 2
        + np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * 30
        + [0.37, -0.21]
    )
    patterns = []
    for seed in range(4):
        pattern = np.full((size, size), 3.0)
        for row, column in truth:
            pattern += 150 * np.clip(8.5 - np.hypot(rows - row, columns - column), 0, 1)
        patterns.append(np.random.default_rng(seed).poisson(pattern))
    vectors = Diffraction2D(
        np.stack(patterns).astype(np.float32)
    ).get_diffraction_vectors(
        method="neural", model=model, center=(0, 0), calibration=(1, 1)
    )
    for peaks in vectors.data:
        to_truth, from_truth = nearest_distances(peaks, truth)
        assert from_truth.max() < 0.1
        planted = to_truth < 1
        assert planted.sum() == len(truth)
        assert np.all(peaks[planted, 3] > 0.5)
        assert np.all(np.isfinite(peaks[planted, 4]) & (peaks[planted, 4] < 0.5))
        # anything else found ranks as less certain than every planted disk
        if (~planted).any():
            assert peaks[~planted, 4].min() > peaks[planted, 4].max()


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


@skip_onnx
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

    def test_panel_is_linked_to_the_tool(self, explorer):
        panel = explorer.widgets["anywidget"]["wdict"]["panel"]
        panel.threshold = 0.99
        assert explorer.threshold == 0.99
        assert explorer.finder.threshold == 0.99
        explorer.min_distance = 7
        assert panel.min_distance == 7
        panel.show_response = True
        assert explorer.widgets["anywidget"]["wdict"]["response_layer"].visible
        panel.friedel_partner = False
        assert not explorer.finder.friedel_partner
        panel.neighbor_refine = True
        assert explorer.parameters["neighbor_refine"]
        assert panel.status.endswith(f"spots at {tuple(explorer.position)}")
        assert panel.neighbor_refine_available

    def test_panel_lists_the_model_and_switches_it(self, explorer, tmp_path):
        panel = explorer.widgets["anywidget"]["wdict"]["panel"]
        # a model_path checkpoint is the only model offered
        assert panel.models == ["spotnet.onnx"]
        assert panel.model == "spotnet.onnx"
        detections = len(explorer.peaks)
        panel.model = "spotnet.onnx"
        assert len(explorer.peaks) == detections

    def test_panel_buttons(self, explorer):
        wdict = explorer.widgets["anywidget"]["wdict"]
        wdict["panel"].compute_clicks += 1
        assert explorer.vectors is not None
        assert not wdict["panel"].busy
        assert wdict["panel"].status.startswith("Computed 20 patterns")
        wdict["panel"].close_clicks += 1

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
