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

"""Interactive tuning of :meth:`Diffraction2D.find_vectors` in a notebook.

The explorer shows the navigator with a crosshair beside the pattern under it,
with the spots the current settings find drawn as circles. Every control
re-runs the detector on that one pattern, so a setting can be judged in the
time it takes to move a slider; :meth:`FindVectorsExplorer.compute` then runs
the whole dataset with the settings as left.

The layout and controls follow the Find Vectors action in SpyDE
(https://github.com/directelectron/spyde). Requires ``anyplotlib`` and
``ipywidgets``.
"""

import contextlib
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from pyxem.utils import _spot_finding as spots
from pyxem.utils._find_vectors import automatic_mask

SPOT_COLOR = "#ff3030"
MASK_COLOR = "#ff8a3d"
THRESHOLD_RANGE = {
    spots.METHOD_NXCORR: (0.0, 1.0, 0.01),
    spots.METHOD_DOG: (0.0, 30.0, 0.1),
    spots.METHOD_NEURAL: (0.0, 1.0, 0.01),
}


def _require_interactive_dependencies():
    try:
        import anyplotlib  # noqa: F401
        import ipywidgets  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "find_vectors(interactive=True) needs anyplotlib and ipywidgets: "
            "`pip install pyxem[interactive]`."
        ) from error


def _plain_function(method):
    """anyplotlib tags its handlers with an attribute, which a bound method
    cannot carry."""

    def handler(event):
        method(event)

    return handler


class FindVectorsExplorer:
    """A notebook widget for tuning the spot finder on single patterns.

    Created by ``signal.find_vectors(..., interactive=True)``. Display it in a
    notebook cell, move the crosshair and the controls until the circles sit
    on the spots, then call :meth:`compute`.

    Attributes
    ----------
    figure : anyplotlib.Figure
        Navigator and pattern panels.
    controls : ipywidgets.VBox
        The parameter controls.
    widget : ipywidgets.HBox
        The figure and the controls together; what a notebook displays.
    """

    def __init__(
        self,
        signal,
        finder,
        mask=None,
        navigation_sigma=0.0,
        neighbour_refine=False,
        center=None,
        calibration=None,
        column_names=None,
        units=None,
    ):
        _require_interactive_dependencies()
        import anyplotlib as apl

        self.signal = signal
        self.finder = finder
        self.navigation_sigma = float(navigation_sigma)
        self.neighbour_refine = neighbour_refine
        self._output_settings = dict(
            center=center,
            calibration=calibration,
            column_names=column_names,
            units=units,
        )
        self._navigation_dimension = signal.axes_manager.navigation_dimension
        self._user_mask = mask
        self._detected_stop = None
        self.mask_enabled = mask is not None
        self.mask_dilation = spots.DEFAULT_MASK_DILATION
        self.show_response = False
        self.position = tuple(
            size // 2 for size in signal.data.shape[: self._navigation_dimension]
        )
        self.peaks = spots.empty_peaks()

        self.figure, axes = apl.subplots(1, 2, figsize=(900, 440))
        navigator_axes, pattern_axes = axes
        self._build_navigator(navigator_axes)
        self.pattern = self._read_pattern()
        self.pattern_plot = pattern_axes.imshow(self.pattern, cmap="gray")
        self.pattern_plot.set_xlabel("kx (px)")
        self.pattern_plot.set_ylabel("ky (px)")
        self.response_layer = self.pattern_plot.add_layer(
            np.zeros_like(self.pattern), cmap="viridis", alpha=0.6, visible=False
        )
        self.spot_markers = self.pattern_plot.add_circles(
            np.zeros((0, 2)),
            name="spots",
            radius=finder.marker_radius(),
            edgecolors=SPOT_COLOR,
            linewidths=1.5,
        )
        self._build_controls()
        self.update()

    # ── display ──────────────────────────────────────────────────────────

    def _ipython_display_(self):
        from IPython.display import display

        display(self.widget)

    def _repr_mimebundle_(self, **kwargs):
        return self.widget._repr_mimebundle_(**kwargs)

    # ── navigator ───────────────────────────────────────────────────────

    def _navigator_image(self):
        navigator = getattr(self.signal, "navigator", None)
        if navigator is not None:
            image = np.asarray(
                navigator.data if hasattr(navigator, "data") else navigator
            )
        else:
            signal_axes = tuple(
                range(self._navigation_dimension, self.signal.data.ndim)
            )
            image = self.signal.data.sum(axis=signal_axes)
            if hasattr(image, "compute"):
                image = image.compute()
        return np.asarray(image, dtype=np.float32)

    def _build_navigator(self, axes):
        self.crosshair = None
        self.navigator_line = None
        if self._navigation_dimension == 0:
            axes.imshow(np.zeros((1, 1), dtype=np.float32))
            return
        image = self._navigator_image()
        if self._navigation_dimension == 1:
            self.navigator_plot = axes.plot(image)
            self.navigator_line = self.navigator_plot.add_vline_widget(
                x=float(self.position[0]), color="#00e5ff"
            )
            self.navigator_line.add_event_handler(
                _plain_function(self._on_navigator_moved), "pointer_move"
            )
            return
        if self._navigation_dimension > 2:
            # Collapse the extra (slower) navigation axes: browse their first index.
            image = image.reshape((-1,) + image.shape[-2:])[0]
        self.navigator_plot = axes.imshow(image)
        self.crosshair = self.navigator_plot.add_crosshair_widget(
            cx=float(self.position[-1]), cy=float(self.position[-2]), color="#00e5ff"
        )
        self.crosshair.add_event_handler(
            _plain_function(self._on_navigator_moved), "pointer_move"
        )

    def _on_navigator_moved(self, event=None):
        shape = self.signal.data.shape[: self._navigation_dimension]
        if self.crosshair is not None:
            row = int(np.clip(round(self.crosshair.cy), 0, shape[-2] - 1))
            column = int(np.clip(round(self.crosshair.cx), 0, shape[-1] - 1))
            position = tuple(self.position[:-2]) + (row, column)
        else:
            position = (int(np.clip(round(self.navigator_line.x), 0, shape[0] - 1)),)
        if position != self.position:
            self.set_position(position)

    def set_position(self, position):
        """Show the pattern at navigation index ``position`` (array order)."""
        self.position = tuple(int(i) for i in position)
        self.pattern = self._read_pattern()
        self.pattern_plot.set_data(self.pattern)
        self.update()

    def _read_pattern(self):
        """The pattern at the current position, navigation-blurred like the batch.

        Only the neighbourhood the blur reaches is read, so a lazy signal
        computes a handful of patterns rather than a chunk.
        """
        data = self.signal.data
        if self._navigation_dimension == 0:
            pattern = data
        elif self.navigation_sigma <= 0:
            pattern = data[self.position]
        else:
            reach = int(math.ceil(3 * self.navigation_sigma))
            window = []
            for axis, index in enumerate(self.position):
                size = data.shape[axis]
                offsets = np.arange(index - reach, index + reach + 1)
                # Half-sample symmetric reflection, as the batch pads its chunks.
                offsets = np.where(offsets < 0, -offsets - 1, offsets)
                offsets = np.where(offsets >= size, 2 * size - offsets - 1, offsets)
                window.append(np.clip(offsets, 0, size - 1))
            # Read the bounding slab once, then pick the reflected indices.
            slab = data[tuple(slice(w.min(), w.max() + 1) for w in window)]
            if hasattr(slab, "compute"):
                slab = slab.compute()
            neighbourhood = np.asarray(slab)[np.ix_(*[w - w.min() for w in window])]
            sigma = (self.navigation_sigma,) * self._navigation_dimension + (0, 0)
            blurred = gaussian_filter(
                neighbourhood.astype(np.float32),
                sigma=sigma,
                mode="reflect",
                truncate=3.0,
            )
            pattern = blurred[(reach,) * self._navigation_dimension]
        if hasattr(pattern, "compute"):
            pattern = pattern.compute()
        return np.asarray(pattern, dtype=np.float32)

    # ── mask ────────────────────────────────────────────────────────────

    @property
    def mask(self):
        """The beam-stop mask the detector is currently using, or None."""
        if not self.mask_enabled:
            return None
        if self._user_mask is not None:
            return self._user_mask
        if self._detected_stop is None:
            stop = automatic_mask(
                self.signal.data, self._navigation_dimension, dilation=0
            )
            self._detected_stop = stop if stop is not None else False
        if self._detected_stop is False:
            return None
        return spots.dilate_mask(self._detected_stop, self.mask_dilation)

    # ── controls ────────────────────────────────────────────────────────

    def _build_controls(self):
        import ipywidgets as widgets

        finder = self.finder
        style = {"description_width": "110px"}

        def layout():
            # One per control: a shared Layout would hide every control at once.
            return widgets.Layout(width="320px")

        def slider(kind, description, value, minimum, maximum, step):
            return kind(
                value=value,
                min=minimum,
                max=maximum,
                step=step,
                description=description,
                continuous_update=True,
                style=style,
                layout=layout(),
            )

        from pyxem import TORCH_INSTALLED

        low, high, step = THRESHOLD_RANGE[finder.method]
        methods = [("NXCORR", "nxcorr"), ("DoG", "dog")]
        if TORCH_INSTALLED:
            methods.insert(0, ("Neural", "neural"))
        self.method_control = widgets.Dropdown(
            options=methods,
            value=finder.method,
            description="Method",
            style=style,
            layout=layout(),
        )
        self.threshold_control = slider(
            widgets.FloatSlider, "Threshold", finder.threshold, low, high, step
        )
        self.min_distance_control = slider(
            widgets.IntSlider, "Min distance", finder.min_distance, 1, 30, 1
        )
        self.disk_radius_control = slider(
            widgets.IntSlider, "Disk radius", finder.disk_radius, 1, 30, 1
        )
        self.sigma1_control = slider(
            widgets.FloatSlider, "DoG σ1", finder.sigma1, 0.4, 3.0, 0.1
        )
        self.sigma2_control = slider(
            widgets.FloatSlider, "DoG σ2", finder.sigma2, 1.0, 6.0, 0.1
        )
        self.spot_radius_control = slider(
            widgets.FloatSlider,
            "Spot size (0=auto)",
            finder.spot_radius or 0.0,
            0.0,
            30.0,
            0.5,
        )
        self.navigation_sigma_control = slider(
            widgets.FloatSlider, "Nav blur σ", self.navigation_sigma, 0.0, 5.0, 0.1
        )
        self.mask_control = widgets.Checkbox(
            value=self.mask_enabled,
            description="Mask beam stop",
            style=style,
            layout=layout(),
        )
        self.mask_dilation_control = slider(
            widgets.IntSlider, "Beam-stop dilate", self.mask_dilation, 0, 20, 1
        )
        self.response_control = widgets.Checkbox(
            value=False, description="Show transform", style=style, layout=layout()
        )
        self.subpixel_control = widgets.Checkbox(
            value=finder.subpixel, description="Subpixel", style=style, layout=layout()
        )
        self.count_label = widgets.HTML()

        for control in (
            self.threshold_control,
            self.min_distance_control,
            self.disk_radius_control,
            self.sigma1_control,
            self.sigma2_control,
            self.spot_radius_control,
            self.navigation_sigma_control,
            self.mask_control,
            self.mask_dilation_control,
            self.response_control,
            self.subpixel_control,
        ):
            control.observe(self._on_control_changed, names="value")
        self.method_control.observe(self._on_method_changed, names="value")

        self.controls = widgets.VBox(
            [
                self.method_control,
                self.threshold_control,
                self.min_distance_control,
                self.disk_radius_control,
                self.sigma1_control,
                self.sigma2_control,
                self.spot_radius_control,
                self.navigation_sigma_control,
                self.mask_control,
                self.mask_dilation_control,
                self.response_control,
                self.subpixel_control,
                self.count_label,
            ]
        )
        self.widget = widgets.HBox([self.figure, self.controls])
        self._show_method_controls()

    def _show_method_controls(self):
        method = self.finder.method
        visible = {
            self.disk_radius_control: method == spots.METHOD_NXCORR,
            self.sigma1_control: method == spots.METHOD_DOG,
            self.sigma2_control: method == spots.METHOD_DOG,
            self.spot_radius_control: method == spots.METHOD_NEURAL,
            self.min_distance_control: method != spots.METHOD_NEURAL,
            self.subpixel_control: method != spots.METHOD_NEURAL,
        }
        for control, show in visible.items():
            control.layout.display = None if show else "none"

    def _on_method_changed(self, change):
        """Switching method resets the threshold: the scales are unrelated."""
        method = change["new"]
        self.finder.method = method
        low, high, step = THRESHOLD_RANGE[method]
        with self.threshold_control.hold_trait_notifications():
            self.threshold_control.min, self.threshold_control.max = low, high
            self.threshold_control.step = step
        with self._paused():
            self.threshold_control.value = spots.DEFAULT_THRESHOLD[method]
            self.min_distance_control.value = spots.DEFAULT_MIN_DISTANCE[method]
        self._show_method_controls()
        self._on_control_changed()

    @contextlib.contextmanager
    def _paused(self):
        """Set several controls without redrawing after each one."""
        self._updating = True
        try:
            yield
        finally:
            self._updating = False

    def _on_control_changed(self, change=None):
        if getattr(self, "_updating", False):
            return
        finder = self.finder
        finder.threshold = float(self.threshold_control.value)
        finder.min_distance = int(self.min_distance_control.value)
        finder.disk_radius = int(self.disk_radius_control.value)
        finder.sigma1 = float(self.sigma1_control.value)
        finder.sigma2 = float(self.sigma2_control.value)
        finder.spot_radius = float(self.spot_radius_control.value) or None
        finder.subpixel = bool(self.subpixel_control.value)
        self.mask_enabled = bool(self.mask_control.value)
        self.mask_dilation = int(self.mask_dilation_control.value)
        self.show_response = bool(self.response_control.value)
        navigation_sigma = float(self.navigation_sigma_control.value)
        if navigation_sigma != self.navigation_sigma:
            self.navigation_sigma = navigation_sigma
            self.pattern = self._read_pattern()
            self.pattern_plot.set_data(self.pattern)
        self.update()

    # ── detection ───────────────────────────────────────────────────────

    def update(self):
        """Re-run the detector on the displayed pattern and redraw."""
        mask = self.mask
        response, self.peaks = self.finder.detect(self.pattern, mask=mask)
        # Markers take (x, y) = (column, row).
        self.spot_markers.set(
            offsets=self.peaks[:, [1, 0]] if len(self.peaks) else np.zeros((0, 2)),
            radius=self.finder.marker_radius(),
        )
        if self.show_response:
            low = self.finder.threshold
            high = (
                float(np.percentile(response, 99))
                if self.finder.method == spots.METHOD_DOG
                else 1.0
            )
            self.response_layer.set_data(response)
            self.response_layer.set(clim=(low, max(high, low + 1e-6)), visible=True)
        else:
            self.response_layer.set(visible=False)
        self.pattern_plot.set_overlay_mask(mask, color=MASK_COLOR, alpha=0.35)
        self.count_label.value = f"<b>{len(self.peaks)}</b> spots at {self.position}"

    @property
    def parameters(self):
        """The current settings, as keyword arguments for ``find_vectors``."""
        parameters = dict(self.finder.parameters)
        parameters.pop("background_sigma", None)
        parameters["navigation_sigma"] = self.navigation_sigma
        parameters["mask"] = (
            (self._user_mask if self._user_mask is not None else "auto")
            if self.mask_enabled
            else None
        )
        parameters["mask_dilation"] = self.mask_dilation
        return parameters

    def compute(self):
        """Find the vectors in every pattern with the current settings.

        Returns
        -------
        DiffractionVectors
            Lazy if the signal is lazy.
        """
        return self.signal._vectors_from_finder(
            self.finder,
            mask=self.mask,
            navigation_sigma=self.navigation_sigma,
            neighbour_refine=self.neighbour_refine,
            **self._output_settings,
        )
