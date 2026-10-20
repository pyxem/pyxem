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

"""anywidget-toolkit GUIs for pyxem's interactive tools.

Registered in ``pyxem/hyperspy_extension.yaml`` under ``GUI: widgets:
anywidget:``, so ``tool.gui()`` finds them through hyperspy's GUI registry.
Each builder takes the tool as ``obj`` and returns ``{"widget", "wdict"}``;
``add_display_arg`` displays the widget instead when ``display=True``.

The plots are anyplotlib figures; the controls are pyxem's own anywidget
panel styled after SpyDE (:mod:`pyxem.signal_tools._panel`), linked to the
tool's traits.
"""

import numpy as np
from hyperspy_gui_anywidget.utils import add_display_arg
from ipywidgets import HBox, Layout
from link_traits import link

from pyxem.signal_tools._panel import FindVectorsPanel

SPOT_COLOR = "#ff3030"
MASK_COLOR = "#ff8a3d"
CROSSHAIR_COLOR = "#00e5ff"


def _navigator_image(signal, navigation_dimension):
    navigator = getattr(signal, "navigator", None)
    if navigator is not None:
        image = np.asarray(getattr(navigator, "data", navigator))
    else:
        image = signal.data.sum(
            axis=tuple(range(navigation_dimension, signal.data.ndim))
        )
        if hasattr(image, "compute"):
            image = image.compute()
    image = np.asarray(image, dtype=np.float32)
    if navigation_dimension > 2:
        # Browse the first index of the extra (slower) navigation axes.
        image = image.reshape((-1,) + image.shape[-2:])[0]
    return image


def _add_navigator(obj, axes, wdict):
    """Navigator panel with a crosshair (2-D scans) or a line (1-D scans) that
    sets ``obj.position``."""
    dimension = obj._navigation_dimension
    shape = obj.signal.data.shape[:dimension]
    if dimension == 0:
        axes.imshow(np.zeros((1, 1), dtype=np.float32))
        return
    image = _navigator_image(obj.signal, dimension)
    if dimension == 1:
        plot = axes.plot(image)
        line = plot.add_vline_widget(x=float(obj.position[0]), color=CROSSHAIR_COLOR)

        def moved(event):
            obj.position = (int(np.clip(round(line.x), 0, shape[0] - 1)),)

        line.add_event_handler(moved, "pointer_move")
        wdict["navigator_line"] = line
    else:
        plot = axes.imshow(image)
        crosshair = plot.add_crosshair_widget(
            cx=float(obj.position[-1]),
            cy=float(obj.position[-2]),
            color=CROSSHAIR_COLOR,
        )

        def moved(event):
            row = int(np.clip(round(crosshair.cy), 0, shape[-2] - 1))
            column = int(np.clip(round(crosshair.cx), 0, shape[-1] - 1))
            position = tuple(obj.position[:-2]) + (row, column)
            if position != tuple(obj.position):
                obj.position = position

        crosshair.add_event_handler(moved, "pointer_move")
        wdict["crosshair"] = crosshair
    wdict["navigator"] = plot


@add_display_arg
def get_diffraction_vectors_aw(obj, **kwargs):
    """GUI for :class:`pyxem.signal_tools.DiffractionVectorFinder`.

    The navigator with a crosshair beside the pattern under it, the detected
    spots drawn as circles, and the detector's settings as linked controls.
    """
    import anyplotlib as apl

    wdict = {}
    figure, (navigator_axes, pattern_axes) = apl.subplots(1, 2, figsize=(860, 430))
    _add_navigator(obj, navigator_axes, wdict)

    pattern_plot = pattern_axes.imshow(obj.pattern, cmap="gray")
    pattern_plot.set_xlabel("kx (px)")
    pattern_plot.set_ylabel("ky (px)")
    response_layer = pattern_plot.add_layer(
        np.zeros_like(obj.pattern), cmap="viridis", alpha=0.6, visible=False
    )
    spot_markers = pattern_plot.add_circles(
        np.zeros((0, 2)),
        name="spots",
        radius=obj.marker_radius(),
        edgecolors=SPOT_COLOR,
        linewidths=1.5,
    )

    panel = FindVectorsPanel(
        models=obj.model_names(),
        neighbor_refine_available=obj._navigation_dimension == 2,
        extra_open=bool(obj.mask_beam_stop),
    )
    linked = [
        "model",
        "threshold",
        "min_distance",
        "spot_radius",
        "navigation_sigma",
        "refine_centers",
        "friedel_partner",
        "neighbor_refine",
        "mask_beam_stop",
        "mask_dilation",
        "show_response",
    ]
    for name in linked:
        link((obj, name), (panel, name))

    def redraw(event=None):
        if pattern_plot.data.shape != obj.pattern.shape or not np.array_equal(
            pattern_plot.data, obj.pattern
        ):
            pattern_plot.set_data(obj.pattern)
        peaks = obj.peaks
        spot_markers.set(
            offsets=peaks[:, [1, 0]] if len(peaks) else np.zeros((0, 2)),
            radius=obj.marker_radius(),
        )
        if obj.show_response and obj.response is not None:
            response_layer.set_data(obj.response)
            response_layer.set(clim=(obj.threshold, 1.0), visible=True)
        else:
            response_layer.set(visible=False)
        pattern_plot.set_overlay_mask(obj.mask, color=MASK_COLOR, alpha=0.35)
        panel.status = f"<b>{len(peaks)}</b> spots at {tuple(obj.position)}"

    obj.observe(redraw, "detected")
    redraw()

    def on_compute(change):
        panel.busy = True
        panel.status = "Computing…"
        try:
            vectors = obj.compute()
            if vectors._lazy:
                vectors.compute(show_progressbar=False)
            panel.status = f"Computed {vectors.axes_manager.navigation_size} patterns"
        finally:
            panel.busy = False

    def on_close(change):
        obj.observe(redraw, "detected", remove=True)
        widget.close()

    panel.observe(on_compute, names="compute_clicks")
    panel.observe(on_close, names="close_clicks")

    # Two anywidgets side by side in one box; it wraps, so in a narrow
    # notebook the panel drops below the figure instead of being clipped.
    widget = HBox(
        [figure, panel],
        layout=Layout(flex_flow="row wrap", align_items="flex-start", gap="8px"),
    )
    wdict.update(
        figure=figure,
        panel=panel,
        pattern_plot=pattern_plot,
        spot_markers=spot_markers,
        response_layer=response_layer,
    )
    return {"widget": widget, "wdict": wdict}
