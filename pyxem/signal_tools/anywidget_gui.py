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

The plots are anyplotlib figures; the controls are hyperspy_gui_anywidget's
widgets, linked to the tool's traits.
"""

import numpy as np
from hyperspy_gui_anywidget.custom_widgets import (
    ButtonWidget,
    CheckboxWidget,
    ContainerWidget,
    FloatSliderWidget,
    HTMLWidget,
    IntSliderWidget,
)
from hyperspy_gui_anywidget.utils import add_display_arg
from ipywidgets import HBox, Layout
from link_traits import link

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
    figure, (navigator_axes, pattern_axes) = apl.subplots(1, 2, figsize=(900, 440))
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

    controls = {
        "threshold": FloatSliderWidget(
            min=0.0, max=1.0, step=0.01, description="Threshold"
        ),
        "min_distance": IntSliderWidget(min=1, max=30, description="Min distance"),
        "spot_radius": FloatSliderWidget(
            min=0.0, max=30.0, step=0.5, description="Spot size (0 = auto)"
        ),
        "navigation_sigma": FloatSliderWidget(
            min=0.0, max=5.0, step=0.1, description="Navigation blur"
        ),
        "mask_beam_stop": CheckboxWidget(description="Mask beam stop"),
        "mask_dilation": IntSliderWidget(min=0, max=20, description="Beam-stop dilate"),
        "show_response": CheckboxWidget(description="Show response"),
    }
    for name, control in controls.items():
        link((obj, name), (control, "value"))
    wdict.update(controls)

    count = HTMLWidget()
    compute = ButtonWidget(
        description="Compute", tooltip="Find the vectors in every pattern."
    )
    close = ButtonWidget(description="Close", tooltip="Close the widget.")
    wdict.update(count=count, compute=compute, close=close)

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
        count.value = f"<b>{len(peaks)}</b> spots at {tuple(obj.position)}"

    obj.observe(redraw, "detected")
    redraw()

    def on_compute(change):
        count.value = "Computing…"
        vectors = obj.compute()
        count.value = f"Computed: {vectors}"

    def on_close(change):
        obj.unobserve(redraw, "detected")
        if hasattr(widget, "close"):
            widget.close()

    compute.observe(on_compute, names="clicks")
    close.observe(on_close, names="clicks")

    panel = ContainerWidget(
        children=[
            *controls.values(),
            ContainerWidget(children=[compute, close], layout="horizontal"),
            count,
        ],
        layout="vertical",
    )
    # The figure is an anywidget of its own, so the outer row is a plain
    # ipywidgets box rather than a flattened container. It wraps, so in a
    # narrow notebook the controls drop below the figure instead of being
    # clipped off its right edge.
    widget = HBox([figure, panel], layout=Layout(flex_flow="row wrap"))
    wdict.update(
        figure=figure,
        pattern_plot=pattern_plot,
        spot_markers=spot_markers,
        response_layer=response_layer,
    )
    return {"widget": widget, "wdict": wdict}
