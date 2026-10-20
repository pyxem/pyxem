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

"""The parameter panel of the find-vectors GUI, styled after SpyDE's Find
Vectors wizard: compact two-column rows of label, slider and editable number,
checkboxes in rows, the rarely used beam-stop settings behind an "Extra"
disclosure, and a status line.

The colors are SpyDE's (Catppuccin Mocha) on a dark notebook and the matching
light palette (Catppuccin Latte) on a light one. The theme is read from the
notebook itself (JupyterLab's ``data-jp-theme-light``, VS Code's body class),
falling back to the system preference.
"""

import pathlib

import anywidget
import traitlets

_HERE = pathlib.Path(__file__).parent


class FindVectorsPanel(anywidget.AnyWidget):
    """The controls of :class:`~pyxem.signal_tools.DiffractionVectorFinder`.

    Each setting is a synced trait of the same name as the tool's, linked to
    it by the GUI builder; ``compute_clicks`` and ``close_clicks`` count
    presses of the two buttons.
    """

    _esm = _HERE / "_panel.js"
    _css = _HERE / "_panel.css"

    title = traitlets.Unicode("Find Diffraction Vectors").tag(sync=True)
    hint = traitlets.Unicode(
        "Move the crosshair on the navigator to preview the found spots."
    ).tag(sync=True)
    models = traitlets.List(traitlets.Unicode()).tag(sync=True)
    model = traitlets.Unicode("").tag(sync=True)
    threshold = traitlets.Float(0.3).tag(sync=True)
    min_distance = traitlets.Int(4).tag(sync=True)
    spot_radius = traitlets.Float(0.0).tag(sync=True)
    navigation_sigma = traitlets.Float(0.0).tag(sync=True)
    refine_centers = traitlets.Bool(True).tag(sync=True)
    friedel_partner = traitlets.Bool(True).tag(sync=True)
    neighbor_refine = traitlets.Bool(False).tag(sync=True)
    mask_beam_stop = traitlets.Bool(False).tag(sync=True)
    mask_dilation = traitlets.Int(5).tag(sync=True)
    show_response = traitlets.Bool(False).tag(sync=True)
    extra_open = traitlets.Bool(False).tag(sync=True)
    neighbor_refine_available = traitlets.Bool(True).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    busy = traitlets.Bool(False).tag(sync=True)
    compute_clicks = traitlets.Int(0).tag(sync=True)
    close_clicks = traitlets.Int(0).tag(sync=True)
