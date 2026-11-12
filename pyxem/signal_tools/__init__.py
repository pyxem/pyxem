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

"""Interactive tools: the settings of a pyxem operation held as traits, with a
GUI registered through hyperspy's GUI system (``tool.gui()``).

Each tool is toolkit-agnostic; its widgets are declared per toolkit in
``pyxem/hyperspy_extension.yaml`` under ``GUI: widgets:``.
"""

import importlib.util

from pyxem.signal_tools._diffraction_vector_finder import DiffractionVectorFinder

__all__ = ["DiffractionVectorFinder", "require_anywidget_gui"]


def require_anywidget_gui():
    """Raise a clear error unless the anywidget GUIs can run: hyperspy picks
    the anywidget toolkit by default from 2.5.0, and the widgets need
    anyplotlib and hyperspy_gui_anywidget."""
    import hyperspy
    from packaging.version import Version

    missing = [
        name
        for name in ("anyplotlib", "hyperspy_gui_anywidget", "link_traits")
        if importlib.util.find_spec(name) is None
    ]
    if Version(hyperspy.__version__) < Version("2.5.0"):
        missing.append("hyperspy>=2.5.0")
    if missing:
        raise ImportError(
            f"The interactive GUI needs {', '.join(missing)}: "
            "`pip install pyxem[interactive]`."
        )
