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

"""Locating and caching the spot-detector weights.

The trained weights are not shipped with pyxem. They are distributed through
em-database (https://github.com/electronmicroscopy/emdatabase) as the
``SpotUNetDiffraction`` entry, which pins each release by date and checks it
on download. A local checkpoint can always be passed instead.
"""

import threading
from pathlib import Path

#: The em-database entry holding the spot-detector weights.
EMDATABASE_ENTRY = "SpotUNetDiffraction"

_cache = {}
_cache_lock = threading.Lock()


def resolve_model_path(model_path=None, version=None):
    """Return a local checkpoint path, downloading it from em-database if needed.

    Parameters
    ----------
    model_path : str or pathlib.Path, optional
        A local checkpoint. When given, nothing is downloaded.
    version : str, optional
        The dated em-database version to fetch (e.g. ``"260716"``). ``None``
        fetches the latest.
    """
    if model_path is not None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No spot-detector checkpoint at {path}")
        return str(path)
    try:
        import emdatabase.data
    except ImportError as error:
        raise ImportError(
            "The neural spot detector downloads its weights with em-database. "
            "Install it with `pip install em-database` (or `pip install "
            "pyxem[neural]`), or pass a local checkpoint with `model_path=`."
        ) from error
    entry = getattr(emdatabase.data, EMDATABASE_ENTRY, None)
    if entry is None:
        raise ImportError(
            f"The installed em-database has no `{EMDATABASE_ENTRY}` entry yet. "
            "Upgrade em-database, or pass a local checkpoint with `model_path=`."
        )
    return str(entry().download(version=version, background=False))


def get_model(model_path=None, version=None, device=None):
    """Load (once per process) and return ``(model, device)``.

    Models are cached by checkpoint path and device, so a lazy computation
    loads the network once rather than once per chunk.
    """
    from .inference import build_model, default_device, load_checkpoint
    import torch

    path = resolve_model_path(model_path, version)
    device = torch.device(device) if device is not None else default_device()
    key = (path, str(device))
    with _cache_lock:
        if key not in _cache:
            _cache[key] = build_model(load_checkpoint(path), device)
        return _cache[key]


def clear_model_cache():
    with _cache_lock:
        _cache.clear()
