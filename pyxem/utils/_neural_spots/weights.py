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

"""Where the trained weights come from, and loading them once per process.

The weights are not shipped with pyxem. Each model is its own em-database
entry (https://github.com/electronmicroscopy/emdatabase), which pins it by
dated version and checks it against its md5 on download. Every entry and
file name the neural spot finder uses is in this module and nowhere else.

A local checkpoint can always be used instead: ``model_path=`` for the
detector, or a directory holding the files below named by the
``PYXEM_SPOTNET_WEIGHTS_DIR`` environment variable, which is looked in first.
"""

import os
import threading
from pathlib import Path

#: Environment variable naming a local directory of weight files.
LOCAL_DIRECTORY_VARIABLE = "PYXEM_SPOTNET_WEIGHTS_DIR"

#: The DE SpotNet detectors (``get_diffraction_vectors(method="neural",
#: model=...)``): the em-database entry and the file each one is.
DETECTORS = {
    "spotnet": ("DE_SpotNet", "de_spotnet.onnx"),
    "dynamical": ("DE_SpotNet_Dynamical", "de_spotnet_dynamical.onnx"),
    "production": ("DE_SpotNet_Production", "de_spotnet_production.onnx"),
    "base16": ("DE_SpotNet_Base16", "de_spotnet_base16.onnx"),
}
#: The default detector, SpyDE's too.
DEFAULT_DETECTOR = "spotnet"

#: The center steps, DE CenterNet (``refine_centers``) and DE FriedelNet
#: (``friedel_partner``), by the ids ``center_refine`` asks for.
REFINERS = {
    "DE_CenterNet": ("DE_CenterNet", "de_centernet.onnx"),
    "DE_FriedelNet": ("DE_FriedelNet", "de_friedelnet.onnx"),
}

_detector_cache = {}
_refiner_cache = {}
_cache_lock = threading.Lock()


def _weights_file(entry_name, file_name, version=None):
    """A local path to ``file_name``: from the local directory when it is
    there, else the md5-checked download of the em-database entry
    ``entry_name``, which must publish exactly that file."""
    directory = os.environ.get(LOCAL_DIRECTORY_VARIABLE)
    if directory and (Path(directory) / file_name).is_file():
        return str(Path(directory) / file_name)
    try:
        import emdatabase.data
    except ImportError as error:
        raise ImportError(
            "The neural spot finder downloads its weights with em-database. "
            "Install it with `pip install em-database` (or `pip install "
            "pyxem[neural]`), or put the weight files in a directory named by "
            f"{LOCAL_DIRECTORY_VARIABLE}."
        ) from error
    entry_class = getattr(emdatabase.data, entry_name, None)
    if entry_class is None:
        raise ImportError(
            f"The installed em-database has no `{entry_name}` entry yet. Upgrade "
            f"em-database, or put {file_name} in a directory named by "
            f"{LOCAL_DIRECTORY_VARIABLE}."
        )
    entry = entry_class()
    if entry.file != file_name:
        raise ValueError(
            f"em-database entry `{entry_name}` publishes {entry.file}, not "
            f"{file_name}; this pyxem expects {file_name}."
        )
    return str(entry.download(version=version, background=False))


def detector_path(model=None, model_path=None, version=None):
    """Local ``.onnx`` file of a detector: ``model_path`` when given, else the
    weights of ``model`` (a key of :data:`DETECTORS`)."""
    if model_path is not None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"No spot-detector model at {path}")
        return str(path)
    model = model or DEFAULT_DETECTOR
    if model not in DETECTORS:
        raise ValueError(f"model must be one of {sorted(DETECTORS)}, not {model!r}")
    entry_name, file_name = DETECTORS[model]
    return _weights_file(entry_name, file_name, version)


def get_model(model=None, model_path=None, version=None, device=None):
    """The detector (a :class:`.networks.Detector`), loaded once per process
    for each file and set of execution providers."""
    from .networks import Detector, providers_for

    path = detector_path(model, model_path, version)
    providers = providers_for(device)
    key = (path, tuple(providers))
    with _cache_lock:
        if key not in _detector_cache:
            _detector_cache[key] = Detector(path, providers)
        return _detector_cache[key]


def get_refiner(model_id, device=None):
    """A center-step network (a key of :data:`REFINERS`), loaded once per
    process for each set of execution providers."""
    from .networks import CenterRefiner, providers_for

    entry_name, file_name = REFINERS[model_id]
    providers = providers_for(device)
    key = (model_id, tuple(providers))
    with _cache_lock:
        if key not in _refiner_cache:
            _refiner_cache[key] = CenterRefiner(
                _weights_file(entry_name, file_name), providers
            )
        return _refiner_cache[key]


def clear_model_cache():
    with _cache_lock:
        _detector_cache.clear()
        _refiner_cache.clear()
