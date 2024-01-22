# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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

import pooch
from pyxem.data._registry import _file_names_hash, _urls
import hyperspy.api as hs
from pathlib import Path
from pyxem.release_info import version
from typing import Optional, Union
import os

# Create a downloader object
kipper = pooch.create(
    path=pooch.os_cache("pyxem"),
    base_url="",
    version=version.replace(".dev", "+"),
    version_dev="main",
    env="PYXEM_DATA_DIR",
    registry=_file_names_hash,
    urls=_urls,
)


def au_grating(allow_download=False, **kwargs):
    """An au_grating 4-D STEM dataset used to show calibration.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~hyperspy.api.load`.

    Examples
    --------
    >>> import pyxem as pxm
    >>> s = pxm.data.au_grating()
    >>> print(s)
    <ElectronDiffraction2D, title: , dimensions: (|254, 254)>
    >>> s.plot()
    """
    grating = Dataset("au_xgrating_100kX.hspy")
    file_path = grating.fetch_file_path(allow_download=allow_download)
    return hs.load(file_path, **kwargs)


def pdnip_glass(allow_download=False, **kwargs):  # pragma: no cover
    """A small PdNiP glass 4-D STEM dataset.

    Data can be acessed from https://zenodo.org/records/8429302

    Data liscenced under CC BY 4.0

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~hyperspy.api.load`.

    Examples
    --------
    >>> import pyxem as pxm
    >>> s = pxm.data.pdnip_glass()
    >>> print(s)
    <ElectronDiffraction2D, title: , dimensions: (128, 128|128, 128)>
    >>> s.plot()
    """
    import zarr

    pdnip = Dataset("PdNiP.zspy")
    file_path = pdnip.fetch_file_path(allow_download=allow_download)
    store = zarr.ZipStore(file_path)  # for loading from zip
    return hs.load(store, **kwargs)


def zrnb_precipitate(allow_download=False, **kwargs):  # pragma: no cover
    """A small 4-D STEM dataset of a ZrNb precipitate for strain mapping.

    Data liscenced under CC BY 4.0

    Data can be acessed from https://zenodo.org/records/8429302

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~hyperspy.api.load`.

    Examples
    --------
    >>> import pyxem as pxm
    >>> s = pxm.data.zrnb_percipitate()
    >>> print(s)
    <ElectronDiffraction2D, title: , dimensions: (128, 128|128, 128)>
    >>> s.plot()
    """
    import zarr

    zrnb = Dataset("ZrNbPercipitate.zspy")
    file_path = zrnb.fetch_file_path(allow_download=allow_download)
    store = zarr.ZipStore(file_path)  # for loading from zip
    return hs.load(store, **kwargs)


def twinned_nanowire(allow_download=False, **kwargs):  # pragma: no cover
    """A small 4-D STEM dataset of a twinned nanowire for orientation mapping.

    Data can be acessed from https://zenodo.org/records/8429302

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~hyperspy.api.load`.

    Examples
    --------
    >>> import pyxem as pxm
    >>> s = pxm.data.twinned_nanowire()
    >>> print(s)
    <Diffraction2D, title: , dimensions: (30, 100|144, 144)>
    >>> s.plot()
    """
    nanowire = Dataset("twinned_nanowire.hdf5")
    file_path = nanowire.fetch_file_path(allow_download=allow_download)
    return hs.load(file_path, **kwargs, reader="hspy")


def sample_with_g(allow_download=False, **kwargs):  # pragma: no cover
    """A small 4-D STEM dataset of a twinned nanowire for orientation mapping.

    Data can be acessed from https://zenodo.org/records/8429302

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~hyperspy.api.load`.

    Examples
    --------
    >>> import pyxem as pxm
    >>> s = pxm.data.sample_with_g()
    >>> print(s)
    <ElectronDiffraction2D, title: , dimensions: (25, 26|256, 256)>
    >>> s.plot()
    """
    import zarr

    sample = Dataset("sample_with_g.zspy")
    file_path = sample.fetch_file_path(allow_download=allow_download)
    store = zarr.ZipStore(file_path)  # for loading from zip
    return hs.load(store, **kwargs)


def mgo_nanocrystals(allow_download=False, **kwargs):  # pragma: no cover
    """A small 4-D STEM dataset of overlapping MgO nanocrystals

    Data can be acessed from https://zenodo.org/records/8429302

    Parameters
    ----------
    **kwargs
        Keyword arguments passed to :func:`~hyperspy.api.load`.

    Examples
    --------
    >>> import pyxem as pxm
    >>> s = pxm.data.mgo_nanocrystals()
    >>> print(s)
    <ElectronDiffraction2D, title: MgO Nano-Crystals, dimensions: (105, 105|144, 144)>
    >>> s.plot()
    """
    import zarr

    sample = Dataset("mgo_nano.zspy")
    file_path = sample.fetch_file_path(allow_download=allow_download)
    store = zarr.ZipStore(file_path)  # for loading from zip
    return hs.load(store, **kwargs)


class Dataset:
    file_relpath: Path
    file_package_path: Path
    file_cache_path: Path
    expected_md5_hash: str = ""

    def __init__(
        self,
        file_relpath: Union[Path, str],
    ) -> None:
        if isinstance(file_relpath, str):
            file_relpath = Path(file_relpath)
        self.file_package_path = Path(os.path.dirname(__file__)) / file_relpath

        file_relpath = file_relpath
        self.file_relpath = file_relpath
        self.file_cache_path = Path(kipper.path) / self.file_relpath

        self.expected_md5_hash = _file_names_hash[self.file_relpath_str]

    @property
    def file_relpath_str(self) -> str:
        return self.file_relpath.as_posix()

    @property
    def is_in_package(self) -> bool:
        return self.file_package_path.exists()

    @property
    def is_in_cache(self) -> bool:
        return self.file_cache_path.exists()

    @property
    def file_path(self) -> Path:
        return self.file_cache_path

    @property
    def file_path_str(self) -> str:
        return self.file_path.as_posix()

    @property
    def md5_hash(self) -> Union[str, None]:
        if self.file_path.exists():
            return pooch.file_hash(self.file_path_str, alg="md5")
        else:  # pragma: no cover
            return None

    @property
    def has_correct_hash(self) -> bool:
        return self.md5_hash == self.expected_md5_hash.split(":")[1]

    @property
    def url(self) -> Union[str, None]:
        if self.file_relpath_str in _urls:
            return _urls[self.file_relpath_str]
        else:
            return None

    def fetch_file_path(
        self, allow_download: bool = False, show_progressbar: Optional[bool] = None
    ) -> str:
        if show_progressbar is None:
            show_progressbar = hs.preferences.General.show_progressbar
        downloader = pooch.HTTPDownloader(progressbar=show_progressbar)

        if self.is_in_cache:
            if self.has_correct_hash:
                file_path = self.file_relpath_str
            elif allow_download:  # pragma: no cover
                file_path = self.file_relpath_str
            else:  # pragma: no cover
                raise ValueError(
                    f"File {self.file_path_str} must be re-downloaded from the "
                    f"repository file {self.url} to your local cache {kipper.path}. "
                    "Pass `allow_download=True` to allow this re-download."
                )
        else:  # pragma: no cover
            if allow_download:
                file_path = self.file_relpath_str
            else:
                raise ValueError(
                    f"File {self.file_relpath_str} must be downloaded from the "
                    f"repository file {self.url} to your local cache {kipper.path}. "
                    "Pass `allow_download=True` to allow this download."
                )

        return kipper.fetch(file_path, downloader=downloader)
