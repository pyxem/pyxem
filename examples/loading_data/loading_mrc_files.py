"""
Loading MRC files (and other binary files)
==========================================

This is a simple example of how to load MRC files using Pyxem. The MRC file format is
a common format for electron microscopy data.  It is a binary format that is used
for storing 3D data, such as electron tomography but because it is a fairly simple format, it has
been adopted in some cases to store 4D STEM data as well.

First we will download a sample MRC file from the Pyxem data repository. This is a good way to host
data if you want to share it with others.  I love putting small versions (up to 50 GB) of every dataset I publish
on Zenodo and then using pooch to automate the download/extraction process.
"""

import os
import zipfile
import pooch

current_directory = os.getcwd()
file_path = pooch.retrieve(
    # URL to one of Pooch's test files
    url="https://zenodo.org/records/15490547/files/ZrNbMrc.zip",
    known_hash="md5:eeac29aee5622972daa86a394a8c1d5c",
    progressbar=True,
    path=current_directory,
)
# Unzip the file
with zipfile.ZipFile(file_path, "r") as zip_ref:
    zip_ref.extractall(current_directory)

# %%
# Loading the MRC file
# --------------------
# We can now load the file using the ``load`` method from hyperspy.  This method uses
# the `MRC Reader <https://hyperspy.org/rosettasciio/supported_formats/mrc.html#mrc-format>`_
# to read the file. In this case, because the file was collected with a Direct Electron camera,
# the metadata is automatically loaded as well.

import hyperspy.api as hs

signal = hs.load(
    "ZrNbMrc/20241021_00405_movie.mrc",
)
# %%
# Loading Lazily
# --------------
# In this case the file was loaded using the numpy.memmap function,
# this won't load the entire file into memory, but if for example you
# do ``signal.sum()`` now the entire file will be loaded into memory.
# In most cases it is better to just use the ``lazy=True`` option to load the file lazily.

signal = hs.load("ZrNbMrc/20241021_00405_movie.mrc", lazy=True)

signal
# %%
# Controlling the Chunk Size
# --------------------------
# The chunk size is the number of frames that will be loaded into memory at once when
# the signal is lazy loaded.  This can be controlled using the ``chunks`` parameter.
# A good place to start is to use the ``auto`` option for the first two dimensions, which will
# automatically determine the chunk size based on the available memory. The last two dimensions
# are the reciprocal space dimensions, as we usually ``map`` over those dimensions we can set them
# to ``-1`` to indicate that we want to load all the data in those dimensions at once.

signal = hs.load("ZrNbMrc/20241021_00405_movie.mrc", lazy=True, chunks=(10, 10, -1, -1))

signal
# %%
# Slicing the Signal
# ------------------
# Interestingly, binary files are sometimes faster than compressed formats.  With compressed file formats,
# like HDF5 or Zarr, you need to decompress the entire chunk before you can access and part of the
# data. For things like Virtual Images or slicing a signal this can add overhead.  With binary files,
# because the underlying data is a memory map, even for dask arrays, you can very efficiently slice parts
# of the data without loading the entire chunk into memory.

slice_sum = signal.isig[0:10, 0:10].sum()
slice_sum.compute()

# %%
# In this case this is faster than the compressed equivalent, because we don't have to load the
# entire chunk into memory just to throw most of it away.
#
# A couple of more things to note.  Performance of binary files is usually better on SSDs than on HDDs,
# because the seek time is much lower on SSDs.  This means that you can have arbitrary dask chunks and it will
# still be fast. On HDDs, you want to keep data that is close together in the same chunk. Usually this means
# you want chunks like (1, "auto", -1, -1).  This is not terribly noticeable for 1-2 GB files, somewhat noticeable
# for 10-20 GB files, and extremely important for 100+ GB files on an HDD.
