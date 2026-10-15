"""
Finding Vectors with find_vectors
=================================

:meth:`~pyxem.signals.Diffraction2D.find_vectors` finds the spots in every
pattern of a dataset and returns them as
:class:`~pyxem.signals.DiffractionVectors`, with subpixel positions and the
mean intensity of each spot.
"""

import pyxem as pxm

s = pxm.data.tilt_boundary_data()

# %%
# Three detectors are available:
#
# - ``"dog"``: a difference-of-Gaussians band-pass. ``sigma1`` removes pixel
#   noise, ``sigma2`` the smooth background; the threshold is a
#   signal-to-noise ratio. Suited to small, sharp spots.
# - ``"nxcorr"``: normalised cross-correlation against a flat disk of
#   ``disk_radius``; the threshold is a correlation score in [-1, 1]. Suited to
#   disks of a known size.
# - ``"neural"``: a U-Net trained to find spots of any size with no tuning
#   (``pip install pyxem[neural]``; the weights are downloaded from
#   em-database).
#
# The disks in this dataset are about 9 px across, so the difference of
# Gaussians is widened to match them.

vectors = s.find_vectors(method="dog", sigma1=2, sigma2=5, threshold=4, min_distance=8)
vectors

# %%
# The same search with the cross-correlation detector, with the disk radius
# estimated from the pattern in the middle of the scan.

vectors_nxcorr = s.find_vectors(
    method="nxcorr", disk_radius="auto", threshold=0.8, min_distance=8
)
print(vectors_nxcorr.metadata.Peaks.parameters)

# %%
# Plot the vectors on the signal as markers.

markers = vectors.to_markers(color="red", sizes=10, alpha=0.5)
s.plot()
s.add_marker(markers)

# %%
# Tuning interactively
# --------------------
#
# In a notebook, ``interactive=True`` returns a widget showing the navigator
# beside the pattern under its crosshair, with the spots the current settings
# find drawn as circles. Move the crosshair and the sliders until every spot is
# circled, then call ``compute()`` to search the whole dataset with those
# settings. It needs ``pip install pyxem[interactive]``.
#
# .. code-block:: python
#
#     explorer = s.find_vectors(interactive=True)
#     explorer  # display it, then tune
#     vectors = explorer.compute()
#     explorer.parameters  # the settings, to reuse in a script

# %%
# Lazy signals are processed one navigation chunk at a time, so the whole
# dataset is never loaded into memory.

lazy_vectors = s.as_lazy().find_vectors(
    method="dog", sigma1=2, sigma2=5, threshold=4, min_distance=8
)
lazy_vectors.compute()
# sphinx_gallery_thumbnail_number = 1
