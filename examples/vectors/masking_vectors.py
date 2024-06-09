"""
Diffraction Vectors to Mask
===========================

This example shows how to take a set of vectors and create a mask
from a set of circles around each vector. Multiplying by the original signal
will remove all the signal except for the circles around the vectors.
"""

import pyxem as pxm
import hyperspy.api as hs

s = pxm.data.tilt_boundary_data()
s.calibration.center = None

s.axes_manager[2].scale = 0.3
temp = s.template_match_disk(disk_r=5, subtract_min=False)  # Just right
vectors = temp.get_diffraction_vectors(threshold_abs=0.4, min_distance=5)

mask = vectors.to_mask(disk_r=7)
masked_s = s * mask
masked_s.plot()

# %%
# Selecting One Position
# ----------------------
# We can also just select one position and create a mask from that.

masked_s = s * mask.inav[0, 0]
masked_s.plot()

# %%
# Removing Vectors
# ----------------
# We can also remove the vectors from the signal by inverting the mask
# and multiplying by the original signal.

vectors_filtered = vectors.filter_magnitude(1, 15)
mask = vectors_filtered.to_mask(disk_r=9)
masked_s = s * ~mask
masked_s.plot()

# %%
# sphinx_gallery_thumbnail_number = 6
