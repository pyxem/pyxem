"""
===========================
Glass Symmetry from Vectors
===========================

This example shows how to identify symmetry (in a glassy system but this could be useful other places)
by looking at the angles between 3 vectors in the diffraction pattern at some radial ring in k to identify
groups of 3 vectors that are subtended by the same angle.

This is a very simple example with more detailed examples to come.
"""

import pyxem as pxm
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np

# %%
# First we load the data and do some basic processing
s = pxm.data.pdnip_glass(allow_download=True)
s.axes_manager.signal_axes[0].offset = -23.7
s.axes_manager.signal_axes[1].offset = -19.3

s.filter(gaussian_filter, sigma=(1, 1, 0, 0), inplace=True)  # only in real space
s.template_match_disk(disk_r=5, subtract_min=False, inplace=True)

vectors = s.get_diffraction_vectors(threshold_abs=0.5, min_distance=3)

# %%
# Now we can convert to polar vectors

pol = vectors.to_polar()

# %%
# This function gets the inscribed angle
# accept_threshold is the maximum difference between the two angles subtended by the 3 vectors

ins = pol.get_angles(min_angle=0.05, min_k=0.3, accept_threshold=0.1)

flat_vect = ins.flatten_diffraction_vectors()

fig, axs = plt.subplots()
axs.hist(flat_vect.ivec["delta phi"].data, bins=60, range=(0, 2 * np.pi / 3))
axs.set_xlabel("delta phi")
axs.set_xticks(
    [0, np.pi / 5, np.pi / 4, 2 * np.pi / 5, np.pi / 2, np.pi / 3, 3 * np.pi / 5]
)
axs.set_xticklabels(
    [
        0,
        r"$\frac{\pi}{5}$",
        r"$\frac{\pi}{4}$",
        r"$\frac{2\pi}{5}$",
        r"$\frac{\pi}{2}$",
        r"$\frac{\pi}{3}$",
        r"$\frac{3\pi}{5}$",
    ]
)

# %%
# cycle through colors in groups of 3 for each symmetry cluster

points = ins.to_markers(
    color=["b", "b", "b", "g", "g", "g", "y", "y", "y", "r", "r", "r"]
)
original_points = vectors.to_markers(color="w", alpha=0.5)
s.axes_manager.indices = (67, 55)  # jumping to a part with some symmetric structure

s.plot(vmin=0.0)
s.add_marker(points)
s.add_marker(original_points)

# %%
