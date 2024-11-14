"""
Background subtraction
======================

If your diffraction data is noisy, you might want to subtract the background from the
dataset. Pyxem offers some built-in functionality for this, with the
`subtract_diffraction_background` class method. Custom filtering is also possible, an
example is shown in the 'Filtering Data'-example.
"""

import pyxem as pxm
import hyperspy.api as hs

s = pxm.data.tilt_boundary_data()

s_filtered = s.subtract_diffraction_background(
    "difference of gaussians",
    inplace=False,
    min_sigma=3,
    max_sigma=20,
)

s_filtered_h = s.subtract_diffraction_background("h-dome", inplace=False, h=0.7)


hs.plot.plot_images(
    [s.inav[2, 2], s_filtered.inav[2, 2], s_filtered_h.inav[2, 2]],
    label=["Original", "Difference of Gaussians", "H-Dome"],
    tight_layout=True,
    norm="symlog",
    cmap="viridis",
    colorbar=None,
)

# %%
# ======================
# Filtering Polar Images
# ======================
# The available methods differ for `Diffraction2D` datasets and `PolarDiffraction2D`
# datasets.
#
# Set the center of the diffraction pattern to its default,
# i.e. the middle of the image

s.calibration.center = None

# %%
# Transform to polar coordinates

s_polar = s.get_azimuthal_integral2d(npt=100, mean=True)

s_polar_filtered = s_polar.subtract_diffraction_background(
    "radial median",
    inplace=False,
)

s_polar_filtered2 = s_polar.subtract_diffraction_background(
    "radial percentile",
    percentile=70,
    inplace=False,
)

hs.plot.plot_images(
    [s_polar.inav[2, 2], s_polar_filtered.inav[2, 2], s_polar_filtered2.inav[2, 2]],
    label=["Original (polar)", "Radial Median", "Radial Percentile"],
    tight_layout=True,
    norm="symlog",
    cmap="viridis",
    colorbar=None,
)

# %%
