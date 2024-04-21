"""
Background subtraction
==============

If your diffraction data is noisy, you might want to subtract the background from the
dataset. Pyxem offers some built-in functionality for this, with the
`subtract_diffraction_background` class method. Custom filtering is also possible, an
example is shown in the 'Filtering Data'-example.
"""

import pyxem as pxm
import hyperspy.api as hs

s = pxm.data.twinned_nanowire(allow_download=True)

s_filtered = s.subtract_diffraction_background(
    "difference of gaussians",
    inplace=False,
    min_sigma=2,
    max_sigma=8,
)

hs.plot.plot_images(
    [s.inav[9, 42], s_filtered.inav[9, 42]],
    label=["Original", "Difference of Gaussians"],
    tight_layout=True,
    norm="symlog",
    cmap="viridis",
    colorbar=None,
)
# %%

"""
The available methods differ for `Diffraction2D` datasets and `PolarDiffraction2D`
datasets.
"""

# Set the center of the diffraction pattern to its default,
# i.e. the middle of the image
s.calibrate.center = None

# Transform to polar coordinates
s_polar = s.get_azimuthal_integral2d(npt=100, mean=True)

s_polar_filtered = s_polar.subtract_diffraction_background(
    "radial median",
    inplace=False,
)

hs.plot.plot_images(
    [s_polar.inav[9, 42], s_polar_filtered.inav[9, 42]],
    label=["Original (polar)", "Radial Median"],
    tight_layout=True,
    norm="symlog",
    cmap="viridis",
    colorbar=None,
)
