"""
====================
Coordinates in Pyxem
====================

Pyxem is flexible in how it handles coordinates for a diffraction pattern.

There are three main ways to handle coordinates in Pyxem:

1. Pixel coordinates
2. Calibrated Coordinates with evenly spaced axes
3. Calibrated Coordinates with unevenly spaced axes (e.g. corrected for the Ewald sphere)
"""

import pyxem as pxm
from skimage.morphology import disk


s = pxm.signals.Diffraction2D(disk((10)))
s.calibration.center = None
print(s.calibration.center)

# %%

s.plot(axes_ticks=True)
# %%
# From the plot above you can see that hyperspy automatically sets the axes ticks to be centered
# on each pixel. This means that for a 21x21 pixel image, the center is at (-10, -10) in pixel coordinates.
# if we change the scale using the calibration function it will automatically adjust the center.  Here it is
# now (-1, -1)

s.calibration.scale = 0.1
s.calibration.units = "nm$^{-1}$"
s.plot(axes_ticks=True)
print(s.calibration.center)


# %%
# Azimuthal Integration
# ---------------------
#
# Now if we do integrate this dataset it will choose the appropriate center based on the center pixel.

az = s.get_azimuthal_integral2d(npt=30)
az.plot()

# %%
# Non-Linear Axes
# ---------------
#
# Now consider the case where we have non-linear axes. In this case the center is still (10,10)
# but things are streatched based on the effects of the Ewald Sphere.

s.calibration.beam_energy = 200
s.calibration.detector(pixel_size=0.1, detector_distance=3)
print(s.calibration.center)
s.plot()

az = s.get_azimuthal_integral2d(npt=30)
az.plot()

# %%
# sphinx_gallery_thumbnail_number = 4
