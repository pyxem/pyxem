"""
Flat Ewald's Sphere Assumption
==============================
In most cases, the Ewald's sphere is assumed to be flat for diffraction patterns when doing 4D STEM.
This is almost always a good assumption.  That being said there are a couple of common units
used with 4D STEM (e.g. nm^-1, mrad, pixel coordinates) and it is important to understand how these
units relate to each other.

We try to make this as easy as possible in Pyxem.  In general, we approach this problem by requiring three
pieces of information to be set in the calibration:

1. The beam energy (in keV)
2. The physical pixel size (in meters)
3. The camera length (in meters)**

The first two are readily available from the microscope and the camera manufacturer. The camera length, especially
given by the microscope manufacturer, is often inaccurate. To get around this you can set the scale directly using
mrad, nm^-1, or A^-1 and then as long as the beam energy and pixel size are set, the camera length will be
automatically calculated.  Once those three pieces of information are set, it's easy to switch between the
different units.


.. image:: CalibrationFlat.png
  :width: 400
  :alt: An image showing the relationship between the camera length, pixel size, beam energy (wavelength), and the
   different units used in Pyxem.

Let's look at an example of this using the `ZrNb Precipitate` dataset.  This dataset was taken using a Gen 1 Titan
at 200 keV on a DE 16 with a physical pixel size of 6.5 um.

It's important to note that rebinning will "effectively" change the pixel size. We've tried to make this work
seamlessly in Pyxem, so you can rebin the data and the calibration will automatically adjust to the new pixel size.
But if you rebin in some other software it is important to keep in mind.
"""

from pyxem.data import pdnip_glass

g = pdnip_glass(allow_download=True)
g.calibration.beam_energy = 200
g.calibration.pixel_size = (
    15e-6 * 2
)  # this dataset was taken with 2x binning on a Celeritas with 15 um pixels

# %%
# Changing to mrad
# ================
# Now we can change the units to mrad. This will automatically calculate the camera length based on the
# beam energy and pixel size.
g.calibration.change_signal_units("mrad")

g.plot()

# %%
# Changing back to nm^-1
# ======================
# We can change back to nm^-1, now that the camera length is (accurately) set, and the
# new scale will be automatically calculated.

g.calibration.change_signal_units("nm^-1")
g.plot()

# %%
# Changing to pixel coordinates
# =============================
# We can also change to pixel coordinates. This will set the scale to 1 and the units to "px". But we won't
# lose the calibration information, so we can switch back to nm^-1 or mrad easily.
g.calibration.change_signal_units("px")

g.plot()

# %%
# Rebinning
# =========
# We can also rebin the data in pyxem and the calibration will automatically adjust to the new pixel size.
# This works by changing the physical pixel size based on the rebinning factor.

g_rebinned = g.rebin(scale=(1, 1, 2, 2))
g_rebinned.calibration.change_signal_units("nm^-1")
g_rebinned.plot()


# %%
# sphinx_gallery_thumbnail_number = 2
