"""
Flat Ewald's Sphere Assumption
==============================
In most cases, the Ewald's sphere is assumed to be flat for diffraction patterns when doing 4D STEM.
This is almost always a good assumption.  That being said there are a couple of common units
used with 4D STEM (e.g. nm\ :sup:`-1`, mrad, pixel coordinates) and it is important to understand how these
units relate to each other.  These units are also related to the camera length, pixel size, and beam energy
(wavelength) of the microscope.  In reality the beam energy is really the only thing that you need to
know.

Let's look at an example of this using the `ZrNb Precipitate` dataset.
"""

from pyxem.data import pdnip_glass

g = pdnip_glass(allow_download=True)
g.calibration.beam_energy = "200 kV"
g.calibration.convergance_angle = "1 mrad"  # set the convergence angle

# %%
# Changing to mrad
# ================
# Now we can change the units to mrad. This will automatically calculate the camera length based on the
# beam energy and pixel size.
g.calibration.convert_signal_units("mrad")

g.plot()

# %%
# Changing back to nm\ :sup:`-1`
# =============================
# We can change back to nm\ :sup:`-1`, now that the camera length is (accurately) set, and the
# new scale will be automatically calculated.

g.calibration.convert_signal_units("nm^-1")
g.plot()

# %%
# Changing to pixel coordinates
# =============================
# We can also change to pixel coordinates. This will set the scale to 1 and the units to "px". But we won't
# lose the calibration information, so we can switch back to nm\ :sup:`-1` or mrad easily.
g.calibration.convert_signal_units("px")
print(g.calibration._mrad_scale)  # scale in mrad

g.plot()
print("after", g.calibration._mrad_scale)  # scale in mrad

# %%
# Rebinning
# =========
# We can also rebin the data in pyxem and the calibration will automatically adjust

g.calibration.convert_signal_units("nm^-1")
g_rebinned = g.rebin(scale=(1, 1, 2, 2))
g_rebinned.plot()


# %%
# sphinx_gallery_thumbnail_number = 2
