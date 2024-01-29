"""
Calibrating a dataset
=====================

There are two different ways to calibrate a dataset in pyxem and depending on what
kind of data you have you may need to use each of these methods.

The first method is to basically ignore the Ewald sphere effects. This is the
easiest method but not the most correct.  For a 200+ keV microscope the assumption
that the Ewald sphere is flat is not a bad one.  For lower energy microscopes
this assumption is not as great but still not terrible.  For x-ray data with longer
wavelengths this assumption starts to break down.
"""

# import pyxem as pxm

# al = pxm.data.al_peaks()

# determine the pixel size from one peak

# al.calibrate(scale=0.1, center=None, units="k_nm^-1")
# al.plot()
