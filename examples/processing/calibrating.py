"""
Calibrating a dataset
=====================

There are three different ways to calibrate a dataset in pyxem and depending on what
kind of data you have you may need to use each of these methods.

The first method is to basically ignore the Ewald sphere effects. This is the
easiest method but not the most correct.  For a 200+ keV microscope the assumption
that the Ewald sphere is flat is not a bad one.  For lower energy microscopes
this assumption is not great but still not terrible.  For x-ray data with longer
wavelengths this assumption isn't great.
"""

import pyxem as pxm

au = pxm.data.au_grating()

# determine the pixel size from one peak

au.calibrate.scale(0.1, units="k_nm^-1")
print(au.calibrate)
au.plot()
# %%

"""
Calibrating from Microscope Parameters
--------------------------------------

It's also possible to calibrate from microscope parameters.  This is more common when you are
working with x-ray data or have pre-calibrated some of the parameters of your microscope. This
method is also more accurate for lower energy microscopes. 

"""

au.calibrate.detector(pixel_size=(0.1, 0.1),
                      detector_distance=0.2,
                      beam_energy=200,
                      unit="k_nm^-1")
print(au.calibrate)
au.plot()
# %%

"""
Calibrating from a Calibration Standard
---------------------------------------

The final method is to calibrate from a calibration standard.  This is the most accurate method
but requires a calibration standard.  This method is equivalent to the method used above but
we determine some of the parameters from the calibration standard.

Note that if you don't know the physical pixel size for your detector you can always just set this to
1.  This will give you an unphyiscal detector_distance (camera length) but the results will be the same
as they are not independent parameters.
"""

au.calibrate.standard(, 0.1, 200, unit="k_nm^-1")

