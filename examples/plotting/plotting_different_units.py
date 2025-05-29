"""
Plotting using Different Units
==============================

Sometimes you might want to plot using different units, such as pixel coordinates,
or mrad or nm^-1. Pyxem makes it easy to plot using different units.  This does require
certain calibration information to be set, such as the beam energy and the pixel size.

"""

from pyxem.data import fe_multi_phase_grains

mulit_phase = fe_multi_phase_grains()

mulit_phase.calibration.beam_energy = 200  # Set the beam energy to 200 keV
mulit_phase.calibration.pixel_size = 15e-6  # Set the pixel size to 15 um
mulit_phase.calibration.units = "nm^-1"  # Set the units to nm^-1
mulit_phase.calibration.scale = 0.1  # Set the scale to 0.1 nm^-1

# %%
# Plotting the diffraction pattern using pixel coordinates
mulit_phase.plot(units="px")
# %%
# Plotting the diffraction pattern using mrad
mulit_phase.plot(units="mrad")
# %%
# Plotting the diffraction pattern using A^-1
mulit_phase.plot(units="A^-1")
