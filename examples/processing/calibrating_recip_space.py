"""Calibrating Scales

This example demonstrates how to calibrate the scales in both reciporical
and real-space.  Ideally this should be done using a calibration standard
with a known lattice spacing.

There are many ways to calibrate the microscope so raise an issue if you
have a specific method you would like to see implemented or if you have
any questions.
"""

from diffsims.generators.simulation_generator import SimulationGenerator
from orix.crystal_map import Phase
import pyxem as pxm
import hyperspy.api as hs

# Load the data and the cif file
au_dpeg = hs.load("au_xgrating_20cm.tif", signal_type="electron_diffraction")
gold_phase = Phase.from_cif("au.cif")

# %%
# Create a Simulation Generator
# -----------------------------
# Use the SimulationGenerator to create a simulated diffraction pattern
# from the gold phase. Note that the acceleration voltage is irrelevant

sim_gen = SimulationGenerator()
sim1d = sim_gen.calculate_diffraction1d(gold_phase)

# %%
# Azimuthal Integration and Calibration
# -------------------------------------
# The polar unwrapping for the 2D image will give a 1D diffraction pattern
# that can be used for calibration.  The calibration is done by comparing
# the positions of the peaks in the simulated and experimental diffraction
# patterns. Fitting with a :class:`hyperspy.model.Model1D` can be used to find the
# scale or the camera length.

au_dpeg.calibration.center = None  # Set the center
az1d = au_dpeg.get_azimuthal_integral1d(npt=200, radial_range=(20, 128))  # in pixels
diffraction_model = az1d.model_simulation1d(sim1d, fit=True)
theta_scale = az1d.model2theta_scale(
    simulation=sim1d, model=diffraction_model, beam_energy=200
)

theta_scale = az1d.model2theta_scale(
    simulation=sim1d, model=diffraction_model, beam_energy=200
)
camera_length = az1d.model2camera_length(
    simulation=sim1d,
    model=diffraction_model,
    beam_energy=200,
    physical_pixel_size=5.5e-5,
)

print("Theta Scale: ", theta_scale, " Rad")
print(
    "Camera Length: ", camera_length, " cm"
)  # Camera length of 22.7 cm (vs 20cm from the microscope)

# %%
# Applying a Camera Length
# ------------------------
# The (calibrated) camera length can be applied to the data to calibrate the scale
# more accurately than a simple single point calibration.  Let's apply the camera
# length to the data and plot the result.

au_dpeg.calibration.detector(
    pixel_size=5.5e-5, detector_distance=camera_length, beam_energy=200
)
print(au_dpeg.axes_manager)  # non uniform axes
au_dpeg.plot(vmax="99th")  # non uniform axes

# %%
# Plotting Corrected Azimuthal Integration
# ----------------------------------------
# When you get the Azimuthal Integration now that things are calibrated it automatically
# accounts for the ewald sphere now and uses the non uniform axes.

calibrated_azim = au_dpeg.get_azimuthal_integral1d(npt=200, radial_range=(2, 15))
calibrated_azim.plot()
