"""
On Zone Orientation
===================
Sometimes you have a tilt boundary and you might want to know the orientation of the
grains on each side of the boundary. This can be done using the
:meth:`pyxem.signals.PolarSignal2D.get_orientation` method.
"""

from pyxem.data import si_tilt, si_phase
from diffsims.generators.simulation_generator import SimulationGenerator
from orix.quaternion import Rotation

simulated_si_tilt = si_tilt()

# %%
# First we center the diffraction patterns and get a polar signal
# Increasing the number of npt_azim with give better polar sampling
# but will take longer to compute the orientation map
# The mean=True argument will return the mean pixel value in each bin rather than the sum
# this makes the high k values more visible

simulated_si_tilt.calibrate.center = None
polar_si_tilt = simulated_si_tilt.get_azimuthal_integral2d(
    npt=100, npt_azim=360, inplace=False, mean=True
)
polar_si_tilt.plot()

# %%

# Now we can get make the orientation map
phase = si_phase()
generator = SimulationGenerator(200)
sim = generator.calculate_diffraction2d(
    phase,
    rotation=Rotation.from_euler(
        [[0, 0, 0], [0, 0, 0]],
        degrees=True,
    ),
    max_excitation_error=0.1,
    reciprocal_radius=1,
)
orientation_map = polar_si_tilt.get_orientation(sim)
orientation_map.plot_over_signal(simulated_si_tilt, annotate=True)
