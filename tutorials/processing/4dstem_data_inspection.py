"""
4D-STEM Data Inspection and Preprocessing
==========================================

This tutorial walks through a complete 4D-STEM data inspection and preprocessing
workflow using a real GaAs twinned nanowire dataset. It covers:

- Loading and inspecting a 4D-STEM dataset
- Calibrating the diffraction space
- Virtual dark-field (VDF) imaging from regions of interest
- Decomposition using SVD and NMF to identify structural phases
- Diffraction vector extraction and peak finding

The dataset is a GaAs nanowire that contains both the zinc-blende (cubic) and
wurtzite (hexagonal) crystal phases, separated by twin boundaries. This makes
it an ideal dataset for learning multi-phase 4D-STEM analysis.

The data can be accessed from https://zenodo.org/records/15490547
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pyxem as pxm

# %%
# Loading the Data
# ----------------
# The twinned nanowire dataset is a 4D-STEM scan of a GaAs nanowire with
# twin boundaries. We load it lazily to avoid reading the full array into memory.

s = pxm.data.twinned_nanowire(allow_download=True, lazy=True)
print(s)

# %%
# Inspecting the Data
# -------------------
# The dataset has 2 navigation axes (the scan positions) and 2 signal axes
# (the diffraction pattern). Let's look at the metadata and axes.

print(s.axes_manager)

# %%
# Set a meaningful title and inspection parameters.
s.metadata.General.title = "GaAs Twinned Nanowire"

# Plot a single diffraction pattern from near the centre of the scan.
# The data is (30, 100 | 144, 144): 30 rows × 100 columns of scan positions,
# each with a 144 × 144 diffraction pattern.

s.inav[15, 50].plot(vmax=500, cmap="inferno")

# %%
# You can navigate the dataset interactively. The mean diffraction pattern
# summarises the average structure over all scan positions.

mean_dp = s.mean()
mean_dp.plot(vmax=200, cmap="inferno")

# %%
# Calibrating Diffraction Space
# -----------------------------
# We calibrate using the known d-spacing of the GaAs {111} planes.
# d₁₁₁ = a/√3 for cubic GaAs (a = 5.6535 Å).

a_GaAs = 5.6535  # Å
recip_d111 = np.sqrt(3) / a_GaAs  # 1/Å
# The {111} spot appears at pixel 11.4 from the centre in this dataset.
recip_cal = recip_d111 / 11.4  # Å⁻¹/pixel

s.calibration(scale=recip_cal, units="1/Å")
print(s.axes_manager)

# %%
# Virtual Dark-Field Imaging
# --------------------------
# Virtual dark-field (VDF) imaging integrates the diffracted intensity within
# a chosen region of interest (ROI) at each scan position, producing a real-space
# map of regions that satisfy a particular diffraction condition.

# Select a small circular aperture around the transmitted beam.
roi_bright = hs.roi.CircleROI(cx=0.0, cy=0.0, r_inner=0, r=0.04)
vdf_bright = s.get_integrated_intensity(roi_bright)
vdf_bright.metadata.General.title = "Bright-field VDF"
vdf_bright.plot(cmap="gray")

# %%
# Select a ring aperture that picks up the {111} diffraction ring.
roi_111 = hs.roi.CircleROI(cx=0.0, cy=0.0, r_inner=0.04, r=0.10)
vdf_111 = s.get_integrated_intensity(roi_111)
vdf_111.metadata.General.title = "VDF from {111} ring"
vdf_111.plot(cmap="viridis")

# %%
# Annular dark-field: integrate everything outside the bright-field disc.
roi_adf = hs.roi.CircleROI(cx=0.0, cy=0.0, r_inner=0.04, r=0.25)
vdf_adf = s.get_integrated_intensity(roi_adf)
vdf_adf.metadata.General.title = "Annular dark-field VDF"
vdf_adf.plot(cmap="viridis")

# %%
# Radial Profile Stacks
# ---------------------
# Azimuthal integration creates a 1D radial intensity profile at every scan
# position. Stacking these profiles as a function of the scan position gives
# a 3D dataset useful for spotting position-dependent structural changes.

s1d = s.get_azimuthal_integral1d(npt=50, inplace=False)
print(s1d)
# Sum over navigation axes to get the average radial profile.
s1d.sum().plot()

# %%
# Decomposition: SVD
# ------------------
# Matrix decomposition factorises the 4D dataset into a set of "factors"
# (representative diffraction patterns) and "loadings" (their spatial distribution).
# SVD (Singular Value Decomposition) is fast and gives a good overview.

# Work on a spatial subset so this runs quickly in the tutorial.
subset = s.inav[:, 30:70].deepcopy()
subset.compute()  # decomposition requires a non-lazy signal
subset.change_dtype("float64")
subset.decomposition(normalize_poissonian_noise=True, algorithm="SVD")
subset.plot_explained_variance_ratio(n=20)

# %%
# The explained-variance ratio shows how many components are needed.
# The first sharp "elbow" in the scree plot indicates the number of meaningful
# components; noise components lie on the flat tail.

subset.plot_decomposition_results()

# %%
# Decomposition: NMF
# ------------------
# Non-negative Matrix Factorisation (NMF) constrains both factors and loadings to
# be non-negative, which is physically motivated for diffraction intensities.
# It often gives more interpretable results than SVD.

n_components = 4
subset.decomposition(
    normalize_poissonian_noise=True,
    algorithm="NMF",
    output_dimension=n_components,
)
subset.plot_decomposition_results()

# %%
# Each NMF factor is a representative diffraction pattern, and each loading map
# shows where in the scan that pattern is dominant.  Compare the loading maps
# with the VDF images created above: the NMF should separate the zinc-blende
# and wurtzite phases along the nanowire.

# %%
# Peak Finding and Diffraction Vectors
# -------------------------------------
# For crystallographic analysis it is useful to locate the diffraction spots in
# each pattern.  We use a difference-of-Gaussians (DoG) blob detector.

# Run peak finding on a small region to keep the tutorial fast.
region = s.inav[:, 40:60]
peaks = region.find_peaks(
    method="difference_of_gaussian",
    min_sigma=1.0,
    max_sigma=6.0,
    sigma_ratio=1.6,
    threshold=0.04,
    overlap=0.99,
    interactive=False,
)
print(peaks)

# %%
# Convert pixel-coordinate peaks to reciprocal-space DiffractionVectors.
# The pattern centre is at pixel (72, 72) for this 144 × 144 detector.

from pyxem.signals.diffraction_vectors import DiffractionVectors

vectors = DiffractionVectors.from_peaks(peaks, center=(72, 72), calibration=recip_cal)
print(vectors)

# %%
# The diffracting-pixels map assigns each scan position an intensity proportional
# to the number of diffraction spots found there.  Crystalline regions light up,
# amorphous or beam-damaged areas remain dark.

crystim = vectors.get_diffracting_pixels_map(binary=False)
crystim.plot(cmap="viridis")

# sphinx_gallery_thumbnail_number = 5
