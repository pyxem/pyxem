"""
Strain Mapping of a ZrNb Precipitate
======================================

Strain mapping in 4D-STEM measures local lattice distortions by tracking how
diffraction spot positions shift relative to an unstrained reference.  Where
orientation mapping asks *"which grain orientation is here?"*, strain mapping
asks *"how is the lattice stretched or sheared here?"*.

Physical Background
-------------------

In a crystalline material under strain the lattice planes are no longer at
their equilibrium spacing.  This shifts the Bragg spots in the diffraction
pattern away from their ideal positions.  The displacement of spot **g** by
**Δg** is related to the distortion of the real-space lattice by the
**displacement gradient tensor** F:

    g_measured = F · g_reference

Decomposing F into its symmetric and antisymmetric parts gives:

- **ε** (symmetric part): the **strain tensor** — normal strains εxx, εyy
  (elongation along x and y) and shear strain εxy.
- **ω** (antisymmetric part): the **rotation tensor** — rigid-body rotation
  of the lattice, not a physical strain.

In practice, we:
1. Detect diffraction disk positions at every scan pixel using template matching.
2. Select a reference region (unstrained matrix far from the feature of interest).
3. Compute F at each pixel by comparing the measured spot positions to the
   reference positions.

This tutorial uses a real 4D-STEM dataset of a ZrNb alloy containing a
second-phase precipitate.  The precipitate has a slightly different lattice
parameter from the matrix, producing a localised strain field at the interface.

Dataset: https://zenodo.org/records/11284654
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pyxem as pxm

# %%
# Loading the Data
# ----------------
# ``zrnb_precipitate`` is a 60 × 40 scan with 256 × 256 diffraction patterns
# recorded at 200 kV.  The calibration is 0.051 nm⁻¹ per pixel.
# The precipitate appears as a darker region in virtual bright-field images.

s = pxm.data.zrnb_precipitate(allow_download=True)
print(s)
print(s.axes_manager)

# %%
# Build a virtual bright-field image to see the real-space structure.
# A circular ROI around the direct beam gives an image where intensity
# roughly tracks sample thickness and crystallinity.

roi_bf = hs.roi.CircleROI(cx=0, cy=0, r=0.1)
vbf = s.get_integrated_intensity(roi_bf)
vbf.plot(cmap="gray")

# %%
# Centring the Direct Beam
# ------------------------
# Strain mapping is extremely sensitive to the position of the beam centre.
# Even a fraction-of-a-pixel error in the centre translates directly into a
# spurious rigid-body shift in the extracted strain field.
#
# ``calibration.center = None`` fits the centre automatically from the
# brightest point in each diffraction pattern.  After centering, the axes
# offsets are updated so that (kx=0, ky=0) coincides with the direct beam.

s.calibration.center = None

# %%
# Look at a single representative diffraction pattern to check the centering
# and to choose template-matching parameters.

s.inav[20, 10].plot(vmax="99th")

# %%
# Disk Detection via Template Matching
# -------------------------------------
# We detect diffraction disks using a cross-correlation with a synthetic disk
# template of radius equal to the measured disk size.
#
# ``template_match_disk(disk_r=...)`` convolves each pattern with a filled
# circular disk and normalises the result to [0, 1].  The output is a
# *correlation map* where peaks indicate disk centres.
#
# ``disk_r`` should match the actual diffraction disk radius in pixels.
# At 256 × 256 pixels and 0.051 nm⁻¹/pixel, a disk of radius ~5 pixels is
# typical for this dataset.  You can estimate this visually from a single
# pattern or by inspecting the radial profile.
#
# ``subtract_min=False`` keeps the baseline at zero, which avoids artefacts
# when the background is non-uniform.

template_matched = s.template_match_disk(disk_r=5, subtract_min=False)
template_matched.plot(vmin=0.3, vmax=1.0)

# %%
# Choosing the Detection Threshold
# ---------------------------------
# The ``threshold_abs`` parameter to ``get_diffraction_vectors`` sets the
# minimum normalised correlation score that counts as a detected disk.
#
# A good strategy is to plot a histogram of the correlation values to find a
# clear gap between the background distribution and the disk peaks.  Setting
# the threshold in this gap minimises both false negatives (missed disks) and
# false positives (noise peaks).

# Use the mean pattern for a quick look at the score distribution.
corr_mean = template_matched.mean()
fig, ax = plt.subplots()
ax.hist(corr_mean.data.ravel(), bins=100, log=True)
ax.axvline(0.35, color="red", linestyle="--", label="threshold = 0.35")
ax.set_xlabel("Normalised correlation score")
ax.set_ylabel("Count (log scale)")
ax.set_title("Disk-detection threshold selection")
ax.legend()

# %%
# Extracting Diffraction Vectors
# ------------------------------
# ``get_diffraction_vectors`` finds local maxima in the correlation map that
# exceed the threshold and are at least ``min_distance`` pixels apart
# (to prevent detecting the same disk twice).
#
# The result is a ``DiffractionVectors`` signal — a ragged array where each
# navigation pixel stores a list of detected spot positions in calibrated
# reciprocal-space units (nm⁻¹ here).

diffraction_vectors = template_matched.get_diffraction_vectors(
    threshold_abs=0.35,
    min_distance=5,
)

# Overlay the detected vectors on the experimental signal as a quick check.
markers = diffraction_vectors.to_markers(color="cyan", sizes=8, alpha=0.7)
s.plot()
s.add_marker(markers)

# %%
# Selecting the First-Order Reflections
# --------------------------------------
# For strain mapping we only want the *first-order* reflections closest to
# the direct beam — these have the largest signal-to-noise and the simplest
# relationship to the lattice parameters.
#
# ``filter_magnitude`` keeps only vectors whose |g| falls in [min, max] in
# calibrated units.  We exclude the direct beam (|g| ≈ 0) and keep only the
# first ring of spots.
#
# Inspect the vector magnitude distribution first to choose sensible limits.

# Plot a histogram of |g| values in a representative pattern.
all_mags = []
vdata = diffraction_vectors.data  # ragged object array, each element is (n, ≥2)
for row in vdata.ravel():
    if row is not None and len(row) > 0:
        mags = np.linalg.norm(row[:, :2], axis=-1)  # use kx, ky columns only
        all_mags.extend(mags.tolist())

fig, ax = plt.subplots()
ax.hist(all_mags, bins=80)
ax.set_xlabel("|g| (nm⁻¹)")
ax.set_ylabel("Count")
ax.set_title("Distribution of diffraction vector magnitudes")
ax.axvline(0.5, color="red", linestyle="--", label="min = 0.5 nm⁻¹")
ax.axvline(5.0, color="orange", linestyle="--", label="max = 5.0 nm⁻¹")
ax.legend()

# %%
first_ring = diffraction_vectors.filter_magnitude(
    min_magnitude=0.5,
    max_magnitude=5.0,
)

# %%
# Choosing a Reference Region
# ----------------------------
# The displacement gradient tensor F is computed *relative* to a reference
# set of vectors.  The reference should come from a scan position that is:
#
# 1. **Unstrained** — away from the precipitate, in the matrix far from the
#    interface.
# 2. **Single-crystal** — ideally one grain with clean, well-defined spots.
# 3. **On-zone** — the pattern should show clear Bragg spots at the expected
#    positions for the matrix structure.
#
# Here we pick a position near the top-left corner of the scan (matrix region).
# Use the virtual bright-field image above to identify a suitable reference.
#
# ``first_ring.inav[col, row]`` follows HyperSpy display order (col, row).

# Choose a reference in the matrix far from the precipitate.
# Inspect the VBF to confirm this is in the unstrained matrix region.
# ``inav[col, row]`` follows HyperSpy display order (col, row).
#
# For ragged DiffractionVectors the single-position data is a 1-element
# object array; use ``.data[0]`` (not ``[()]``) to get the inner 2D array.
ref_col, ref_row = 5, 5
unstrained_vectors = first_ring.inav[ref_col, ref_row].data[0]
print(f"Reference vectors at scan position ({ref_col}, {ref_row}):")
print(f"  {len(unstrained_vectors)} vectors, shape: {unstrained_vectors.shape}")
print(unstrained_vectors[:, :2])  # show kx, ky columns only

# %%
# Computing the Strain Maps
# -------------------------
# ``get_strain_maps`` solves a least-squares problem at each pixel relating
# the measured vectors to the reference, then decomposes the displacement
# gradient tensor F = RU (polar decomposition) into:
#
# - **R**: rotation matrix (rigid-body rotation)
# - **U**: right stretch tensor (symmetric, captures pure strain)
#
# The returned ``StrainMap`` stores four 2D images stacked along its navigation
# axis: **e11** (εxx), **e22** (εyy), **e12** (εxy), and **θ** (rotation angle).
# ``return_residuals=False`` skips saving the per-pixel fitting residuals.

strain_maps = first_ring.get_strain_maps(
    unstrained_vectors=unstrained_vectors,
    return_residuals=False,
)
print(strain_maps)

# %%
# Visualising the Strain Components
# ----------------------------------
# ``plot()`` navigates through the four components. Each component is a 2D
# image of the scan area.  Use the HyperSpy navigator or the ``inav`` accessor
# to step between components.

strain_maps.plot()

# %%
# Extract each component as a 2D numpy array for custom matplotlib plots.
# The navigation axis stores the four components in order: e11, e22, e12, θ.
# ``inav[i]`` selects the i-th component as a 2D image signal;
# ``.data`` gives the underlying numpy array.

eps_xx = strain_maps.inav[0].data  # εxx — normal strain along x
eps_yy = strain_maps.inav[1].data  # εyy — normal strain along y
eps_xy = strain_maps.inav[2].data  # εxy — shear strain
theta = strain_maps.inav[3].data  # θ   — rigid-body rotation (rad)

# %%
# Plot the three strain components side by side.
# The colour scale is symmetric around zero: blue = compression, red = tension.

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
clim = 0.02  # ± 2 % strain range

titles = [
    r"$\varepsilon_{xx}$ (x-normal strain)",
    r"$\varepsilon_{yy}$ (y-normal strain)",
    r"$\varepsilon_{xy}$ (shear strain)",
]

for ax, comp, title in zip(axs, [eps_xx, eps_yy, eps_xy], titles):
    im = ax.imshow(comp, cmap="RdBu_r", vmin=-clim, vmax=clim, origin="upper")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="strain")

plt.tight_layout()

# %%
# Rigid-Body Rotation Map
# ------------------------
# The antisymmetric rotation θ captures rigid-body rotation of the lattice
# relative to the reference.  In a single-crystal specimen near a precipitate,
# this maps bending of the matrix; in a polycrystalline sample it highlights
# grain boundaries.

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(theta, cmap="PiYG", vmin=-0.02, vmax=0.02, origin="upper")
ax.set_title("Rigid-body rotation θ (rad)")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rotation (rad)")
plt.tight_layout()

# %%
# Interpreting the Results
# ------------------------
# In this ZrNb dataset the precipitate has a slightly different lattice
# parameter from the Zr matrix.  At the precipitate–matrix interface the
# crystal must accommodate this mismatch, generating a localised strain field.
#
# Expected signatures:
#
# - **εxx / εyy**: positive (tensile) on one side of the interface, negative
#   (compressive) on the other — forming a "butterfly" pattern around the
#   precipitate.
# - **εxy**: shear strain concentrated at the interface corners.
# - **θ**: small rotation of the matrix lattice near the interface as it bends
#   to accommodate the mismatch.
#
# The magnitude of the strain gives an estimate of the lattice mismatch f:
#
#     f = (a_precipitate - a_matrix) / a_matrix
#
# For ZrNb alloys f is typically 1–3 % depending on Nb content.

# Summarise the strain statistics in the map.
print("Strain statistics:")
for name, comp in [("εxx", eps_xx), ("εyy", eps_yy), ("εxy", eps_xy)]:
    print(
        f"  {name}: mean = {np.nanmean(comp)*100:.2f}%,  "
        f"std = {np.nanstd(comp)*100:.2f}%,  "
        f"range = [{np.nanmin(comp)*100:.2f}%, {np.nanmax(comp)*100:.2f}%]"
    )

# %%
# Practical Tips
# ---------------
# **Detector resolution**: more pixels = more accurate peak positions = lower
# strain noise.  512 × 512 or larger is preferred for quantitative work.
#
# **Disk radius**: ``disk_r`` should closely match the actual disk size.
# Too small → noisy peaks; too large → merged peaks in dense patterns.
#
# **Reference selection**: always verify the reference by overlaying its
# vectors on the pattern.  A poor reference introduces a systematic offset
# into every strain measurement.
#
# **Precession**: adding beam precession averages over the Ewald sphere tilt
# and reduces dynamical diffraction effects, improving accuracy.  The strain
# mapping workflow is identical with precession data.
#
# **Multiple reference spots**: using more spots (larger ``max_magnitude``)
# overdetermines the system and reduces the effect of individual noisy peaks.
# However, at high |g| the disks may be weaker and the positions noisier.

# %%
# sphinx_gallery_thumbnail_number = 6
