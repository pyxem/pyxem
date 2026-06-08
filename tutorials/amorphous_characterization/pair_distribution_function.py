"""
# Pair Distribution Function Analysis

The **pair distribution function** (PDF), g(r), gives the probability of finding
an atom at distance r from a reference atom.  Unlike conventional diffraction,
the PDF contains information about local atomic structure even in disordered or
amorphous materials where Bragg peaks are absent.

The reduced intensity I_red(s) is obtained by subtracting the independent atomic
scattering from the total diffracted intensity, where s = 2 sin(θ)/λ is the
scattering variable.  The PDF is then the sine transform of I_red(s).

This tutorial demonstrates:

1. Creating a 1D electron diffraction intensity profile
2. Fitting and subtracting the independent atomic scattering baseline
3. Applying damping functions to reduce termination ripples
4. Computing the pair distribution function
5. Interpreting the PDF in terms of atomic pair distances

Reference: S. J. L. Billinge & M. G. Kanatzidis, *Chem. Comm.* 2004, 749.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pyxem as pxm

# %%
# Creating a Synthetic 1D Diffraction Profile
# --------------------------------------------
# We construct a realistic 1D diffraction profile for amorphous SiO₂.
# In a real experiment this would come from azimuthal integration of a
# 4D-STEM or powder diffraction dataset.
#
# The profile has broad peaks at scattering vectors characteristic of
# amorphous silica, superimposed on a smoothly decaying background from
# the independent atomic scattering.

n_pts = 400
s_min, s_max = 0.01, 9.5  # scattering vector s = 2sinθ/λ in Å⁻¹
s_axis = np.linspace(s_min, s_max, n_pts)
scale = (s_max - s_min) / (n_pts - 1)

# Amorphous SiO₂ broad diffraction peaks (approximate positions in Å⁻¹)
peaks = [
    (0.40, 0.07, 1600),  # first sharp diffraction peak
    (0.72, 0.09, 1000),
    (1.05, 0.10, 550),
    (1.50, 0.12, 280),
    (2.00, 0.15, 140),
    (2.60, 0.18, 70),
]
intensity = sum(A * np.exp(-0.5 * ((s_axis - c) / w) ** 2) for c, w, A in peaks)
# Smooth decaying background from independent atomic scattering
intensity += 3000 * np.exp(-0.4 * s_axis) + 20

# Wrap in an ElectronDiffraction1D signal.
# The generator requires a (ny, nx | s) shape, so we add two navigation axes.
rp = pxm.signals.ElectronDiffraction1D([[intensity]])
rp.axes_manager[-1].scale = scale
rp.axes_manager[-1].offset = s_min
rp.axes_manager[-1].units = "1/Å"
rp.axes_manager[-1].name = "s"
print(rp)
rp.sum().plot()
plt.xlabel("s (Å⁻¹)")
plt.ylabel("Intensity (a.u.)")
plt.title("Synthetic 1D diffraction profile of amorphous SiO₂")

# %%
# Reduced Intensity: Fitting Atomic Scattering
# ---------------------------------------------
# The ``ReducedIntensityGenerator1D`` subtracts the independent (incoherent)
# atomic scattering to isolate the structural signal.
#
# We specify the composition as element symbols and molar fractions.
# For SiO₂: Si occupies 1/3 and O occupies 2/3 of atomic sites.

rigen = pxm.generators.ReducedIntensityGenerator1D(rp)

elements = ["Si", "O"]
fractions = [1 / 3, 2 / 3]

rigen.fit_atomic_scattering(
    elements,
    fractions,
    scattering_factor="lobato",
    plot_fit=True,
    iterpath="serpentine",
)
plt.title("Fitted atomic scattering baseline")
plt.xlabel("s (Å⁻¹)")

# %%
# Once the fit looks good, retrieve the reduced intensity.

ri = rigen.get_reduced_intensity()
print(ri)
ri.sum().plot()
plt.xlabel("s (Å⁻¹)")
plt.ylabel("Reduced intensity I_red(s)")
plt.title("Reduced intensity of amorphous SiO₂")

# %%
# Setting the s-range
# -------------------
# Restricting the s-range removes artefacts at very low s (direct beam, noise)
# and at high s (where the signal-to-noise ratio is poor).

rigen.set_s_cutoff(s_min=0.25, s_max=8.5)
ri_cut = rigen.get_reduced_intensity()
ri_cut.sum().plot()
plt.xlabel("s (Å⁻¹)")
plt.ylabel("Reduced intensity")
plt.title("Reduced intensity with s-range cutoff")

# %%
# Damping Functions
# -----------------
# Before Fourier transformation, a **damping function** is applied to suppress
# the high-s oscillations that would otherwise produce Gibbs ripples (termination
# ripples) in the final PDF.  Three common choices are shown below.

# Get fresh reduced intensity for each damping test.
ri_exp = rigen.get_reduced_intensity()
ri_lorch = rigen.get_reduced_intensity()
ri_erfc = rigen.get_reduced_intensity()

ri_exp.damp_exponential(b=0.15)
ri_lorch.damp_lorch(s_max=8.5)
ri_erfc.damp_low_q_region_erfc(offset=4)

s_vals = ri_exp.axes_manager[-1].axis
fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for ax, ri, label in zip(
    axs,
    [ri_exp, ri_lorch, ri_erfc],
    ["Exponential", "Lorch", "Low-q ERFC"],
):
    ax.plot(s_vals, ri.inav[0, 0].data)
    ax.set_title(label)
    ax.set_xlabel("s (Å⁻¹)")
axs[0].set_ylabel("Damped reduced intensity")
plt.tight_layout()

# %%
# Computing the PDF
# -----------------
# The ``PDFGenerator1D`` performs the sine Fourier transform of the damped
# reduced intensity to produce g(r).  Peaks in g(r) correspond to preferred
# inter-atomic distances.

ri_final = rigen.get_reduced_intensity()
ri_final.damp_exponential(b=0.15)

pdfgen = pxm.generators.PDFGenerator1D(ri_final)

s_min_pdf = 0.25  # Å⁻¹ — exclude direct beam
s_max_pdf = 8.5  # Å⁻¹
r_max = 10.0  # Å  — maximum real-space distance

pdf = pdfgen.get_pdf(s_min=s_min_pdf, s_max=s_max_pdf, r_max=r_max)
print(pdf)

# %%
# Plot the PDF.
# The first peak at ~1.6 Å corresponds to the nearest-neighbour Si-O bond.
# The second peak at ~2.6 Å is the O-O distance in SiO₄ tetrahedra.
# The third peak at ~3.1 Å is the Si-Si next-nearest-neighbour distance.

pdf.sum().plot()
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.title("Pair distribution function of amorphous SiO₂")

# %%
# Add vertical guide lines at known bond distances.
known_distances = {
    "Si-O (1.61 Å)": 1.61,
    "O-O (2.63 Å)": 2.63,
    "Si-Si (3.12 Å)": 3.12,
}
for label, dist in known_distances.items():
    plt.axvline(dist, linestyle="--", alpha=0.7, label=label)
plt.legend(fontsize=8)

# sphinx_gallery_thumbnail_number = 5
