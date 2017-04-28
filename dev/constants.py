import numpy as np
import pycrystem as pc

DISTORTION = np.array(
    [
        [1.45, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00]
    ]
)

cubic = pc.Lattice.cubic(5.65)
coords = [
    [0.0,  0.0,  0.0 ],
    [0.5,  0.5,  0.0 ],
    [0.5,  0.0,  0.5 ],
    [0.0,  0.5,  0.5 ],
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
]
atoms = [
    "Ga",
    "Ga",
    "Ga",
    "Ga",
    "As",
    "As",
    "As",
    "As",
]
GAAS = pc.Structure(cubic, atoms, coords)

cubic = pc.Lattice.cubic(3.52)
coords = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
]
atoms = [
    "Ni",
    "Ni",
    "Ni",
    "Ni",
]
NICKEL = pc.Structure(cubic, atoms, coords)