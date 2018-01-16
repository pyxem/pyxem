pyxem roadmap
=================

pyxem-v0.1
--------------

Conventions : check axes alignments, Bunge convention etc. write up for docs
Simulations : update atomic scattering factors to Doyle-Turner
Documentation : write documentation
Testing : coverage to >75% using pytest
Template Matching : cubochoric grid generation, symmetry constraints, alternative
Peak Finding : include routine for associating peaks with g-vectors and calculating orientations
Strain Mapping : 3xmethods plus written up
HyperSpy Dependancy : clean hyperspy dependancy to minor release V1.3
Demos : create notebooks illustrating usage and key functionalities

horizons
--------

Simulations : add multi-slice and Bloch options
Kikuchi Patterns : add support for Kikuchi pattern based Mapping
Orientation Mapping : smarter workflows for getting "correct" solution
Strain Tomography

current bugs
------------

- Correlation over a single image doesn't work. Possibly an error in HyperSpy?
