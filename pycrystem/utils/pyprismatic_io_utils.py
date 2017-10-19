import numpy as np
import pymatgen as pmg

def generate_pyprismatic_input(unit_cell_structure,scaling_for_supercell=1,comment="Default Comment"):
	""" Generates a .xyz file on which Pyprismatic can run

	Args:

	unit_cell_structure:

	scaling: 

	comment:str

	"""

	lattice_params = unit_cell_structure.lattice.abc
	#now scale up
	structure = unit_cell_structure.make_supercell(scaling_for_supercell)
	line_1 = comment
	line_2 = lattice_params
	
	# TODO vectorise this
	Z_list = []
	for element in structure.species:
    		Z_list.append(element.Z)

    dw = 0.076
	atomic_numbers = np.asarray(Z_list).reshape(len(structure.species),1) 
	cart_coords = structure.cart_coords
	percent_occupied  = np.ones_like(atomic_numbers)
	debeye_waller = dw*np.ones_like(atomic_numbers)
	# combines the rest of what needs to come out
	printing_array = np.hstack([atomic_numbers,cart_coords,percent_occupied,debeye_waller])

	print(line_1)
	print(line_2)
	print(printing_array)
	print("-1")
