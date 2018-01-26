import numpy as np
import pymatgen as pmg
import os
import pyprismatic as pr

def generate_pyprismatic_input(structure,delete_mode=False):
        """     Generates a .XYZ file on which Pyprismatic can run
        
        Args:
            structure: pymatgen.Structure object 
                The entire structure
        Returns:
            None
            file 'PP_input.XYZ'
        
        """
        if delete_mode:
            ##remove the file
            pass
        if os.path.exists('PP_input.XYZ'):
            raise IOError('The file the program is attempting to write to already exists, please remove it.')
            #If one turned this warning off extra data would be added to the existing file (NOT a simple overwrite) 
        
        atomic_numbers = np.asarray([S.Z for S in structure.species]).reshape(len(structure.species),1)
        cart_coords = structure.cart_coords
        percent_occupied  = np.ones_like(atomic_numbers)
        # TODO Get debeye-wallers from elements' names
        debeye_waller = np.zeros_like(atomic_numbers)
        printing_array = np.hstack([atomic_numbers,cart_coords,percent_occupied,debeye_waller])
        
        try:
            unit_cell_size = [lattice_vector for lattice_vector in structure.lattice.abc]
        ### exception handling to deal with non-periodic structures
        
        with open('PP_input.XYZ', 'a') as f:
            print("Default Comment",file=f)
            print('    {0:.3g}   {1:.3g}   {2:.3g}'.format(unit_cell_size[0],unit_cell_size[1],unit_cell_size[2]),file=f)
            for row in printing_array:
                    print('{0:.3g} {1:.4f} {2:.4f} {3:.4f} {4:.3f} {5:.3f}'.format(
                            row[0],row[1],row[2],row[3],row[4],row[5]),
                        file=f)
            print("-1",file=f)
        return None
        
def run_pyprismatic_simulation(prismatic_kwargs=None):
    if prismatic_kwargs == None:
        prismatic_kwargs = {}
    if 'filenameAtoms' not in prismatic_kwargs.keys():
        prismatic_kwargs.update({'filenameAtoms':"PP_input.XYZ"})
    if 'filenameOutput' not in prismatic_kwargs.keys():
        prismatic_kwargs.update({'filenameOutput':"PP_output.mrc"})
    #print(prismatic_kwargs)
    meta = pr.Metadata(**prismatic_kwargs) ##Sticks to defaults apart from the unpacked dict
    meta.go()
    # TODO Clean up here so that users don't have loads of files floating around
    return meta

def import_pyprismatic_data(meta):
    # XXX
    pass
    
