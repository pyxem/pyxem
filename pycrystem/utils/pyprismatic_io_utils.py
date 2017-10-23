import numpy as np
import pymatgen as pmg

def generate_pyprismatic_input(structure,comment="Default Comment"):
        """ Generates a .xyz file on which Pyprismatic can run

        Args:

        structure: pymatgen.Structure object 
        The entire structure

        comment: str
        
        """
        lattice_params = structure.lattice.abc
        
        line_1 = comment
        line_2 = lattice_params #This need a little bit of thoughts
        
        
        Z_list = []
        for element in structure.species:
                Z_list.append(int(element.Z))

        # TODO Get dw from Element's names
        if True:
            dw = 0
    
        atomic_numbers = np.asarray(Z_list).reshape(len(structure.species),1)
        cart_coords = structure.cart_coords
        percent_occupied  = np.ones_like(atomic_numbers)
        debeye_waller = dw*np.ones_like(atomic_numbers)
        
        printing_array = np.hstack([atomic_numbers,cart_coords,percent_occupied,debeye_waller])

        print(line_1)
        print('    {0:.3g}   {1:.3g}   {2:.3g}'.format(line_2[0],line_2[1],line_2[2]))
        for row in printing_array:
                print('{0:.3g} {1:.4f} {2:.4f} {3:.4f} {4:.3f} {5:.3f}'.format(row[0],row[1],row[2],row[3],row[4],row[5]))
        print("-1")
        return None
