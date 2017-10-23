import numpy as np
import pymatgen as pmg
import os

def generate_pyprismatic_input(structure,destination_path,comment="Default Comment"):
        """ Generates a .xyz file on which Pyprismatic can run

        Args:

        structure: pymatgen.Structure object 
        The entire structure
        
        destination_path: str
        File to be created, must have .XYZ as the suffix
            
        comment: str
        
        """
        
        if destination_path[-4:] != '.xyz' and destination_path[-4:] != '.XYZ':
            raise IOError('File suffix is not .xyz')
        
        if os.path.exists(destination_path):
            raise IOError('The file you have specified already exists')
        
        line_1 = comment
   
        Z_list = []
        for element in structure.species:
                Z_list.append(int(element.Z))

        atomic_numbers = np.asarray(Z_list).reshape(len(structure.species),1)
        percent_occupied  = np.ones_like(atomic_numbers)
        cart_coords = structure.cart_coords
        
        # TODO Raise an error if (0,0,0) isn't a co-ord
        line_2 = [np.max(cart_coords[:,0]),np.max(cart_coords[:,1]),np.max(cart_coords[:,2])]
        ## This line comes from the concept illustrated by the sample used in Ophus's 2017 paper
        
        # TODO Get dw from Element's names
        if True:
            dw = 0
    
        debeye_waller = dw*np.ones_like(atomic_numbers)
        
        printing_array = np.hstack([atomic_numbers,cart_coords,percent_occupied,debeye_waller])
        
        with open(destination_path, 'a') as f:
            print(line_1,file=f)
            print('    {0:.3g}   {1:.3g}   {2:.3g}'.format(line_2[0],line_2[1],line_2[2]),file=f)
            for row in printing_array:
                    print('{0:.3g} {1:.4f} {2:.4f} {3:.4f} {4:.3f} {5:.3f}'.format(
                            row[0],row[1],row[2],row[3],row[4],row[5]),
                        file=f)
            print("-1",file=f)
        return None
