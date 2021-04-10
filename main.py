import glob
import sys
import os
from progress.bar import Bar


import modani2x.modani2x as ani # type: ignore



folder_path= "/home/lab09/Projects/glide_verification"


def folder_search(search_type =None, target=None, crystal=False):
    """Search for required file format when enter a main directory. targted file 
    format can be used otherwise deafult searches for pdb as protein and xyz as ligand
    Usage sample: folder_serch(search_type="mol2", taget="protein". """
    
    #proteins = sorted(glob.glob(f"{folder_path}/**/*protein.pdb", recursive = True)) 
    # #search for pdbqt files and get the list for nxt
    #print("Number of proteins: " + str(len(proteins)))
    
    if crystal == False: 
        ligands = sorted(glob.glob(f"{folder_path}/**/*{target}.{search_type}", recursive = True)) 
    else:  
        ligands = sorted(glob.glob(f"{folder_path}/**/poses/*{target}.{search_type}", recursive = True)) 
    print("Total number of files found: " + str(len(ligands)))
    return ligands



def give_id(file):
    """Function to return the main file name excluding "." extension.

    Args:
        file (list): Name of file with "." extenasion.

    Returns:
        Name: Name without extension.
    """
    
    file_name = os.path.basename(file)
    file_name = file_name.rsplit('.')[0]
    
    return file_name



def main():
    
    """Functions list below, run the required specific function only. 
    However main search function maybe required for most of the other functions."""
    
    
    print(r"""   
        Started TorchANI2x Data Preparation.........
        """)
    ligands_list = folder_search(search_type="xyz", target="output*", crystal=True)
    proteins_list =  folder_search(search_type="pdb", target="protein", crystal=False)
    
    #print(ligands_list[1], proteins_list[1])
    count = 0
    for protein in Bar("Calculating...").iter(proteins_list):
        protein_name = give_id(protein)[:4].lower()
        for ligand in Bar("Enumerating.....").iter(ligands_list):
            ligand_name = give_id(ligand)[:4].lower()
            if protein_name == ligand_name:
                ani.main(protein, ligand)
                count += 1
                if count ==2: break
                
                
if __name__ == '__main__':
    main()