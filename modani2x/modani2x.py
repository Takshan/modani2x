import torch
import torchani
from ase.optimize import BFGS, LBFGS
import numpy as np
import glob
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.io import read, write
import os
import warnings
warnings.filterwarnings("ignore")
from datetime import  datetime, date
import sys


time = datetime.now()
date= date.today()


ChemiToInts={'H': 1,
            'C': 6,
            'Fe':26,
            'F': 9,
            'Cl':17, 
            'N': 7, 
            'O': 8,
            'S': 16
            }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchani.models.ANI2x(periodic_table_index=True).to(device)
calculator = torchani.models.ANI2x().ase()


class FileError(Exception):
    def __init__(self, msg):
        self.msg=msg
        
    def fileformat(self):
        print("Unrecognized file format. Please use xyz for pose and pdb for protein." , self.msg)
        
    def filenotfound(self):
        print("File not Found. Check if directory or file exits", self.msg)
    
    def mismatch_length(self):
        print("Entered file lenght doesn't match. Please enter equal length list/array", self.msg)
        
    def mismatch_protein_ligand(self):
        print("Entered Protein and ligand name didn't seems to be same. please check! ", self.msg)
    
    
    
        
        

def printer(id, crystal_id,ligand_id, rmsd, ener):
    pid = id[:-4]    
    with open(f"{pid}_v7_log_minimized.txt", "a") as ani_out:  
        #print(f"Today's date: {time} ", file = ani_out)
        #print("         ##############################################", file = ani_out)
        #print("         ############# RESULT #########################", file = ani_out)
        #print("         ##############################################", file = ani_out)
        print('{}\t{}\tRMSD: {} A {}'.format(crystal_id, ligand_id, rmsd, ener), file = ani_out)
        
        


def parser(name, crystal=True):
    ''' path as the input'''
    file = ''.join(name)
    id = os.path.basename(file)
    index =[]
    xyz = []
    atoms = []
    try:
        if id[-3:].lower() == "xyz":
            with open(file, 'r') as ligfile:
                for line in  ligfile:
                    if line[:2].strip() in ChemiToInts:
                        split_line = [line[9:19], line[24:34], line[39:49], line[:2]]
                        x = split_line[0].strip()
                        y = split_line[1].strip()
                        z = split_line[2].strip()
                        a = split_line[-1].strip()
                        xyz_temp = (x,y,z)
                        xyz_temp = [float(_) for _ in xyz_temp]
                        xyz.append(xyz_temp)
                        a = ''.join([i for i in a if not i.isdigit()])
                        atoms.append(a.title())
                        if a.title() in ChemiToInts:
                            index.append(ChemiToInts.get(a.title()))
                
                heavy_atoms_index = [i for i, j in enumerate(atoms)if j != "H"]
                heavy_atoms_xyz = (np.array(xyz)[heavy_atoms_index])
            
            
                if crystal== True:
                    return id,heavy_atoms_xyz            
                else:
                    return id, xyz, index, atoms, heavy_atoms_xyz

        elif id[-3:].lower() == "pdb":
            with open(file, 'r') as pdbfile:
                for line in pdbfile:
                    if line[:4]=="ATOM" :
                        splitted_line = line[31:38], line[39:46], line[47:54], line[76:78]
                        x = splitted_line[0].strip()
                        y = splitted_line[1].strip()
                        z = splitted_line[2].strip()
                        i = (splitted_line[-1].strip())
                        xyz_ =(x,y,z)
                        xyz_ = [float(_) for _ in xyz_]
                        xyz.append(xyz_)
                        i =''.join([x for x in i if not x.isdigit()])
                        atoms.append(i.title())
                        #print(i)
                        if i.title() in ChemiToInts:
                            index.append(ChemiToInts.get(i))

            return id, xyz, index, atoms 
        
    except FileError as error:
        error.fileformat_exception()
    
    
def printenergy(a):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
        'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
        

def rmsd_calculator(crystal, conformer):
    if  len(crystal) == len(conformer):
        N = len(crystal)
        diff=np.array(crystal)-np.array(conformer)
        rms = np.sqrt((diff*diff).sum()/N)
        return rms
    else:
        print("Length of crystal and model doesnt match")
    


def calculation(proteins, ligands):    
    lig_id = ligands.lower()
    prot_id = proteins.lower()
    
    ###disenable for now
    
    #protein = glob.glob(f"../{prot_id}/**/{prot_id}/*.pdb")
    #ligand = glob.glob(f"../{prot_id}/**/{prot_id}/{lig_id}")
    
    #crystal_path = glob.glob(f"../{prot_id}/**/{prot_id}/{prot_id}*_crystal.xyz")
    #crystal_id, crystal_xyz= parser(crystal_path,True)
    #crystal = np.asarray(crystal_xyz)
    
    #print(protein)
    protein_id, protein_xyz, protein_index, protein_atoms = parser(proteins)
    


    ligand_id, ligand_xyz,ligand_index,ligand_atoms, ligand_heavy_xyz = parser(ligands, False)
    #atoms = Atoms(ligand_atoms, positions=ligand_xyz)  
    
    #complex_atoms = protein_atoms + ligand_atoms
    #complex_position = protein_xyz + ligand_xyz
    atoms = Atoms(ligand_atoms, positions =ligand_xyz)





    atoms.set_calculator(calculator)
    
    """Energy minimization"""
    lig_id = os.path.basename(ligands)
    lig_id = lig_id.rsplit(".")[0]
    
    #print("Strating minimization of ligands.......")
    #print("---------------------------------------------------------------------------")
    opt=BFGS(atoms, trajectory=f"/mnt/d/lab/Projects/glide_verification/scripts/run/minimization_dump/{lig_id}_minimization.traj")
    opt.run(fmax=0.001,steps=1000)
    
    #print(
    '''optimized ligands'''    
    
    optimized_xyz = atoms.get_positions()
    optimized_index = atoms.get_atomic_numbers()
    
    
    #print(len(optimized_xyz), len(optimized_index))
    xyz_4_tensor = protein_xyz + list(optimized_xyz)
    index_4_tensor = protein_index + list(optimized_index)
    
    #len_xyz_tensor = len(xyz_4_tensor)         
    #xyz_tensor_ = torch.FloatTensor(xyz_4_tensor)
    #xyz_tensor=xyz_tensor_.reshape(len_xyz_tensor,3).unsqueeze(dim=0)
    #atom_index_ = torch.tensor(index_4_tensor)     
    #atom_index = atom_index_.reshape(len_xyz_tensor).unsqueeze(dim=0)
    #coordinates = torch.tensor(xyz_tensor, requires_grad= True, device=device)
    #species = torch.tensor(atom_index,device=device)
    
    
    
    
    '''non optimized structure'''
    #xyz_4_tensor = protein_xyz + ligand_xyz
    #index_4_tensor = protein_index + ligand_index  #index=atomic numbers
    
    complex = Atoms(index_4_tensor, positions=xyz_4_tensor)
    complex.set_calculator(calculator)
    ener = ("single point energy: " + str(complex.get_potential_energy()))
    with open(f"/mnt/d/lab/Projects/glide_verification/scripts/result/{protein_id}_log.txt", "w") as logout:
        print(f"{protein_id}\t {ligand_id[:-4]}\t {ener}", file = logout )
    
    
    
    #print(ener)
    #optimized_xyz_array = np.asarray(optimized_xyz)
    #optimized_heavy_atoms_index = [i for i, j in enumerate(optimized_index)if j != 1]
    #optimized_heavy_atoms_xyz = (np.array(optimized_xyz_array)[optimized_heavy_atoms_index])
        
    
    #conformer=np.asarray(optimized_heavy_atoms_xyz)  
    #rmsd= rmsd_calculator(crystal, conformer)

    #print('\n{}\t + \t{}==>\tRMSD: {} A'.format(crystal_id[:-4], ligand_id[:-4], rmsd))
    #print("---------------------end-----------------------------------------")
    
    #printer(protein_id,crystal_id[:-4], ligand_id[:-4], rmsd, ener)
    


def main(protein, ligand):
    
    """ Runs a script for analyzing protein ligand binding.
    """
    
    
    print("\n\nRuning Mod ANI_2x....................................\n\n")
    #print(protein, ligand)
    calculation(protein, ligand)
    
    

if __name__ == '__main__':
    print("\n\nStarted Mod_ANI_2x.....................", end="\n\n")
    ligand = sys.argv[-1]
    protein = sys.argv[-2]
    main(protein, ligand)