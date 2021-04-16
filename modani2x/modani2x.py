import torch
import argparse
import torchani
from ase.optimize import BFGS, LBFGS, FIRE, MDMin
import numpy as np
import glob
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.io import read, write
import os
import warnings
warnings.filterwarnings("ignore")
from datetime import  datetime, date
import sys, getopt



time = datetime.now()
date= date.today()



ChemiToInts={'H': 1,
             'C': 6,
             'Fe':26,
             'F': 9,
             'Cl':17, 
             'N': 7, 
             'O': 8,
             'S': 16}

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
    
    
    
        
        

def printer(id,ligand_id, ener):
    pid = id[:-4]  
    if not os.path.exists("run"): os.makedirs("run")  
    with open(f"./run/{pid}_MINI_log_minimized.txt", "a") as ani_out:  
        #print(f"Today's date: {time} ", file = ani_out)
        #print("         ##############################################", file = ani_out)
        #print("         ############# RESULT #########################", file = ani_out)
        #print("         ##############################################", file = ani_out)
        print(' {}\t{}\tEnergy: {}'.format(id, ligand_id,  ener), file = ani_out)
        
        


def parser(name, crystal=False):
    """A function to parse the information from molecules.

    Args:
        name (molecule): A molecule name with extension
        crystal (bool, optional): Search for crsytal molecules with specific name consisting of *_crystal.extendion.
        Defaults to True.

    Returns:
        id: Name of the molecules parsed.
        xyz: xyz coordinates.
        index: index of the atoms i.e atomic number.
    """

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

            return id, xyz, index 
        
    except FileError as error:
        error.fileformat()
    
    
def printenergy(a):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
        'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))
        




def rmsd_calculator(crystal, conformer):
    """Calculates RMSD between reference and pose.

    Args:
        crystal (list): A list of xyz coordinates of coordinates
        conformer (list): A list of xyz coordinates of pose.

    Returns:
        float: RMSD value
    """
    
    
    
    if  len(crystal) == len(conformer):
        N = len(crystal)
        diff=np.array(crystal)-np.array(conformer)
        rms = np.sqrt((diff*diff).sum()/N)
        return rms
    else:
        print("Length of crystal and model doesnt match")
    


def optimization(id,xyz, atoms, ALGO, num_step):

    """Energy Minimization  of small molecule
    Args:
        id: Name/id of the molecule.
        xyz: xyz Coordinates of the molecules.
        atoms: Atoms of the molecules.
        ALGO: Name of minimization ALGOrithm to use for optimization.
        num_step: Number of steps to perform by the optimization ALGOrithm.
    Returns:
        optimized_xyz: A list containing the optimized coordinates in xyz.
        optimized_index: A list of indexes of atoms from periobic table/atomic number. 
        
    Note:
        trajectory: Writes trajectory file in trajectory folder.
        xyz: Writes minimzed xyz file in minimized folder.
    """
    
    atoms = Atoms(atoms, positions= xyz)
    atoms.set_calculator(calculator)
    if not os.path.exists('trajectory'): os.makedirs('trajectory')
    if ALGO =="BFGS" : 
        opt = BFGS(atoms, trajectory=f"./trajectory/{id}_BFGS_minimization.traj")
    elif ALGO == "FIRE":
        opt = FIRE(atoms, trajectory=f"./trajectory/{id}_FIRE_minimization.traj")
    elif ALGO == "LBFGS":
        opt = LBFGS(atoms, trajectory=f"./trajectory/{id}_LBFGS_minimization.traj")   
    else:
        print("Unknown minimization ALGOrithm.")
        usage()
        exit()
        
    opt.run(steps=num_step)
    if not os.path.exists('minimized'): os.makedirs('minimized')
    write(f"./minimized/{id}_minimized.xyz", atoms)
    optimized_xyz = atoms.get_positions()
    optimized_index = atoms.get_atomic_numbers()

    return optimized_xyz, optimized_index




def folder_search(target=None,search_type =None):
    """Search for required file format when enter a main directory. targted file 
    format can be used otherwise deafult searches for pdb as protein and xyz as ligand
    Usage sample: folder_serch(search_type="mol2", taget="protein". """

    files = sorted(glob.glob(f"./**/{target}.{search_type}", recursive = True)) 
    print("Total number of files found: " + str(len(files)))
    return files


def calculation(protein, ligand, ALGO=False,verbose=False, steps=0, log=None):  


    def file_format(molecule):
        molecule_id = os.path.basename(molecule)
        molecule_name, molecule_format = molecule_id.rsplit(".")
        return molecule_name, molecule_format
    
    pid, pformat = file_format(protein)
    lid,lformat = file_format(ligand)
    
    
    found_protein = folder_search(pid,pformat)
    found_ligand = folder_search(lid, lformat)
    print("\nStarted ......")

    protein_id, protein_xyz, protein_index = parser(found_protein)
    
    ligand_id, ligand_xyz,ligand_index,ligand_atoms, ligand_heavy_xyz = parser(found_ligand, False)
    
    
    
    if ALGO!= False:
        ALGOrithm = ALGO    
        num_steps = int(steps)
        optimized_xyz , optimized_index  = optimization(ligand_id[:-4], ligand_xyz, ligand_atoms,ALGOrithm, num_steps)
        xyz_4_tensor = protein_xyz + list(optimized_xyz)
        index_4_tensor = protein_index + list(optimized_index)
        optimized_xyz_array = np.asarray(optimized_xyz)
        optimized_heavy_atoms_index = [i for i, j in enumerate(optimized_index)if j != 1]
        optimized_heavy_atoms_xyz = (np.array(optimized_xyz_array)[optimized_heavy_atoms_index])
        #conformer=np.asarray(optimized_heavy_atoms_xyz)  
    else:
        xyz_4_tensor = protein_xyz + ligand_xyz
        index_4_tensor = protein_index +ligand_index
        #conformer = np.asrray(ligand_heavy_xyz)
    
    
    #len_xyz_tensor = len(xyz_4_tensor)         
    #xyz_tensor_ = torch.FloatTensor(xyz_4_tensor)
    #xyz_tensor=xyz_tensor_.reshape(len_xyz_tensor,3).unsqueeze(dim=0)
    #atom_index_ = torch.tensor(index_4_tensor)     
    #atom_index = atom_index_.reshape(len_xyz_tensor).unsqueeze(dim=0)
    #coordinates = torch.tensor(xyz_tensor, requires_grad= True, device=device)
    #species = torch.tensor(atom_index,device=device)

    
    complex = Atoms(index_4_tensor, positions=xyz_4_tensor)
    complex.set_calculator(calculator)
    print("calculating.....")
    energy = complex.get_potential_energy() 
    ener = ("single point energy: " + str(energy))
    print(ener)

    #rmsd= rmsd_calculator(crystal, conformer)

    #if verbose == True:
    #    print('\n{}\t + \t{}'.format(crystal_id[:-4], ligand_id[:-4]))
    #    print("---------------------end-----------------------------------------")
    #
    printer(protein_id, ligand_id[:-4], energy)
    



def usage():
    
    print(r"""Usage information ~

Input:
  	-r [ --receptor ] arg         rigid part of the receptor (PDB)
  	-l [ --ligand ] arg           ligand(xyz)
  	-m [ --minimzation ] arg      BFGS/LBFGS/FIRE
  	-s [ --steps ] arg            (optional) default: 1000
  	-o [ --output ] arg           (optional) default: default_log.txt
  	-v [ --verbose ] arg          (option) True/False default: False
Information: 
	-h [ --help ] 	              display usage summary
  	""")

    return 0



def main(argv):

    try:
        ALGO = False
        step = 1000
        output_file = "default_log.txt"
        verbose = False
        ligand = ''
        ALGORITHMS = ["BFGS", "LBFGS", "FIRE"]

        opts, argv= getopt.getopt(argv,"hr:l:m:s:o:v",["receptor=","minization=", "liagnd=","steps=","output="])
        

        if len(opts) <2 and opts not in ("h", "--help"):
            print("Error: Atleast 2 arguments is required.")
            usage()
            exit()

        for opt, arg in opts:

            if opt in ("-h", "--help"):
                usage()
                exit()

            elif opt in ("-r", "--receptor"):
                protein = arg
                try:
                    if protein[-3:].lower() == "pdb" or len(protein) > 7: #pdb_id+.pdb equals to 7 atleast
                        pass
                except Exception as e:    
                    FileError.fileformat(e)
                    exit()
                    
            elif opt in ("-l", "--ligand"):
                ligand = arg
                
                try:
                    if ligand[-3:].lower() == "xyz" or len(ligand) > 5:
                        pass #something+.pdb equals to 5 atleast
                    
                except Exception as e:   
                    FileError.fileformat(e)
                    exit()
            
            elif opt in ("-m", "--minimzation") and arg in ALGORITHMS:
                ALGO = arg
            
            elif opt in ("-s", "--steps"):
                    step = int(arg)
            
            elif opt in ("-o", "--output"): 
                output_file = arg

            elif opt in ("-v", "--verbose"):
                verbose = True

            else:
                assert False, f"unhandled option: {opt}" 

    except getopt.GetoptError as err:
        print(err)
        print ("\nUse --help option to view usage information.")
        sys.exit(2)    
    

    calculation(protein, ligand, ALGO, verbose = verbose, steps= step, log =output_file)

if __name__ == "__main__":
    main(sys.argv[1:])