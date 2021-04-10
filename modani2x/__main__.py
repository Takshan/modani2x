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
                 'S': 16}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchani.models.ANI2x(periodic_table_index=True).to(device)
    calculator = torchani.models.ANI2x().ase()

    ligand = sys.argv[-1]
    protein = sys.argv[-2]   
    main(protein, ligand)