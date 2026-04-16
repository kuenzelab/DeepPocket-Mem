import os
import sys
import numpy as np
from rdkit.Chem import AllChem as Chem
import glob


fpocket_path = "./fpocket/"
ligand_dir = "./ligands/"


def load_ligand_atoms(prot_id, ligand_dir):
    ligand_pattern = os.path.join(ligand_dir, f"{prot_id}_lig*.pdb")
    ligand_files = glob.glob(ligand_pattern)
    atom_nps = []
    for lig_file in ligand_files:
        mol = Chem.MolFromPDBFile(lig_file, sanitize=False)
        c = mol.GetConformer()
        atom_np = c.GetPositions()
        atom_nps.append(atom_np)
    if len(atom_nps) == 0:
        return None
    # all ligand atoms positions into one array
    all_positions = np.vstack(atom_nps)
    return all_positions


def types_from_file(f, holo_f, ligand_dir):
    distance = 4
    with open(f, 'r') as prot_file:  
        for line in prot_file.readlines():
            prot = line.strip().replace('_nowat', '')
            atom_np = load_ligand_atoms(prot, ligand_dir)
            centers = np.loadtxt(os.path.join(fpocket_path, prot + "_nowat_out/pockets/bary_centers.txt"))
            # when there is only one pocket => shape is (4,)
            if centers.ndim == 1 and centers.shape[0] == 4:
                centers = np.expand_dims(centers, axis=0)
            if centers.ndim == 2 and centers.shape[1] == 4:
                sorted_centers = centers[:, 1:] 
                for i in range(sorted_centers.shape[0]):
                    center = sorted_centers[i]
                    dist = np.linalg.norm(atom_np - center, axis=1)
                    label = 1 if np.any(dist <= distance) else 0
                    holo_f.write(f"{label} {center[0]} {center[1]} {center[2]} {prot}_nowat.gninatypes\n")
             

train_prots="all.txt"
holo_train=open('all.types','w')
types_from_file(train_prots,holo_train,ligand_dir)




