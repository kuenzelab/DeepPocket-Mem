"""This script calculates lipid RSCC"""

import sys
import csv
import os

filename = sys.argv[1]
pdb_code = os.path.basename(filename).split('.')[0][0:4]
output_file = f"{pdb_code}_rscc.csv"

lipid_name = "PLM" # lipid name
output_file = f"{pdb_code}_rscc.csv"

lipid_rscc = []

with open(filename) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 6:
            chain, resname, resid, rscc, volume, occupancy = parts
            if resname == lipid_name:
                lipid_rscc.append((chain, int(resid), float(rscc)))


if lipid_rscc:
    for chain, resid, rscc in lipid_rscc:
        print(f"Chain {chain}, Residue {resid}: RSCC = {rscc:.3f}")
    avg = sum(x[2] for x in lipid_rscc) / len(lipid_rscc)
    

   
    with open(output_file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Chain", "Residue", "RSCC"])
        writer.writerows(lipid_rscc)
        writer.writerow([])
        writer.writerow(["Average", "", f"{avg:.3f}"])

    
