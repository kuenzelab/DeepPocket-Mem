'''This script aligns membrane embedded protein part from the OPM database with the cleaned protein.'''

from pymol import cmd
import sys
import os

cleaned = sys.argv[1]
membrane = sys.argv[2]
pdb_code = os.path.basename(cleaned).split('.')[0][0:4]
output_pdb_file = pdb_code + "_consurf_2use.pdb" 



cmd.load(cleaned, 'structure1')
cmd.load(membrane, 'structure2')

cmd.align('structure2', 'structure1')
cmd.select('overlap', 'byres structure1 within 2.0 of structure2')

cmd.remove('structure1 and not overlap')


cmd.save(output_pdb_file, 'structure1')
