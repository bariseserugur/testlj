import sys
import numpy as np
import os

molecule = sys.argv[1]

molecule_pdb = '/scratch/gpfs/WEBB/bu9134/NQE_Letter/system_preparation_files/PDB_FILES/one_{}.pdb'.format(molecule)
MOLECULE_ATOM_COUNT = sum(1 for line in open(molecule_pdb,'r').readlines() if any(i in line for i in ['ATOM','HETATM']))
MOLECULE_COUNT = int(np.ceil(10000/MOLECULE_ATOM_COUNT))

molin = open('mol.in','w')
molin.write('{} {} no\n'.format(molecule_pdb, MOLECULE_COUNT))
molin.close()

L = 46.4158883361 
os.system('python /scratch/gpfs/WEBB/bu9134/NQE_Letter/system_preparation_files/pdb_generator.py mol.in -avoid_overlap False -box "{} {} {}" -init random'.format(L,L,L))
os.system('rm mol.in')
