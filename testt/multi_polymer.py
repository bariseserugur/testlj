import os
import numpy as np
import sys

mol_in = open('mol.in','w')

for i in open('../../mixed_parameters.txt','r').readlines():
    exec(i)
    if 'mass_dict' in i:
        break

DH = 76
n_monomers = 50
poly_type = str(sys.argv[1]) 

water_mass = mass_dict['WWW']
polymer_mass = (n_monomers * DH / 100 ) * mass_dict['AAA'] + (n_monomers * (100-DH) / 100 ) * mass_dict['BBB']

POLYMER_COUNT = 50
for i in range(POLYMER_COUNT):
    os.system('python ../../system_generation/make_polymer.py {} {} 180 {}'.format(poly_type,n_monomers,DH))
    os.system('mv polymer_random.pdb poly_{}.pdb'.format(i))
    mol_in.write('poly_{}.pdb 1 no\n'.format(i))

total_polymer_bead_count = int(POLYMER_COUNT * n_monomers)

WATER_CONC = 0.9
water_count = int(np.ceil((WATER_CONC * polymer_mass * POLYMER_COUNT) / (1 - WATER_CONC) / water_mass))

V_per_bead = 691.2

V = V_per_bead * (total_polymer_bead_count + water_count)
L = V**(1/3)

mol_in.write('/scratch/gpfs/WEBB/bu9134/LJ/system_generation/pdbs/WWW.pdb {} no\n'.format(water_count))

mol_in.close()
print(L)
os.system("python ../../system_generation/pdb_generator.py mol.in -box '{} {} {}' -avoid_overlap False".format(L,L,L))
os.system('rm poly_*.pdb')
