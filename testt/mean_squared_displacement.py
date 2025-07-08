import numpy as np
import sys
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.waterdynamics import WaterOrientationalRelaxation as WOR
import pickle
import joblib
from joblib import Parallel,delayed
import mdtraj as md
import copy
import itertools
import time
from multiprocessing import Pool
mass_dict = {'C':12.011,'H':1.00794,'O':15.994,'Cl':35.4527,'N':14.00674,'Br':79.904, 'S': 32.066}

lammpstrj_file_name = 'cubic_pellet.lammpstrj'
lammpstrj_file = open(lammpstrj_file_name,'r')

topology = md.load('init.pdb').topology

ATOM_COUNT = topology.n_atoms
#WATER_COUNT = len([r for r in topology.residues if #r.name == 'WWW'])
POLYMER_COUNT = 
additive_residues = [[x.index for x in r.atoms] for r in topology.residues if r.name == 'WWW']

for i in range(2):
    line = lammpstrj_file.readline()
L = float(line.split()[-1])
lammpstrj_file.seek(0)

def go_to_line(file_name, line_index):
    ''' Bring the file to the indicated line. '''
    file_to_jump = open(file_name)
    if line_index != 0:
        next(itertools.islice(file_to_jump, line_index-1, None))
    return file_to_jump

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

def unwrap_coordinates(positions, box_size):
    n_frames, n_atoms, _ = positions.shape
    unwrapped = np.zeros_like(positions)
    unwrapped[0] = positions[0]

    for t in range(1, n_frames):
        displacement = positions[t] - positions[t-1]
        displacement = displacement - np.round(displacement / box_size) * box_size
        unwrapped[t] = unwrapped[t-1] + displacement

    return unwrapped

def unwrapped_distances(x0, x1):
    '''Returns distance**2'''
    delta = x0-x1
    return np.sqrt(np.sum(delta**2, axis=1))

def dist_pbc(x0, x1, dimensions):
    '''Returns distance**2'''
    delta = x0-x1
    delta = np.where(delta > 0.5*dimensions, delta-dimensions, np.where(delta < -0.5*dimensions, delta+dimensions, delta))
    return np.linalg.norm(delta, axis=1)**2

def get_MSD(additive_group):

    additive_group = [additive_residues[i] for i in additive_group]

    all_msds = []
    for START_PT in STARTS:
        msd_array = np.zeros((len(additive_group),WINDOW_SIZE))
        
        L_line_index =  (ATOM_COUNT+9) * (START_PT) + 5
        file_at_START_PT = go_to_line(lammpstrj_file_name,L_line_index)
        L = float(file_at_START_PT.readline().split()[-1])

        line_index =  (ATOM_COUNT+9) * (START_PT) + np.min(additive_group) + 9
        file_at_START_PT = go_to_line(lammpstrj_file_name,line_index)

        for frame in range(WINDOW_SIZE):
            additive_group_atom_count = sum([len(i) for i in additive_group])
            
            frame_coords = np.zeros((additive_group_atom_count,3))
            for atom_index in range(additive_group_atom_count):
                line = file_at_START_PT.readline()
                frame_coords[atom_index] = np.array([float(x) for x in line.split()[-3:]])
            frame_coords = wrap_coords(frame_coords,L)

#            com_coords = np.zeros((len(additive_group),3))
#            index_in_atom_group = 0
#            for group_ix,atom_indices_in_system in enumerate(additive_group):
#                molecule_coords = np.zeros((len(atom_indices_in_system),3))
#
#                for index_in_molecule,index_in_system in enumerate(atom_indices_in_system):
#                    molecule_coords[index_in_molecule] = frame_coords[index_in_atom_group]
#                    index_in_atom_group += 1
#
#                first_atom = molecule_coords[0].copy()
#                for k in range(3):
#                    molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])>(L/2),molecule_coords[:,k]-L,molecule_coords[:,k])
#                    molecule_coords[:,k] = np.where((molecule_coords[:,k]-first_atom[k])<(-L/2),molecule_coords[:,k]+L,molecule_coords[:,k])
#                
#                masses = np.array(additive_masses[group_ix])
#                masses = np.array([masses,]*1).transpose()
#
#                # print(molecule_coords)
#                molecule_coords = molecule_coords * masses 
#                # print(molecule_coords)
#                molecule_com = np.sum(molecule_coords,axis=0)/np.sum(masses)
#                # print(molecule_com)
#                # print(fsdsdf)
#
#                com_coords[group_ix] = molecule_com
            com_coords = copy.deepcopy(frame_coords)

            if frame > 0:
                displacement = com_coords - previous_unwrapped_frame
                displacement = displacement - np.round(displacement / L) * L
                unwrapped_com_coords = previous_unwrapped_frame + displacement
            else:
                unwrapped_com_coords = com_coords

            if frame != WINDOW_SIZE-1:
                for i in range(ATOM_COUNT - 1 - np.max(additive_group) + 5):
                    next(file_at_START_PT)
                L = float(file_at_START_PT.readline().split()[-1])
                for i in range(3):
                    next(file_at_START_PT)
                for i in range(np.min(additive_group)):
                    next(file_at_START_PT)

            if frame == 0:
                initial_coords = copy.deepcopy(com_coords)
                msd_array[:,0] = 0

            else:
                squared_displacement = unwrapped_distances(unwrapped_com_coords,initial_coords)**2
                for iterindex,atom_index in enumerate(additive_group):
                    msd_array[iterindex][frame] = squared_displacement[iterindex]

            previous_unwrapped_frame = copy.deepcopy(unwrapped_com_coords)
                    
        all_msds.append(msd_array)

    stacked_msds = np.vstack(all_msds)

    return stacked_msds


CORE_COUNT = 24
STARTS = [200]
WINDOW_SIZE = 250

additive_groups = split(list(range(WATER_COUNT)),CORE_COUNT)

outputs = Parallel(n_jobs=CORE_COUNT, backend='multiprocessing')(delayed(get_MSD)(additive_groups[i]) for i in range(CORE_COUNT))

all_msds = np.vstack([i for i in outputs])

f_out = open('msd.out','wb')
np.save(f_out,all_msds)
f_out.close()
