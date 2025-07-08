import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import numpy as np
from scipy.stats import sem
import os
def wrap_coords(system_coords,L):
    return np.where(system_coords > L, system_coords-L, np.where(system_coords < 0, system_coords+L, system_coords))

for POLY_TYPE in ['block','random','spaced']:
    sim_values = []
    for simno in [3]:
        os.chdir('{}/{}'.format(POLY_TYPE,simno))
        traj_file = 'cubic_pellet.lammpstrj'
        f = open(traj_file,'r')
        for i in range(4):
            line = f.readline()
        N = int(line)
        f.seek(0)
        
        bead_type_list = []
        pdb_file = open('init.pdb','r').readlines()
        for line in pdb_file:
            if 'ATOM' in line or 'HETATM' in line:
                res = line[17:20]
                bead_type_list.append(res)
        bead_type_list = np.array(bead_type_list)
       
        values = []
        frame_limit = 199
        for frame in range(200):
            if frame < frame_limit:
                for i in range(9+N):
                    next(f)
                continue

            frame_coords = []
            for i in range(6):
                line = f.readline()
            L = float(line.split()[1])
            for i in range(3):
                line = f.readline()
            for i in range(N):
                line = f.readline()
                coord = np.array([float(x) for x in line.split()[-3:]])
                frame_coords.append(coord)
            frame_coords = np.vstack(frame_coords)
            frame_coords = wrap_coords(frame_coords,L)
        
            A_coords = frame_coords[bead_type_list=='AAA']
            B_coords = frame_coords[bead_type_list=='BBB']
            W_coords = frame_coords[bead_type_list=='WWW']
        
            cutoff = 6.0  # in angstroms
        
            dists = contacts.distance_array(B_coords, B_coords, [L,L,L,90,90,90])
            dists = dists[dists > 5.1]
            num_contacts = ((dists < cutoff) & (dists > 2)).sum()

            a,b = np.histogram(dists[dists < 15],bins=500)
            print('a = {}'.format(list(a[::2])))
            print('b = {}'.format(list(b[::2])))
            print('plt.plot(b[:-1],a,label="{}")'.format(POLY_TYPE))
            values.append(num_contacts)

        sim_values.append(np.mean(values))

        os.chdir('../../')
    #print('{}: {}'.format(POLY_TYPE,np.mean(sim_values)))
    #print('{}: {}'.format(POLY_TYPE,sem(sim_values)))
