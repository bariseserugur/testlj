import numpy as np
from datetime import datetime
import sys
import random

def random_unit_vector_within_angle(prev_vec, max_angle_deg):
    """
    Generate a random unit vector within max_angle_deg of the given vector.
    """
    max_angle_rad = np.radians(max_angle_deg)
    
    # Create a random direction in the local frame
    while True:
        # Generate a random vector
        random_vec = np.random.normal(size=3)
        random_vec /= np.linalg.norm(random_vec)

        # Compute angle between previous vector and candidate
        cos_angle = np.dot(prev_vec, random_vec)
        if cos_angle >= np.cos(max_angle_rad):
            return random_vec

def generate_array(N, X):
    min_required = 4 * X - 1
    if N < min_required:
        raise ValueError("N too small for X separated groups")

    arr = [0] * N
    spots = list(range(N - 2))
    random.shuffle(spots)
    used = set()
    count = 0
    for i in spots:
        if any(j in used for j in range(i - 1, i + 4)):
            continue
        for j in range(i, i + 3): arr[j] = 1; used.add(j)
        count += 1
        if count == X: break
    return arr

def random_ones_array(N, X):
    if X > N:
        raise ValueError("X cannot be greater than N")
    arr = [0] * N
    for i in random.sample(range(N), X):
        arr[i] = 1
    return arr

def evenly_distributed_ones(N, X):
    if X > N:
        raise ValueError("X cannot be greater than N")
    arr = [0] * N
    step = N / X
    for i in range(X):
        index = round(i * step)
        if index >= N:
            index = N - 1
        arr[index] = 1
    return arr

block_dict = {1:'BBB',0:'AAA'}
block_dict_atom = {1:'Bb',0:'Aa'}
def generate_random_polymer_pdb(type='block', filename="polymer_random.pdb", n_monomers=100, bond_length=5.0, seed=None, max_bend_deg=60, DH=76):
    """
    Generates a random-coil polymer and writes it to a PDB file with header and CONECT entries.

    Parameters:
        filename (str): Output PDB file name
        n_monomers (int): Number of monomers (particles)
        bond_length (float): Distance between each monomer in angstroms
        seed (int): Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    DA = 100 - DH

    BLOCK_SIZE = 6
    assert int(DA/100 * n_monomers) % BLOCK_SIZE == 0

    if type == 'block':
        n_blocks = int(DA/100 * n_monomers / BLOCK_SIZE)
        block_array = generate_array(n_monomers,n_blocks)
    elif type == 'random':
        block_array = random_ones_array(n_monomers, int(DA/100 * n_monomers))
    elif type == 'spaced':
        block_array = evenly_distributed_ones(n_monomers, int(DA/100 * n_monomers))


    # Generate positions using a 3D random walk with fixed step length
    positions = [np.array([0.0, 0.0, 0.0])]
    direction = np.array([1.0, 0.0, 0.0])  # initial direction
    for _ in range(1, n_monomers):
        direction = random_unit_vector_within_angle(direction, max_bend_deg)

        # vec = np.random.normal(size=3)
        # vec /= np.linalg.norm(vec)
        new_pos = positions[-1] + bond_length * direction
        positions.append(new_pos)

    with open(filename, "w") as f:
        # HEADER
        today = datetime.now().strftime("%d-%b-%y").upper()
        f.write(f"HEADER    RANDOM POLYMER CHAIN         {today}\n")
        f.write("TITLE     Freely Jointed Chain Polymer Model\n")
        f.write("REMARK    Each monomer is 5.0 Å apart and represented as one carbon atom.\n")

        # ATOM lines
        for i, pos in enumerate(positions):
            atom_serial = i + 1
            residue_number = 1
            x, y, z = pos
            f.write(
                "ATOM  {:5d} {}{}{}  {:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {}\n".format(
                    atom_serial, str(atom_serial).ljust(3), block_dict_atom[block_array[i]], block_dict[block_array[i]], residue_number, x, y, z, block_dict_atom[block_array[i]]
                )
            )

        # CONECT lines (only consecutive bonds)
        for i in range(n_monomers - 1):
            f.write("CONECT{:5d}{:5d}\n".format(i + 1, i + 2))

        f.write("END\n")

poly_type = str(sys.argv[1])
Np = int(sys.argv[2])
angle = float(sys.argv[3])
DH = int(sys.argv[4])

generate_random_polymer_pdb(n_monomers=Np, bond_length=5.0, seed=None, max_bend_deg=angle, DH=DH, type=poly_type)
