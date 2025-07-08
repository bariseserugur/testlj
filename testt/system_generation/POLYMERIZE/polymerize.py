#!/usr/bin/env python
__author__ = "Michael A. Webb"
__version__ = "1.0.1"
__credits__ = ["Michael A. Webb"]
__email__ = "mawebb@princeton.edu"
__license__ = "GPL"

"""
This program (and other distributed files) enable the facile
generation of polymers based on constitutent monomer .pdbs.
The monomer .pdbs must be formatted such that the FIRST atom
Available options/syntax can be queried by launching the program
with the "-h"
"""

#==================================================================
# MODULES
#==================================================================
import sys,argparse,datetime
from math import *
import numpy as np
import time
import os
import random
import copy
from scipy.spatial import distance as sd
from importlib.machinery import SourceFileLoader
relpath = os.path.dirname(__file__)
if (relpath == ""):
  relpath = "."
sys.path.append(relpath)
graphs = SourceFileLoader('graphMod','{}/polymods/graphMod.py'.format(relpath)).load_module()
caps    = SourceFileLoader('capTypes','{}/polymods/capTypes.py'.format(relpath)).load_module()
massMod = SourceFileLoader('massesDb','{}/polymods/massesDb.py'.format(relpath)).load_module()

#==================================================================
#  GLOBAL VARIABLES
#==================================================================
npush_max = 500

#==================================================================
#  AUX: create_parser
#==================================================================
def create_parser():

  parser = argparse.ArgumentParser(description='Generates a polymer .pdb file from monomer .pdb files. Arguments with multiple values need to be enclosed by quotation marks.')

  # DO REQUIRED POSITIONAL ARGUMENTS
  parser.add_argument('list_file',help = 'File that lists paths for monomer .pdb files or all units to appear in the polymer. One .pdb file is included per row. The path is the first column and its frequency of occurrence appears in the second column.  ')


  parser.add_argument('-type'   ,dest='poly_type',default="homo",help = 'Polymerization type keyword. Note that block polymers are a special case of alternating copolymers that can be specified by increasing the integers in -ratio. (options = alt,co,homo) (default = homo)')

  parser.add_argument('-N'   ,dest='N',default="1",help = 'Number of repeat units in polymer. For homopolymers, this is the number of monomers. For alternating copolymers, it is the number of repeating segments with all components. For random copolymers, it is the number of monomer attempts. (default = 1)')

  parser.add_argument('-precision'   ,dest='prec',default=0.001,help = 'Precision for determined monomer ratios if fractions are specified. (default = 0.1)')

  parser.add_argument('-max_rot'   ,dest='max_ang',default=0.0,help = 'Maximum rotation angle for polymerization (in units of pi). (default: 0)')

  parser.add_argument('-name'   ,dest='root_name',default="poly",help = 'Root name for generated output files. (default: poly)')

  parser.add_argument('-Np'   ,dest='Np',default=1,help = 'Number of polymers to be generated. (default: 1)')

  parser.add_argument('-seq'   ,dest='seq',default="auto",help = 'Monomer sequence for each polymer. The sequences are specified in a file (one polymer per line) as a sequence of integers matched to the order of .pdbs listed in the list_file. This option only needs to be included if you want a specific sequence that cannot be achieved via the other options provided in -type. (default: "auto")')

  parser.add_argument('-tail'   ,dest='tail',default=None,help = 'Indication of what functional group can be used to cap the polymer tail. Built-in options include "hydrogen", "methyl", "ethyl", "hydroxy", "methoxy", and "ethoxy". If a string other than a built-in option is specified then a particle bead with that name will be placed. (default: None)')

  parser.add_argument('-head'   ,dest='head',default=None,help = 'Indication of what functional group can be used to cap the polymer head. See -tail for options (default: None)')

  parser.add_argument('-rmin'   ,dest='rmin',default=1.5,help = 'Minimum allowable contact distance between atoms in generated polymer configuration. This is used in quick pushoff algorithm to avoid overlaps after all monomers are placed. (default: 1.8)')
  return parser

#==================================================================
#  AUX: write_quick_xyz
#==================================================================
def write_quick_xyz(molnum,atms):
  fxyz = open("test.{}.xyz".format(molnum),"w")
  Natm = len(atms)
  fxyz.write("{}\n\n".format(Natm))
  for num,atm in atms.items():
    x,y,z = list(atm['crds'][:])
    fxyz.write("{} {:>10.3f} {:>10.3f} {:>10.3f}\n".format(atm['name'],x,y,z))
  fxyz.close()
  return

#==================================================================
#  AUX: seq_from_file
#==================================================================
def seq_from_file(fid):
  seq = []
  lines = [line.strip().split() for line in fid]
  lines = [[int(val) for val in line] for line in lines]
  for line in lines:
    seq.append([])
    for val in line:
      seq[-1].append(val)
  return seq

#==================================================================
#  AUX: convert_args
#==================================================================
def convert_args(args):
  # PARSE REQUIRED ARGUMENT FILE
  flist     = open(args.list_file,"r")
  lines     = [line.strip().split() for line in flist]
  pdb_list  = [line[0] for line in lines]           # path to .pdb files with monomer structures
  rat       = [float(line[1]) for line in lines]    # relative ratio of polymer components

  # OPTIONAL ARGUMENTS
  prec      = float(args.prec)                      # preceision for occurrency frequency
  poly_type = args.poly_type                        # polymerization type = (homo, alt, random, block)
  root_name = args.root_name                        # root name for file gneration
  N   = int(args.N)                                 # degree of polymerization
  Np  = int(args.Np)                                # number of polymers to construct
  if (args.seq == "auto" ):                         # sequence information
      seq  = args.seq                               # by default, the sequence is determined at run time
  else:                                             # however, specific sequences can be generated
      fid  = open(args.seq,"r")                     # within a file (each polymer on its own line, with
      seq  = seq_from_file(fid)                     # monomer units indexed by order in the pdb_list file)

  max_ang = np.pi*float(args.max_ang)               # rotation angle to apply to generated monomer sequence
  rmin    = float(args.rmin)

  # GET AND CHECK CAP GROUPS
  print("# Checking end-cap groups...")
  tailGroup = args.tail                             # type of tail group cap
  headGroup = args.head                             # type of head group cap
  for i,cap in enumerate([tailGroup,headGroup]):
    if cap is None:
      print("# No cap group assigned for {}".format(caps.caplbl[i]))
    else:
      if cap in caps.caps:
        print("# {} will use built-in geometry for {}".format(caps.caplbl[i],cap))
      else:
        print("# {} will use bead with label of {}".format(caps.caplbl[i],cap))
        caps.add_generic_cap(cap)

  # FIND EQUIVALENT INTEGERS FOR RATIO IF NECESSARY
  norm_rat = float(sum(rat))
  rat_frac = [v/norm_rat for v in rat ]
  rat_test = rat[:]
  if any([v%1 for v  in rat]):
    print("# Finding equivalent set of ratio integers for ", rat, "...")
    print("# New set should yield mole fraction of ", rat_frac, "\n# to a precision of ", prec, "...")
    equiv_int_found = False
    pwr = 0
    while (not equiv_int_found):
      pwr+=1
      rat_test = [int(10**pwr*v) for v in rat[:]]
      norm     = sum(rat_test)
      rat_comp = [float(v)/norm for v in rat_test]
      rat_diff = [abs(v2-v1) for v1,v2 in zip(rat_frac,rat_comp)]
      equiv_int_found = all([v <= prec for v in rat_diff])
    rat = rat_test
    rat_frac = rat_comp[:]
    print("# Equivalent set of ratio integers determined as ", rat)
    print("# This yields mole fraction incorporation of ", rat_frac)
  rat = [int(v) for v in rat]

  # INITIALIZE DICTIONARY FOR CONTAINING MOLECULE INFORMATION
  MOLS = {}
  for i,the_pdb in enumerate(pdb_list):
    MOLS[i] = {}
    MOLS[i]['pdb'] = the_pdb

  files = {'root': root_name,\
          'polypdb': [root_name + "_pdb_files/"+ root_name,'.pdb'] ,\
          'polyxyz': [root_name + "_xyz_files/"+ root_name , '.xyz']}
  sys = {'pdb': pdb_list,'ratio': rat,'freq':rat_frac, 'type': poly_type,'degree': N,'max_ang':max_ang,\
          'npoly':Np,'seq': seq, 'tail': tailGroup,'head' : headGroup,'rmin' : rmin}

  # PRINT OUT SOME CONFIRMATION
  print("/n********************************************************")
  print("~~~~~~~~~~~~~~~~~~~~OPTION SUMMARY~~~~~~~~~~~~~~~~~~~~~~")
  print("********************************************************")
  print("# Polymerization type set to ", poly_type)
  print("# Degree of polymerization will be ", N)
  print("# Number of polymers to be generated is ", Np)
  print("# Polymer components defined in ", pdb_list)
  print("# Target polymerization set in ratio of ",rat)
  print("# Target mole fraction icorporation is thus ", rat_frac)
  print("********************************************************")

  return sys,MOLS,files

#==================================================================
#  AUX: make_missing_dir
#==================================================================
def make_dir(dirname):
  try:
    os.mkdir(dirname)
  except:
      print("\n# WARNING: Directory ({}) already exists! Overwrite of data is possible.".format(dirname))
  return None

#==================================================================
#  AUX: find_bonds
#=================================================================
def find_bonds(atoms):
  bonds = set()
  N     = len(atoms)
  adjmat=np.zeros([N,N])
  for atmkey in atoms:
    for j in atoms[atmkey]['adj']:
      if (atmkey < j):
        bonds.add((atmkey,j))
        adjmat[atmkey-1,j-1] = 1 # add also to adjacency matrix
        adjmat[j-1,atmkey-1] = 1 # adjacency matrix is symmetric

  return bonds,adjmat

#==================================================================
#  AUX: find_angles
#=================================================================
def find_angles(atoms,bonds):
  angles = set()
  # loop over all the bonds
  for [i,j] in bonds:
    # check atoms bonded to i that are not j
    for k in [l for l in atoms[i]['adj'] if l != j]:
      if (j<k):
        angles.add((j,i,k))
      else:
        angles.add((k,i,j))
    # check atoms bond to j that are not i
    for k in [l for l in atoms[j]['adj'] if l != i]:
      if (i<k):
        angles.add((i,j,k))
      else:
        angles.add((k,j,i))

  return angles

#==================================================================
#  AUX: find_dihedrals
#=================================================================
def find_dihedrals(atoms,angles):
  dihedrals = set()
  # loop over all angles
  for [i,j,k] in angles:
    # check for atoms that are bonded to i that are not j
    # then add listing lowest index first
    for l in [m for m in atoms[i]['adj'] if m != j]:
      if ( l < k ):
        dihedrals.add((l,i,j,k))
      else:
        dihedrals.add((k,j,i,l))
    # check for atoms that are bonded to k that are not j
    # then add listing lowest index first
    for l in [m for m in atoms[k]['adj'] if m != j]:
      if ( l < i ):
        dihedrals.add((l,k,j,i))
      else:
        dihedrals.add((i,j,k,l))
  return dihedrals

#==================================================================
# AUX: quick_file_read
#==================================================================
# reads file, eliminating empty lines and new line characters
def quick_file_read(fid):
  lines = [line.strip() for line in fid]
  lines = [line.split() for line in lines if line]
  return lines

#==================================================================
# AUX: pdb_loose_process
#==================================================================
# reads file, eliminating empty lines and new line characters
def pdb_loose_process(lines):
  atoms = {}
  for line in lines:
    record = line[0]
    if (record == "HETATM" or record =="ATOM"):
      ind=int(line[1])
      typ=line[-1]
      crd=np.array([float(i) for i in line[5:8]])
      atoms[ind] = {}
      atoms[ind]['name']     = typ
      atoms[ind]['crds']     = crd
    if (record == "CONECT"):
      ind=int(line[1])
      adjlist = [int(i) for i in line[2:]]
      atoms[ind]['adj'] = adjlist

  return atoms

#==================================================================
# AUX: remove_overlaps
#==================================================================
def remove_overlaps(poly,id_map,rmin):
  # set cut off of interaction for soft-core potential
  rcut = rmin/0.8

  # get initial coordinates
  N = len(id_map)
  crds   = np.zeros([N,3])
  adjMat = np.zeros([N,N])
  for imon,mon in enumerate(poly):
    for atm,info in mon.items():
      ind_i = id_map[(imon,atm)] -1
      crds[ind_i,:] = info['crds'][:]
      for cnct in info['adj']:
        ind_j = id_map[cnct]-1
        adjMat[ind_i,ind_j] = 1
        adjMat[ind_j,ind_i] = 1

  # now apply pushoff steps
  for step in range(npush_max):
    # compute distances
    r  = crds[:,np.newaxis,:] - crds[np.newaxis,:,:]  # generates NxNx3 matrix of distance vectors
    d  = np.sqrt(np.sum(r**2,axis=-1))

    # if first step do all initializations
    if step ==0:
      r0 = d * adjMat
      fmag = np.zeros([N,N])
      f    = np.zeros([N,N,3])
      bonded = adjMat == 1
      angled = np.matmul(adjMat,adjMat) > 0

    # Check whether there is significant overlap amongst any particles
    bigOverlaps = d < rmin
    bigOverlaps[bonded] = False
    bigOverlaps[np.diag_indices(N)] = False
    if not bigOverlaps.any():
      print("# No substantial overlaps found after {} steps...\n\n".format(step))
      break
    else:
      if step%50 == 0:
        print("# Significant overlaps detected. Attempting pushoff step {}".format(step+1))

    # initialize the forces
    f[:,:,:]  = 0.0 # set all elements to zero
    fmag[:,:] = 0.0

    # compute all pairwise forces using a soft-core potential Uij = A[a+ cos(pi*rij/rc)]
    # find elements where the distance matrix is less than the cutoff
    overlapping = d < rcut      # boolean array of indices where the distance is less than the cutoff
    overlapping[bonded] = False # set anything bonded and less than cutoff to be false
    overlapping[np.diag_indices(N)] = False # also set the diagonal/angles to be false
    fmag[overlapping] = 0.50*np.sin(np.pi*d[overlapping]/rcut)/rcut/d[overlapping] # force magnitudea

    # also find the bonded force (already divided by distance)
    fmag[bonded] = r0[bonded]/d[bonded] -1.0

    # compute vector force
    f =  r*fmag [:,:,None]  # gets vector force

    # get sum of force on each particle
    fi = np.sum(f,axis=1)

    # evolve positions
    crds[:,:] += 0.5*fi[:,:]

  return crds

#==================================================================
#  AUX: get_mol_spec
#=================================================================
def get_mol_spec(MOLS):

  # Read each pdb file and store information in MOLS dictionary
  for molkey in MOLS:
    with open(MOLS[molkey]['pdb'],"r") as fid:
      print("\n\n********************************************************")
      print("# PROCESSING MOLECULE ", repr(molkey), "...")
      lines     = quick_file_read(fid)
      atoms     = pdb_loose_process(lines)     #
      bonds,Am  = [],[]#find_bonds(atoms)
      angles    = [] #find_angles(atoms,bonds)
      dihedrals = [] #find_dihedrals(atoms,angles)
      print("# Topology for molecule determined as follows:")
      print("# --Number of atoms    : ",repr(len(atoms)))
      print("# --Number of bonds    : ",repr(len(bonds)))
      print("# --Undirected adjacency matrix: ")
      print(Am)
      print("# --Number of angles   : ",repr(len(angles)))
      print("# --Number of dihedrals: ",repr(len(dihedrals)))
      print("********************************************************")

    # ADD RESULTS TO MOLECULE DICTIONARY
    MOLS[molkey]['atoms']     = atoms.copy()
    MOLS[molkey]['bonds']     = list(bonds)
    MOLS[molkey]['angles']    = list(angles)
    MOLS[molkey]['dihedrals'] = list(dihedrals)
    MOLS[molkey]['adj']       = Am

  return

#==================================================================
#  AUX: get_rand_type
#=================================================================
def get_rand_type(MOLS,freq):
  nmol   = len(MOLS)
  rnum   = random.random()
  thresh = [0]
  thresh.extend(np.cumsum(freq[:-1]))
  thresh = [rnum > v for v in thresh]
  return nmol - thresh[::-1].index(True) - 1

#==================================================================
# AUX: align_to_x
#==================================================================
def align_to_x(mol,Am):
  n      = len(mol)
  # first we will use dijskstra's algorithm to find the shortest path
  # between the "head" node which is first and the "tail" node
  g   = graphs.Graph(Am)
  dht = int(g.dijkstra(0,n-1))

  # if dht is odd, then that means that a molecule in all trans configuration
  # would terminate "out-of-phase", if it is even, then it terminates "in-phase"
  # for in-phase: the tail is used for aligning
  # for out-of-phase: the atom before the tail is used for aligning
  ihead = 1
  if dht % 2 ==0: # even
    itail = n
  else:
    itail = mol[n]['adj'][0]

  rtrans = mol[ihead]['crds'].copy()
  dr     = mol[itail]['crds'][:] - rtrans[:]

  # COMPUTE ROTATION ANGLES
  theta = np.arctan2(-dr[2] , dr[1])  # theta = arctan(-z/y)
  cost  = np.cos(theta)
  sint  = np.sin(theta)
  alpha = np.arctan2(-(cost*dr[1]-sint*dr[2]),dr[0])
  cosa  = np.cos(alpha)
  sina  = np.sin(alpha)

  # CONSTRUCT ROTATION MATRIX
  R = np.array(\
       [[cosa,-sina*cost,sina*sint],\
       [sina,cosa*cost,-cosa*sint],\
       [0,sint,cost]])

  # DO THE COORDINATE TRANSFORMATION
  for atm,info in mol.items():
    info['crds']    -= rtrans[:]
    mol[atm]['crds'] = np.dot(R,info['crds'].transpose()).transpose()

#==================================================================
# AUX: rotate_molecule
#==================================================================
# theta is a list of three floats
# rot_order is a list of integers
def rotate_molecule(mol,theta = None, max_ang = 1.0,rot_order =  None):

  # if no rotation angle is specified, then apply random rotation
  if theta is None:
    # Select random rotation rangles about x,y, and z axes
    theta = [(random.random()-0.5)*max_ang for i in range(3)]

  # Construct the rotation matrices
  R  = [[],[],[]]
  R[0] = np.array([\
         [1,0,0],\
         [0, cos(theta[0]),-sin(theta[0])],\
         [0,sin(theta[0]),cos(theta[0]) ]] )
  R[1] = np.array([\
         [cos(theta[1]),0,sin(theta[1])],\
         [0,1,0],\
         [-sin(theta[1]),0,cos(theta[1])]])
  R[2] = np.array([\
         [cos(theta[2]),-sin(theta[2]),0],\
         [sin(theta[2]),cos(theta[2]),0],\
         [0,0,1]])

  # Create random rotation order
  if rot_order is None:
    rot_order = [i for i in range(3)]
    random.shuffle(rot_order)

  # Perform the random rotation
  for atm,info in mol.items():
    for j in rot_order:
      mol[atm]['crds'] = np.dot(R[j],info['crds'].transpose())
      mol[atm]['crds'] = mol[atm]['crds'].transpose()

  return mol

#==================================================================
# AUX: write_poly_pdb
#==================================================================
def write_poly_pdb(ip,files,id_map,poly):
  fname = files['polypdb'][0]+"-"+(repr(ip))+files['polypdb'][1]
  print("# Writing out polymer structure/connectivity to ",fname," ...")
  # WRITE THE HEADER
  fid = open(fname,"w")
  fid.write("COMPND    UNNAMED\n")
  fid.write("AUTHOR    GENEATED BY polymerize.py\n")
  # WRITE HETATM ENTRIES
  mass = 0.0
  for imon,mon in enumerate(poly):
    for atm,info in mon.items():
        fid.write("{:<6s}{:>5d}  {:<3s}{:1s}{:>3s}  {:>4d}{:<1s}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      {:>4s} {:<s}\n".\
                format("ATOM",id_map[(imon,atm)],info['name'][:3]," ","UNL",imon," ",\
                info['crds'][0],info['crds'][1],info['crds'][2],1.00,0.00,"",info['name']))
        #fid.write("{:<6s}{:>5d} {:>4s}{:1s}{:>3s}  {:>4d}{:<1s}    {:>7s} {:>7s} {:>7s}{:>6.2f}{:>6.2f}      {:>4s} {:<s}\n".\
        #        format("ATOM",id_map[(imon,atm)],info['name'][:3]," ","UNL",imon," ",\
        #        repr(round(info['crds'][0],3))[:7],repr(round(info['crds'][1],3))[:7],repr(round(info['crds'][2],3))[:7],1.00,0.00,"",info['name']))
        mass += massMod.guess_mass(info['name'][:])
  # WRITE CONECT ENTRIES
  for imon,mon in enumerate(poly):
    for atm,info in mon.items():
      fid.write("{:<6s}{:>5d}".format("CONECT",id_map[(imon,atm)]))
      for cnct in sorted(info['adj']):
        fid.write("{:>5d}".format(id_map[cnct]))
      fid.write("\n")

  print("The approximate mass of this polymer is {}".format(mass))
  return None

#==================================================================
# AUX: polymerize_mols
#==================================================================
def polymerize_mols(ip,sys,MOLS,files):

  # GENERATE SEQUENCE
  print("# Generating sequence of monomers...")
  if sys['seq'] == "auto":
    seq = []
    for i in range(sys['degree']):
      # Homopolymers
      if (sys['type'] == 'homo'):
        seq.append(0)
      # Alternating copolymers
      elif (sys['type'] == 'alt'):
        for j,n in enumerate(sys['ratio']):
          for k in range(n):
            seq.append(j)
      # Random copolymers
      elif (sys['type'] == 'co'):
        seq.append(get_rand_type(MOLS,sys['freq']))
      else:
        pass
    print("# Generated sequence is as follows:", seq)
    if sys['type'] == 'co':
      print("# Ratios: ")
      nmols = len(MOLS)
      for i in range(nmols):
          rat = 1.0*seq.count(i)/sys['degree']
          print("Type ", i, ": ", rat)

  else:
    seq = sys['seq'][ip]

  # BEGIN POLYMERIZATION
  poly  = [] # list that keeps track of all placed monomers
  heads = [] # list that tracks the index of the "head" of each placed monomer
  tails = [] # list that tracks the index of the "tail" of each placed mononer

  # PLACE HEAD CAP GROUP (IF ANY)
  if sys['head'] is not None:
    headCap = copy.deepcopy(caps.caps[sys['head']])
    n = len(headCap)
    headCap[1]['adj'].append('head')
    heads.append([1])
    tails.append([1])
    # invert coordinates on the head group
    for i,atm in headCap.items():
      atm['crds'][:] *= -1
      atm['crds'][0] -= 0.9
    poly.append(headCap)

  # PLACE FIRST MONOMER
  print("# Placing starting monomer of type {}...".format(seq[0]))
  mon   = copy.deepcopy(MOLS[seq[0]]['atoms'])  # copy of monomer information
  g     = graphs.Graph(MOLS[seq[0]]['adj']) # graph object for monomer
  n     = len(mon)                      # number of atoms in the monomer
  Lbb   = int(g.dijkstra(0,n-1)) -1     # path length along backbone (accounting for eventual elimination of tail)
  tail  = mon.pop(n, None)              # remove endpt from monomer/store adjacency info
  idlink= tail['adj'][0]                # this is the id of the monomer connected to the tail
  mon[idlink]['adj'].pop(mon[idlink]['adj'].index(n)) # removes the tail atom from its adjacency list
  rn    = tail['crds'].copy()           # position of the tail point of monomer
  if sys['head'] is not None:           # if there is a cap group, then it needs to be included in adjacency list
    mon[1]['adj'].append('tail')
  poly.append(copy.deepcopy(mon))
  for j,jmon in enumerate(poly):
    for k,katm in jmon.items():
      poly[j][k]['crds'] -= rn[:]
  heads.append([1])
  tails.append([idlink])

  # LOOP OVER REMAINING MONOMERS AND ADJOIN THEM AT LINK POINTS
  flippedLastTime = False
  for i,imon in enumerate(seq[1:]):
    heads.append([1])
    tails.append([])
    print("# Placing monomer {} of type {}...".format(i+2,imon))

    # create copy of monomer with atom key shift
    # shift coordinates to align with first link
    mon     = copy.deepcopy(MOLS[imon]['atoms'])  # copy of monomer information

    # if the previously placed monomer has an odd-number of bacbkone atoms (even path length)
    # --> we will want to rotate the monomer unit by 180 degrees about the x-axis
    #     such that this monomer placement will be in trans configuration by default
    # HOWEVER, if the last placement was flipped, then that means the monomer will already
    # be in the correct orientation
    if Lbb%2 == 0:
      if not flippedLastTime:
        mon = rotate_molecule(mon,theta=[np.pi,0.,0.])
        flippedLastTime = True
      else:
        # pass on flipping this one
        flippedLastTime = False

    # set up next calculation
    g     = graphs.Graph(MOLS[imon]['adj']) # graph object for monomer
    n     = len(mon)                      # number of atoms in the monomer
    Lbb   = int(g.dijkstra(0,n-1)) -1     # path length along backbone (accounting for eventual elimination of tail)

    # Here we will rotate the coordinates based on the
    cnt     = len(poly)
    poly[cnt-1][idlink]['adj'].append('head') # add connection of previous monomer to head of this monomer
    mon[1]['adj'].append('tail')          # add connection of head to tail of previous

    # Rotate monomer coordinates to new direction (confined by max angle)
    new_mon = rotate_molecule(mon,max_ang=sys['max_ang'])

    # shift coordinates of all placed atoms such that
    # the next link atom is at the origin
    tail  = new_mon.pop(n, None)            # remove tail point from monomer/store adjacency info
    idlink= tail['adj'][0]
    new_mon[idlink]['adj'].pop(new_mon[idlink]['adj'].index(n)) # remove connection for tail point
    tails[-1].append(idlink)                # add atom to head group
    linkt = copy.deepcopy(new_mon[idlink])  # get the tail link atom information
    rn    = tail['crds'].copy()            # position of the endpoint of monomer
    poly.append(copy.deepcopy(new_mon))
    for j,jmon in enumerate(poly):
      for k,katm in jmon.items():
        poly[j][k]['crds'] -= rn[:]

  # HANDLE THE TAIL CAP GROUP
  if sys['tail'] is not None:
    tailCap = copy.deepcopy(caps.caps[sys['tail']])   # this gets atoms info for the cap group
    n = len(tailCap)
    tailCap[1]['adj'].append('tail')                  # add connection for cap to tail of last monomer
    poly[-1][idlink]['adj'].append('head')            # and the last monomer to head of the cap
    rn = tailCap[n]['crds'].copy()
    poly.append(tailCap)
    for j,jmon in enumerate(poly):
      for k,katm in jmon.items():
        poly[j][k]['crds'] -= rn[:]
    heads.append([1])
    tails.append([n])

  # CREATE A MAPPING DICTIONARY AND REWRITE ADJACENCY LISTS
  cnt = 0
  id_map = {}
  for imon,mon in enumerate(poly):
    for atm,info in mon.items():
      cnt+=1
      id_map[(imon,atm)]=cnt
      adjMap = []
      for icnct,cnct in enumerate(info['adj']):
        if cnct == 'head':
          for ihead in heads[imon+1]:
            adjMap.append((imon+1,ihead))
        elif cnct == 'tail':
          for itail in tails[imon-1]:
            adjMap.append((imon-1,itail))
        else:
          adjMap.append((imon,cnct))
      info['adj'] = adjMap[:]

  # PRINT OUT COORDINATES TO XYZ FILE#
  fname = files['polyxyz'][0]+"-"+(repr(ip))+files['polyxyz'][1]
  print("# Writing out polymer structure to ",fname," ...")
  fid = open(fname,"w")
  ntot = sum([len(i) for i in poly])
  fid.write("{}\n\n".format(ntot))
  cnt  = 0
  for imon,mon in enumerate(poly):
    for atm,info in mon.items():
      cnt+=1
      icrds = info['crds'][:]
      attype   = info['name']
      fid.write("{:<8} {:>10.5f} {:>10.5f} {:>10.5f}\n".format(attype,icrds[0],icrds[1],icrds[2]))
  #fid.close()

  # RUN MINIMIZATIONTO PERTURB GEOMETRY TO REMOVE BAD OVERLAPS
  crds = remove_overlaps(poly,id_map,sys['rmin'])

  # assign new coordinates
  cnt = 0
  for imon,mon in enumerate(poly):
      for atm,info in mon.items():
          info['crds'][:] = crds[cnt,:]
          cnt+=1

  ntot = sum([len(i) for i in poly])
  fid.write("{}\n\n".format(ntot))
  cnt  = 0
  for imon,mon in enumerate(poly):
    for atm,info in mon.items():
      cnt+=1
      icrds = info['crds'][:]
      attype   = info['name']
      fid.write("{:<8} {:>10.5f} {:>10.5f} {:>10.5f}\n".format(attype,icrds[0],icrds[1],icrds[2]))
  fid.close()
  return id_map,poly


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  MAIN: _main
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def main(argv):
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # PARSE ARGUMENTS
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  parser = create_parser()
  args                         = parser.parse_args()
  (sys,MOLS,files) =\
                                 convert_args(args)

  t0 = time.time()
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # INTERPRET MOLECULE STRUCTURE
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # MOLS will contain references for each molecule .pdb for
  # the geometry and connectivity, separated in terms of bonds,
  # angles, and dihedrals.
  # MOLS keys for each molecules: 'atoms','bonds','angles','dihedrals','pdb'
  get_mol_spec(MOLS)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # ALIGN MONOMER UNITS TO X-AXIS
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  for molkey,the_mol in MOLS.items():
    align_to_x(the_mol['atoms'],the_mol['adj'])
   #write_quick_xyz(molkey,the_mol['atoms'])

  make_dir(files['root']+"_xyz_files")
  make_dir(files['root']+"_pdb_files")
  for ip in range(sys['npoly']):
    print("\n********************************************************")
    print("~~~~~~~~~~~~~~POLYMERIZATION OF POLYMER {}~~~~~~~~~~~~~~".format(ip+1))
    print("********************************************************")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # POLYMERIZE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    id_map,poly = polymerize_mols(ip,sys,MOLS,files)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # WRITE NEW PDB FILE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    write_poly_pdb(ip,files,id_map,poly)
  print("# Total elapsed time: {} seconds".format(time.time()-t0))

#==================================================================
#  RUN PROGRAM
#==================================================================
if __name__ == "__main__":
  main(sys.argv[1:])
