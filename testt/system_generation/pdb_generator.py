#!/usr/bin/env python
__author__ = "Michael A. Webb"
__version__ = "1.0.1"
__credits__ = ["Michael A. Webb"]
__email__ = "mawebb@princeton.edu"
__license__ = "GPL"

"""
This program (and other distributed files) drive the preparation
of simulations in LAMMPS through generation of necessary "data"
and "settings" files. Similar functionality might be achieved
through a program like moltemplate. The primary requirements for the
program are an annotated .pdb file speciying atom types and an
avaliable force field database using the same atom types
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
from importlib.machinery import SourceFileLoader
import random
import re
#==================================================================
# GLOBAL VARIABLES
#==================================================================
version_num = 1
version_date= "May 2020"
the_date = datetime.date.today()
dirs = ['x','y','z']
natt_max  = 100
npush_max = 100
rpush     = 2.5
rint      = 0.9
Na = 6.0221409e23 # Avogadro's numbwr

mass_dictionary = {'H': 1.00797, 'He': 4.00260, 'Li': 6.941, 'Be': 9.01218, 'B': 10.81, 'C': 12.011, 'N': 14.0067, 'O': 15.9994, 'F': 18.998403, 'Ne': 20.179, 'Na': 22.98977, 'Mg': 24.305, 'Al': 26.98154, 'Si': 28.0855, 'P': 30.97376, 'S': 32.06, 'Cl': 35.453, 'K': 39.0983, 'Ar': 39.948, 'Ca': 40.08, 'Sc': 44.9559, 'Ti': 47.90, 'V': 50.9415, 'Cr': 51.996, 'Mn': 54.9380, 'Fe': 55.847, 'Ni': 58.70, 'Co': 58.9332, 'Cu': 63.546, 'Zn': 65.38, 'Ga': 69.72, 'Ge': 72.59, 'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.80, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.9059, 'Zr': 91.22, 'Nb': 92.9064, 'Mo': 95.94, 'Tc': 98, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.4, 'Ag': 107.868, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.69, 'Sb': 121.75, 'I': 126.9045, 'Te': 127.60, 'Xe': 131.30, 'Cs': 132.9054, 'Ba': 137.33, 'La': 138.9055, 'Ce': 140.12, 'Pr': 140.9077, 'Nd': 144.24, 'Pm': 145, 'Sm': 150.4, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.9254, 'Dy': 162.50, 'Ho': 164.9304, 'Er': 167.26, 'Tm': 168.9342, 'Yb': 173.04, 'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.85, 'Re': 186.207, 'Os': 190.2, 'Ir': 192.22, 'Pt': 195.09, 'Au': 196.9665, 'Hg': 200.59, 'Tl': 204.37, 'Pb': 207.2, 'Bi': 208.9804, 'Po': 209, 'At': 210, 'Rn': 222, 'Fr': 223, 'Ra': 226.0254, 'Ac': 227.0278, 'Pa': 231.0359, 'Th': 232.0381, 'Np': 237.0482, 'U': 238.029, 'M':0, 'Ww': 18.01528, 'Dd': 314.47, 'Pp': 179.17, 'Aa': 18.01528, 'Bb': 86.092 }

#==================================================================
#  AUX: create_parser
#==================================================================
def create_parser():

  parser = argparse.ArgumentParser(description='Generates LAMMPS data/settings files from molecule .pdb files based on known parameters from a database file.\
          The last column of each ATOM or HEATM entry in the .pdb file must be an assigned "atom type" that can be queried in the prepared database file.\
                                                Any arguments with multiple values are space-delimited and\
                                                 need to be enclosed by quotation marks.')

  # DO REQUIRED POSITIONAL ARGUMENTS
  parser.add_argument('list_file',help = 'File containing a Space-delimited list of .pdb files that will appear in system in the first column, the number of those molecules to appear in the second column, and whether they are aligned to the x-axis appears in the third column.')

  parser.add_argument('-lmps'     ,dest='data_name',default="sys.data",help = "Name of desired output LAMMPS data file. \
                         (default = sys.data)")

  parser.add_argument('-typing'     ,dest='typing',default=True,help = "Flag that indicates whether types are included in the data file. \
                         (default = True)")

  parser.add_argument('-settings' ,dest='settings_name',default="sys.settings",help = "Name of desired LAMMPS settings file. \
                         (default = sys.settings)")

  parser.add_argument('-init' ,dest='init_type',default="lattice",help = "Keyword specifying the system initialization type. \
          Note that the fixed option additionally requires specification of coordinates within a file named 'coords.init', which lists \
          the center of mass of each molecule and an associated radius (which is useful if there is a major disparity in particle sizes) \
                         (options = lattice, random, fixed) (default = lattice)")

  parser.add_argument('-avoid_overlap' ,dest='avoidFlag',default=True,help = "Flag that indicates whether the code will attempt to avoid overlaps\
          in the initial configuration. You may want to set this option to 'False' if the your system contains a large number of particles or if \
          overlapping configurations are not a big deal/will be handled as part of equilibration. A minimization step is recommended for your MD\
          engine, irrespective of whether the option is True or False. (default: True)")


  parser.add_argument('-solvate' ,dest='solv_pdb',default="none",help = "Specification of a .pdb file for solvent molecules.\
                       (default: none, only the molecules specified in list_file will be included.)")

  parser.add_argument('-buffer' ,dest='nbuff',default=0,help = "Number of buffer cells to exclude around solutes for solvent molecules.\
          Regions containing solute are automatically excluded; this adds an additional buffer region. (default: 0)")

  parser.add_argument('-solv_dens' ,dest='solv_dens',default=1.0,help = "Desired solvent density (g/cm^3).\
                       (default: 1.0, only used if solvate option is also used)")

  parser.add_argument('-box' ,dest='box',default="100.0 100.0 100.0",help = "Specification for box side lengths in x, y, and z\
                        directions. If fewer than 3 values are supplied, then the remaining sides are automatically populated \
                         based on the last entry cubic simulation cell is assumed. \
                         (default = '100. 100. 100.')")

  parser.add_argument('-box0' ,dest='box0',default=None,help = "Specification for box side lengths in x, y, and z\
                        directions to initially place molecules into. This is usually for diffuse system setup, then the coordinates can be subsequently scaled.\
                         (default = '100. 100. 100.')")

  parser.add_argument('-contain' ,dest='contain',default="no",help = "Indicates whether or not box should be adjusted to contain\
    all atoms in the simulation. (default = 'no')")

  parser.add_argument('-exclude' ,dest='excludes',default=None,help = "Interaction types to exclude when writing data file. Should be written as a space-delimited list enclosed by quotes. (default: None, options: bonds, angles, dihedrals)")

  parser.add_argument('-mix' ,dest='mix',default=None,help = "Indicates whether mixing rules should be applied on non-bonded pairwsie coefficients. (default: None, options: 'arithmetic')")

  return parser

#==================================================================
#  AUX: check_arg_consistency
#==================================================================
# this function just condenses a check that a proper number of
# values were specified by the user
def check_arg_consistency(argname,arg,N):
  nval = len(arg)
  if (nval == N):
    pass
    return 1
  elif (nval < N):
    print("****ARGUMENT ERROR: {}****\n Too few arguments ({}) provided for number of molecule types ({}).\n "
     "Better luck next time! Exiting now...\n".format(argname,nval,N))
    exit()
  elif (nval > N):
    print("****ARGUMENT ERROR: {}****\n Too many arguments ({}) provided for number of molecule types ({}).\n "
          "Better luck next time! Exiting now...\n".format(argname,nval,N))
    exit()
  return 0

#==================================================================
#  AUX: convert_args
#==================================================================
def convert_args(args):

  # PARSE REQUIRED ARGUMENTS
  sys   = {}
  flist = open(args.list_file,"r")
  lines = [line.strip().split() for line in flist]
  pdb_list = [line[0] for line in lines]
  nmol     = [int(line[1]) for line in lines]
  align_mol= [line[2] for line in lines]

  # PARSE OPTIONAL ARGUMENTS
  nmol_type    = len(pdb_list)
  chkflg       = check_arg_consistency("nmol",nmol,nmol_type)
  data_name    = args.data_name       		      # name of data file
  settings_name= args.settings_name   	              # name of settings file
  init_type    = args.init_type			      # atom arrangement for placement
  try:
    avoidFlag    = eval(args.avoidFlag)			      # flag of whether to check for overlaps
  except TypeError:
    avoidFlag    = args.avoidFlag			      # flag of whether to check for overlaps
  chkflg       = check_arg_consistency("align",align_mol,nmol_type)
  box          = [float(i) for i in args.box.split()] # Box side lengths
  contain      = args.contain
  while (len(box) < 3):         # if less than 3 box lengths, fill out array
    box.append(box[-1])
  box = np.array(box)

  if args.box0 is None:
      box0 = box.copy()
  else:
    box0         = [float(i) for i in args.box0.split()] # initial Box side lengths
    while (len(box0) < 3):
      box0.append(box0[-1])
    box0 = np.array(box0)

  # CHECK FOR SOLVENT
  solv_pdb    = args.solv_pdb                         # .pdb file for solvent (if desired)
  if (solv_pdb != "none"):
    pdb_list.append(solv_pdb)
    nmol.append(0)
    solv_dens = float(args.solv_dens)
    nbuff = int(args.nbuff)
    sys['buffer']  = nbuff
    nmol_type +=1

  excludeFlags = {'bonds': False, 'angles': False, 'dihedrals': False}
  try:
    excludes = args.excludes.split()
    excludes = [iType.lower() for iType in excludes]
    for iType in excludes:
      if iType in excludeFlags:
        excludeFlags[iType] = True
      else:
        print("Unrecognized interaction type for excludes: {}".format(iType))
        exit()
  except:
    pass
                                                      # replicated in the system
  # INITIALIZE DICTIONARY FOR CONTAINING MOLECULE INFORMATION
  MOLS = {}
  for i in range(nmol_type):
    MOLS[i] = {}
    MOLS[i]['pdb'] = pdb_list[i]
    MOLS[i]['nmol']= nmol[i]
    if (pdb_list[i] == solv_pdb):
      MOLS[i]['solvent'] = solv_dens

  # INITIALIZE DICTIONARY FOR FILES
  files = {}
  #files['data']     = open(data_name,"w")

  #INITIALIZE DICTIONARY FOR SYSTEM BUILDING/OPTIONS
  sys['init'] = init_type
  sys['avoid'] = avoidFlag
  sys['align']= align_mol
  sys['crds'] = []
  sys['box']  = box
  sys['box0']  = box0
  sys['contain']  = contain
  sys['excludes']  = excludeFlags
  sys['mix']   = args.mix
  try:
    sys['typing']  = eval(args.typing)
  except TypeError:
    sys['typing']  = args.typing

  # PRINT OUT CONFIRMATION OF PARAMETERS
#  print("********************************************************")
#  print("# Provided " , repr(nmol_type), " .pdb files for molecule specification...")
#  print("# List of .pdb file(s): ", pdb_list)
#  print("# System to be generated with the following molecules:")
#  for i in range(nmol_type):
#    if solv_pdb != 'none' and pdb_list[i] == solv_pdb:
#      print( "# -- Molecule",repr(i),"-- reported as solvent!")
#    else:
#      print("# -- Molecule",repr(i),"--", repr(nmol[i]),"molecule(s) using", pdb_list[i])
#  print("# System initialization type set to ", init_type)
#  print("# Include types in data file? ", sys['typing'])
#  print("********************************************************")

  return (nmol_type,MOLS,files,sys)

#==================================================================
# AUX: mix_params
#==================================================================
def mix_params(mixType,p1,p2):
  n = len(p1)
  p = [0.0]*n
  if mixType == 'arithmetic':
    for i in range(n):
        p[i] = 0.5*(p1[i]+p2[i])
  elif mixType == 'geometric':
    for i in range(n):
        p[i] = (p1[i]*p2[i])**0.5
  elif mixType =='LB':
    p[0] = (p1[0]*p2[0])**0.5
    p[1] = 0.5*(p1[1]+p2[1])
  elif mixType =='Urry':
    p[0] = 0.5*(p1[0]+p2[0])
    p[1] = 0.5*(p1[1]+p2[1])
    p[2] = 0.5*(p1[2]+p2[2])-0.08 # comes from optimal shift/dilatiion
  return p

#==================================================================
# AUX: get_distance
#==================================================================
def get_distance(ri,rj,L):
  dr   = rj - ri
  indp = dr > 0.5*L
  indn = dr <-0.5*L
  dr[indp] -= L[indp]
  dr[indn] += L[indn]
  return np.sqrt(np.dot(dr,dr)),dr

#==================================================================
# AUX: scale_coordinates
#==================================================================
def scale_coordinates(Lf,L0,r0):
  r0or  = 0.5*L0
  rfor  = 0.5*Lf
  alpha = Lf/L0
  r0_shift = r0 - r0or
  rf_shift = alpha*r0_shift
  rf = rfor + rf_shift
  return rf

#==================================================================
# AUX: quick_pushoff
#==================================================================
def quick_pushoff(sys,ri,xi):

  posi   = ri[:] + xi           # initial positions
  N      = posi.shape[0]        # number of atoms in molecule

  for step in range(npush_max):
    frci   = np.zeros(posi.shape) # pushoff forces on atoms in current molecule

    # compute pushoff forces based on any overlaps with other system atoms
    # pushoff forces are applied to the whole molecule in the direction that depends
    # on individual interatomic distances; however, the magnitude of the force
    # scales with the relative distance, such that atoms that are farther from the
    # push-off point are less affected
    any_overlap = False
    for j,(rj,xj,aj) in enumerate(sys['crds']):
      posj   = rj[:] + xj
      for r_j in posj:
        for i,r_i in enumerate(posi):
          dij,dr = get_distance(np.array(r_i),np.array(r_j),sys['box0'])
          if (dij < rpush):
            if (dij < 0.8*rpush):
              any_overlap = True
            #frci[i,:] -= 0.25*dr*np.sin(np.pi*dij/rpush)/(dij*rpush)
            for k,r_k in enumerate(posi):
              dkj,drki =  get_distance(np.array(r_k),np.array(r_j),sys['box0'])
              frci[k,:] -= 0.25*dr*np.sin(np.pi*dij/rpush)/(dij*rpush)*np.sin(0.5*dij/dkj*np.pi)

    ## compute smaller pushoff forces amongst internal atoms
    #for i in range(N-1):
    #  r_i = posi[i,:]
    #  for j in range(i+1,N):
    #    r_j = posi[j,:]
    #    dij,dr = get_distance(np.array(r_i),np.array(r_j),sys['box0'])
    #    if (dij < rint):
    #      frci[i,:] -= 0.25*dr*np.sin(np.pi*dij/rint)/(dij*rint)
    #      frci[j,:] += 0.25*dr*np.sin(np.pi*dij/rint)/(dij*rint)

    # if no overlaps encountered, then stop doing this.
    if (any_overlap is False):
      print("# Pushoff successful after {} steps...\n\n".format(step+1))
      break

    # now shift positions based on the pushoff forces (mass neglected)
    posi[:,:] += frci[:,:]*0.5

  xinew = posi[:,:] - ri[:]
  return xinew

#==================================================================
# AUX: check_overlaps
#==================================================================
def check_overlaps(sys,ri,xi,ai):
  overlaps = False
  if sys['avoid'] is True:
    # THE FOLLOWING CODE TESTS EACH ATOM EXPLICITLY AND CAN BE SLOW
    # IF THERE ARE MANY ATOMS
    posi   = ri[:] + xi
    ai_avg = 0.5
    for j,(rj,xj,aj) in enumerate(sys['crds']):
      posj   = rj[:] + xj
      aj_avg = 0.5
      for r_j in posj:
        for r_i in posi:
          dij,dr = get_distance(np.array(r_i),np.array(r_j),sys['box0'])
          if (dij < ai_avg + aj_avg):
            overlaps = True
            break
  return overlaps

#==================================================================
# AUX: correct_cell
#==================================================================
def correct_cell(n,ppd):
  if (n < 0):
    n+=ppd
  elif (n >= ppd):
    n-= ppd
  return n

#==================================================================
# AUX: align_to_x
#==================================================================
def align_to_x(mol):
  # 1. get indices for the links
  # 2. get translation vector
  # 3. get end-to-end vector
  n      = len(mol)
  ihead  = 1
  itail  = n
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

  return mol

#==================================================================
# AUX: rotate_molecule
#==================================================================
def rotate_molecule(crds):
  # Select random rotation rangles about x,y, and z axes
  theta = [2*random.random()*np.pi for i in range(3)]

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
  rot_order = list(range(3))
  random.shuffle(rot_order)

  # Perform the random rotation
  rotcrds = crds.copy()
  for j in rot_order:
    rotcrds = np.dot(R[j],rotcrds.transpose())
    rotcrds = rotcrds.transpose()

  return rotcrds

#==================================================================
# AUX: place_molecule
#==================================================================
def place_molecule(molnum,molkey,crds,Rmax,sys):

  placed = False
  att    = 0        # number of placement attempts
  while not placed:
    att += 1
#    print("# Starting Molecule Placement Attempt #{}".format(att))
    # generate possible position
    if sys['init'] == 'lattice':
      rind         = random.randint(0,len(sys['grid']['avail'])-1)
      grdpt        = sys['grid']['avail'][rind]
      center       = sys['grid']['pts'][grdpt][:]
    elif sys['init'] == 'random':
      center = np.array([random.random()*L for L in sys['box0']])
    elif sys['init'] == 'fixed':
      crdsi   = sys['crds0'][molnum]
      center  = crdsi[0:3]
      Rmax    = crdsi[-1]
    else:
      print("# ****PLACEMENT ERROR: {}****\n Unrecognized placement command...\n".format(sys['init']))
      exit()
    if (sys['align'][molkey] == "yes"):
      rotcrds = crds[:]
    else:
      rotcrds = rotate_molecule(crds)

    # check if molecule placement results in overlaps
    overlapping = check_overlaps(sys,center,rotcrds,Rmax)
    if not overlapping:
      placed = True
#      print("# ...Molecule placed after", repr(att), "attempts...")
    elif att > natt_max:
      print("# ****PLACEMENT ERROR****\n")
      print("# Maximum number of attempts reached...")
      print("# Consider increasing box size to avoid overlapping molecules.")
      print("# Attempting pushoff minimization...")
      rotcrds = quick_pushoff(sys,center,rotcrds)
      placed = True

  sys['crds'].append((center,rotcrds,Rmax))

  if sys['init'] == 'lattice':
    grdpt = sys['grid']['avail'].pop(rind)

  return sys

#==================================================================
# AUX: init_interaction_dict()
#==================================================================
def init_interaction_dict():
  the_dict = {}
  the_dict['atoms']     = {}
  the_dict['bonds']     = {}
  the_dict['angles']    = {}
  the_dict['dihedrals'] = {}
  return the_dict

#==================================================================
# AUX: process_coords
#==================================================================
# reads file, eliminating empty lines and new line characters
def process_coords(fid):
  lines = [line.strip() for line in fid]
  lines = [line.split() for line in lines if line]
  coords= [[float(x) for x in line] for line in lines]
  fid.close()
  return coords

#==================================================================
# AUX: quick_file_read
#==================================================================
# reads file, eliminating empty lines and new line characters
def quick_file_read(fid):
  lines = [line.strip() for line in fid]
  return lines

#==================================================================
# AUX: pdb_loose_process
#==================================================================
# reads file, eliminating empty lines and new line characters
def pdb_loose_process(lines):
  atoms = {}
  for line in lines:
    record = line[:6].strip()
    if (record == "HETATM" or record =="ATOM"):
      spltLine = line.split()
      ind=int(spltLine[1])
      typ=spltLine[2]
      crdsLine= line[30:54]
      crd=np.array([float(crdsLine[i:i+8]) for i in range(0,24,8)])
      atoms[ind] = {}
      atoms[ind]['name']     = spltLine[-1] ###modified by Eser, it was previously equal to typ
      atoms[ind]['crds']     = crd
      atoms[ind]['adj']      =  []
    if (record == "CONECT"):
      line = line.split()
      ind=int(line[1])
      adjlist = [int(i) for i in line[2:]]
      atoms[ind]['adj'].extend(adjlist)

  return atoms

#==================================================================
# AUX: solvate_system
#==================================================================
def solvate_system(nsolv,crds,Rmax,sys):
  # DEFINE GRID SPACING
  pts_per_dim   = int(np.ceil(nsolv**(1./3.)))
  spacing       = [ d/pts_per_dim for d in sys['box0']]
#  print("# Creating {}-point mesh for solvent molecule placement".format(pts_per_dim**3))
#  for adir,a in zip(dirs,spacing):
#    print("# Grid spacing in {}-direction: {:>10.5f}".format(adir,a))

  # CREATE MESH FOR POSSIBLE SOVLENT PLACEMENT
  pts = []
  for nx in range(pts_per_dim):
    xpos = (nx+0.5)*spacing[0]
    for ny in range(pts_per_dim):
      ypos = (ny+0.5)*spacing[1]
      for nz in range(pts_per_dim):
        zpos = (nz+0.5)*spacing[2]
        cntr = np.array([xpos,ypos,zpos])
        pts.append(cntr)

  # EXCLUDE ANY POINTS THAT ALREADY CONTAIN SYSTEM ATOMS
  nbuff = sys['buffer']
  xclude_list = set()
  for i,(ri,xi,ai) in enumerate(sys['crds']):
    for rj in xi:
      crdsj = ri[:] + rj[:]
      # wrap back into box
      indp  = crdsj > sys['box0'][:]
      indn  = crdsj < sys['box0'][:]*0
      crdsj[indp]  -= sys['box0'][indp]
      crdsj[indn]  += sys['box0'][indn]
      n = [int(rk/dk) for rk,dk in zip(crdsj,spacing)]
      # add containing cell and neighboring cells (as buffer) to exclude list
      for nx in range(n[0]-nbuff,n[0]+nbuff+1,1):
        nx = correct_cell(nx,pts_per_dim)
        for ny in range(n[1]-nbuff,n[1]+nbuff+1,1):
          ny = correct_cell(ny,pts_per_dim)
          for nz in range(n[2]-nbuff,n[2]+nbuff+1,1):
            nz = correct_cell(nz,pts_per_dim)
            icell = nz + ny*pts_per_dim + nx*pts_per_dim**2
            xclude_list.add(icell)

  # DELETE THESE FROM GRID
  for ex in sorted(list(xclude_list))[::-1]:
    del pts[ex]

  # ADD SOLVENT TO THE SYSTEM
  nsolv = len(pts)
  print("# Adding {} solvent molecules to system...".format(nsolv))
  for cntr in pts:
    sys['crds'].append((cntr,rotate_molecule(crds),Rmax))

  return sys,nsolv

#==================================================================
# AUX: define_system_grid
#==================================================================
def define_system_grid(sys,ntot):
  pts_per_dim   = int(np.ceil(ntot**(1./3.)))
  #spacing       = [ d/pts_per_dim for d in sys['box']]
  spacing       = [ d/pts_per_dim for d in sys['box0']]
  pts           = []
#  print("# Creating {}-point mesh for molecule placement".format(pts_per_dim**3))
#  for adir,a in zip(dirs,spacing):
#    print("# Grid spacing in {}-direction: {:>10.5f}".format(adir,a))
  for nx in range(pts_per_dim):
    xpos = (nx+0.5)*spacing[0]
    for ny in range(pts_per_dim):
      ypos = (ny+0.5)*spacing[1]
      for nz in range(pts_per_dim):
        zpos = (nz+0.5)*spacing[2]
        pts.append([xpos,ypos,zpos])
  avail = list(range(pts_per_dim**3))
  sys['grid'] = {'pts': pts,'avail': avail,'occ': []}
  return sys,spacing

#==================================================================
# AUX: make_system_geometry
#==================================================================
def make_system_geometry(sys,MOLS):
#  print("********************************************************")
#  print("~~~~~~~~~~~~~~~~~MOLECULE PLACEMENT ~~~~~~~~~~~~~~~~~~~~")
#  print("********************************************************")

  crds = []
  # GET TOTAL NUMBER OF MOLECULES AND ATOMS (EXCLUDING SOLVENT)
  nmol_tot = sum([MOLS[molkey]['nmol'] for molkey in MOLS])

  # DEFINE GRID IF NECESSARY
  if (sys['init']== 'lattice'):
    sys,a = define_system_grid(sys,nmol_tot)
  elif (sys['init'] == 'fixed'):
    sys['crds0'] = process_coords(open("coords.init","r"))
#  print("********************************************************")

  # LOOP OVER EACH MOLECULE TYPE
  for molkey,the_mol in MOLS.items():
    # COLLECT ATOM COORDINATES FOR THE MOLECULE
    natms   = len(the_mol['atoms'])
    molcrds = np.zeros([natms,3])
    molmass = np.zeros([natms])
    molchrg = np.zeros([natms])
    the_mol['atoms'] = align_to_x(the_mol['atoms'])
    for the_atm,prop in the_mol['atoms'].items():
      if len(prop['name'])>1:
          if prop['name'][1].islower()==True:
              elementname = prop['name'][:2]
          else:
              elementname = prop['name'][0]
      else:
          elementname = prop['name'][0]
      molcrds[the_atm-1,:] = prop['crds'][:]
      molmass[the_atm-1]   = mass_dictionary[elementname]

    # CALCULATE CENTER OF MASS AND SHIFT COORDINATES
    mtot    = sum(molmass)
    Rcom    = np.dot(molcrds.transpose(),molmass) / mtot
    molcrds = molcrds - Rcom[:]
#    print("\n****************MOLECULE TYPE {}*************************\n".format(molkey))
#    print("#----Molar mass for molecule {:>3}: {:<10.5f}".format(molkey,mtot))

    # NOW COMPUTE MAX-CONTAINING RADIUS
    R2      = np.einsum('ij,ij->i',molcrds,molcrds)
    Rmax    = max(R2)**0.5
    MOLS[molkey]['rmax'] = Rmax*1.0
#    print("#----Containing radius for molecule {:>3}: {:<10.5f}\n".format(molkey,Rmax))
    if (sys['init']== 'lattice'):
      if (any(Rmax > a)):
        print("# *****************WARNING******************\n"
        "# Container radius for molecule {} is larger than grid spacing...\n"
        "# ******************************************\n".format(molkey))

    # PLACE STANDARD MOLECULES
    if 'solvent' not in the_mol:
      for molnum in range(the_mol['nmol']):
#        print("# Placing molecule {} of type {}...".format(molnum,molkey))
        sys = place_molecule(molnum,molkey,molcrds,Rmax,sys)
  #    print("********************************************************\n")
    # HANDLE SOLVENT MOLECULES
    else:
      # DETERMINE NUMBER OF SOLVENT MOLECULES FOR GIVEN BOX SIZE
      dens      = the_mol['solvent']
      nsolv     = int(dens * np.prod(sys['box0']) * Na / 1e24 / mtot)
      nperside  = np.ceil(nsolv**(1./3))
      # EXPAND BOX TO ENABLE SOLVATION
      nsolv_new = int(nperside)**3
      scale     = (1.0*nsolv_new/nsolv)**(1.0/3)
      sys['box0']= np.array([i*scale for i in sys['box0']])
      sys,nsolv = solvate_system(nsolv_new,molcrds,0.25*Rmax,sys)
      MOLS[molkey]['nmol'] = nsolv
      print("********************************************************\n")
  # PRINT OUT COORDINATES TO XYZ AND LAMMPSTRJ
  natm_tot = sum([MOLS[molkey]['nmol']*len(MOLS[molkey]['atoms']) for molkey in MOLS])
  fid   = open('init.pdb',"w")

  fid.write("COMPND    UNNAMED\n".format(natm_tot))
  fid.write("AUTHOR    GENERATED BY MODIFIED GENDATA (by Eser)\n")
  fid.write("CRYST1{0:>9.3f}{1:>9.3f}{2:>9.3f}  90.00  90.00  90.00 P 1           1\n".format(sys['box'][0],sys['box'][1],sys['box'][2]))
  molid = 0
  ind   = 0
  for molkey,the_mol in MOLS.items():
    for molnum in range(the_mol['nmol']):
      the_crds = sys['crds'][molid]
      center   = the_crds[0]
      mol_crds = center + the_crds[1].copy()
      molid+=1
      for i,icrds in enumerate(mol_crds):
        ind +=1
        attype  = MOLS[molkey]['atoms'][i+1]['name'].capitalize() + '{}'.format(i+1)
        scrds = scale_coordinates(sys['box'],sys['box0'],icrds)
        if attype not in ['H1','H2','O','M']:
            if 'W' in attype:
                moltype = 'WWW'
            elif 'D' in attype:
                moltype = 'DDD'
            elif 'A' in attype:
                moltype = 'AAA'
            elif 'B' in attype:
                moltype = 'BBB'
            fid.write("{0: <4}{1:>7} {2:<5}{3:<3}{4:>6}{5:>12.3f}{6:>8.3f}{7:>8.3f}{8:>6.2f}{9:>6.2f}".format('ATOM',ind,attype,moltype,molid,scrds[0],scrds[1],scrds[2],1,0)+ '{:>12}'.format(''.join([x for x in attype.strip() if x.isalpha()])) + '\n')
        else:
            fid.write("{0: <4}{1:>7} {2:<5}{3:<3}{4:>6}{5:>12.3f}{6:>8.3f}{7:>8.3f}{8:>6.2f}{9:>6.2f}".format('ATOM',ind,attype,'AAA',molid,scrds[0],scrds[1],scrds[2],1,0)+ '{:>12}'.format(''.join([x for x in attype.strip() if x.isalpha()])) + '\n')
            #        fid.write("{:<8} {:>10.5f} {:>10.5f} {:>10.5f}\n".format(attype,scrds[0],scrds[1],scrds[2]))

  fid.close()
  return sys

#==================================================================
#  AUX: find_bonds
#=================================================================
def find_bonds(atoms):
  bonds = set()
  for atmkey in atoms:
    for j in atoms[atmkey]['adj']:
      if (atmkey < j):
        bonds.add((tuple([1]),(atmkey,j)))
  return bonds

#==================================================================
#  AUX: find_angles
#=================================================================
def find_angles(atoms,bonds):
  angles = set()

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
#  AUX: get_mol_spec
#=================================================================
def get_mol_spec(MOLS):

  # Read each pdb file and store information in MOLS dictionary
  for molkey in MOLS:
    with open(MOLS[molkey]['pdb'],"r") as fid:
      #print("")
  #    print("********************************************************")
  #    print("# PROCESSING MOLECULE ", repr(molkey), "...")
      lines     = quick_file_read(fid)
      atoms     = pdb_loose_process(lines)     #
      bonds     = find_bonds(atoms)
      angles    = find_angles(atoms,bonds)
      dihedrals = find_dihedrals(atoms,angles)
 #     print("# Topology for molecule determined as follows:")
 #     print("# --Number of atoms    : ",repr(len(atoms)))
 #     print("# --Number of bonds    : ",repr(len(bonds)))
 #     print("# --Number of angles   : ",repr(len(angles)))
 #     print("# --Number of dihedrals: ",repr(len(dihedrals)))
 #     print("********************************************************")

    # ADD RESULTS TO MOLECULE DICTIONARY
    MOLS[molkey]['atoms']     = atoms.copy()
    MOLS[molkey]['bonds']     = list(bonds)
    MOLS[molkey]['angles']    = list(angles)
    MOLS[molkey]['dihedrals'] = list(dihedrals)
  return MOLS

#==================================================================
#  AUX: count_parameters
#=================================================================
# counts the number of parameters included for the interaction time
def count_parameters(param_type):
  #tmp = [x[1].values()[0] for x in param_type]
  tmp = [list(x[1].values())[0] for x in param_type]
  return sum([len(x) if isinstance(x,list) else 1 for x in tmp ])

#==================================================================
# AUX: get_named_type
#==================================================================
# creates a tuple of atom names for a given class of interaction types
# this tuple is used for finding parameters
def get_named_type(molid,ind,MOLS):
  return tuple([MOLS[molid]['atoms'][i]['name'] for i in ind])

#==================================================================
# AUX: assign_numeric_type
#==================================================================
# assigns numeric type to the given class of interaction based on
# growing list given by the 'avail' structure
def assign_numeric_type(the_class,the_type,the_params,avail):
  if the_type not in avail[the_class]:
    ninc = 1
    if 'type' in the_params:
        if the_params['type'] == "charmm":
          tmp = list(the_params.values())[1]
          ninc = len(tmp) if isinstance(tmp,list) else 1
    navail = count_parameters(avail[the_class].values())
    avail[the_class][the_type] = (list(range(navail+1,navail+1+ninc,1)),the_params)
  return avail,avail[the_class][the_type][0]

#==================================================================
#  AUX: parameter_lookup
#=================================================================
# this function goes through the molecules to link parameters to
# atoms, bonds, angles, and dihedrals
def parameter_lookup(MOLS,ff_db,excludeFlags):
  # DEFINE DICTIONARIES FOR PARAMETER STORAGE
  # these are used to track which parameters we have
  # and which are need to complete the program
  # the different keys point to sets to avoid redundancies
  present = init_interaction_dict()
  missing = {'atoms': set(),'bonds':set(),'angles':set(),'dihedrals':set()}

  params=0
  for molkey in MOLS:
    # ATOM TYPE LOOKUP
    for atm in MOLS[molkey]['atoms']:
      atmtype = MOLS[molkey]['atoms'][atm]['name']
      # check if atom type is available
      try:
        params = ff_db.atm_types[atmtype].copy()
        present,num_type = assign_numeric_type('atoms',atmtype,params,present)
        MOLS[molkey]['atoms'][atm]['type']   = num_type
        MOLS[molkey]['atoms'][atm]['mass']   = params['m']
        MOLS[molkey]['atoms'][atm]['charge'] = params['q']
        if 'pot' in params.keys():
          MOLS[molkey]['atoms'][atm]['pot'] = params['pot'].upper()
        else:
          MOLS[molkey]['atoms'][atm]['pot'] = 'LJ'
      except KeyError:
        missing['atoms'].add(atmtype)
        print("# ****ATOM TYPE ERROR: atom {}****\n Database missing type {}...\n".format(atm,atmtype))

    # WRITE AN ATOM MAP FILE
    #atom_map = [(num[0][0],name) for name,num in present['atoms'].items()]
    atom_map = {num[0][0]:name for name,num in present['atoms'].items()}
    fid = open("lmps2type.map","w")
    for num in sorted(atom_map.keys()):
      fid.write("{:<5d} {:<10}\n".format(num,atom_map[num]))
    fid.close()

    # BOND TYPE LOOKUP
    if excludeFlags['bonds'] is False:
      for i,bon in enumerate(MOLS[molkey]['bonds']):
        btype = get_named_type(molkey,bon,MOLS)
        try:
          params = ff_db.bon_types[btype].copy()
          present,num_type = assign_numeric_type('bonds',btype,params,present)
          MOLS[molkey]['bonds'][i] = (num_type,bon)
        except KeyError:
          try:
            btype = btype[::-1]
            params = ff_db.bon_types[btype].copy()
            present,num_type = assign_numeric_type('bonds',btype,params,present)
            MOLS[molkey]['bonds'][i] = (num_type,bon)
          except KeyError:
            print("# ****BOND TYPE ERROR: atoms {} & {} ****\n Database missing bond {}--{}...\n".\
            format(bon[0],bon[1],btype[0],btype[1]))
            # if the reverse has not already been added, add to the set of missing terms
            if btype[::-1] not in missing['bonds'] and btype not in missing['bonds']:
              if (btype[0][0] < btype[-1][0]):
                missing['bonds'].add(btype)
              else:
                missing['bonds'].add(btype[::-1])

    # ANGLE TYPE LOOKUP
    if excludeFlags['angles'] is False:
      for i,ang in enumerate(MOLS[molkey]['angles']):
        atype = get_named_type(molkey,ang,MOLS)
        try: # try to extract parameters from data base
          params = ff_db.ang_types[atype].copy()
          present,num_type = assign_numeric_type('angles',atype,params,present)
          MOLS[molkey]['angles'][i] = (num_type,ang)
        except KeyError:
          try:
            atype = atype[::-1]
            params = ff_db.ang_types[atype].copy()
            present,num_type = assign_numeric_type('angles',atype,params,present)
            MOLS[molkey]['angles'][i] = (num_type,ang)
          except KeyError: # if no parameters, raise error and write to file
            print("# ****ANGLE TYPE ERROR: atoms {}-{}-{} ****\n Database missing angle {}--{}--{}...\n".\
              format(ang[0],ang[1],ang[2],atype[0],atype[1],atype[2]))
            if atype[::-1] not in missing['angles'] and atype not in missing['angles']:
              if (atype[0][0] < atype[-1][0]):
                missing['angles'].add(atype)
              else:
                missing['angles'].add(atype[::-1])

    # DIHEDRAL TYPE LOOKUP
    if excludeFlags['dihedrals'] is False:
      for i,dih in enumerate(MOLS[molkey]['dihedrals']):
        dtype = get_named_type(molkey,dih,MOLS)
        try:   # try to extract parameters from database
          params = ff_db.dih_types[dtype].copy()
          present,num_type = assign_numeric_type('dihedrals',dtype,params,present)
          MOLS[molkey]['dihedrals'][i] = (num_type,dih)
        except KeyError:
          try:
            dtype = dtype[::-1]
            params = ff_db.dih_types[dtype].copy()
            present,num_type = assign_numeric_type('dihedrals',dtype,params,present)
            MOLS[molkey]['dihedrals'][i] = (num_type,dih)
          except KeyError:  # if no parameters, print error, and write parameter to file
            print("# ****DIHEDRAL TYPE ERROR: atoms {}-{}-{}-{} ****\n Database missing dihedral {}--{}--{}--{}...\n".\
              format(dih[0],dih[1],dih[2],dih[3],dtype[0],dtype[1],dtype[2],dtype[3]))
            # if the reverse has not already been added, add to the set of missing terms
            if dtype[::-1] not in missing['dihedrals'] and dtype not in missing['dihedrals']:
              if (dtype[0][0] < dtype[-1][0]):
                missing['dihedrals'].add(dtype)
              else:
                missing['dihedrals'].add(dtype[::-1])


  # PRINT PARAMETER SUMMARY
  print("")
  chk_order = ['atoms','bonds','angles','dihedrals']
  #print("\n\n********************************************************")
  print("~~~~~~~~~~~AVAILABLE PARAMETER SUMMARY ~~~~~~~~~~~~~~~~~")
###  print("********************************************************")
  for param in chk_order:
    print("# Number of available parameter sets for {}: {}".format(param,len(present[param])))

  # PRINT LIST OF PARAMETERS THAT ARE REQUIRED FOR THE SIMULATION AND EXIT
  print( "\n\n********************************************************")
  print("~~~~~~~~~~~~~~MISSING PARAMETER SUMMARY~~~~~~~~~~~~~~~~~")
  print("********************************************************")
  if (any([len(i) for i in missing.values()])):
    ff_db.make_missing_dir()
    for param in chk_order:
      print("# Number of missing parameter sets for {}: {}".format(param,len(missing[param])))
      for cnt,needed in enumerate(sorted(missing[param])):
        print("#--Missing {} {}: {}".format(param[:-1],cnt,needed[0:]))
        ff_db.print_missing_data[param](needed)
    print("\n\n#$$$$$$$$$~~~~~EXITING DUE TO MISSING PARAMETERS~~~~~$$$$$$$$$")
    exit()
  else:
    print("********************************************************\n\n")
    print("\n\n# Congratulations! There are no missing parameters!")
  print("********************************************************\n\n")

  # RE-MAP THE PARAMETER SETS
  param_types = init_interaction_dict()
  for the_class in param_types:
    param_types[the_class] = sorted(present[the_class].values())

  return MOLS,param_types,atom_map

#==================================================================
#  AUX: gen_settings_file
#=================================================================
# writes the settings file with all force-field information
#deleted by Eser
#==================================================================
#  AUX: finish_data
#=================================================================
# finishes the write to data file
def finish_data(sys,MOLS,files,number):
#  print("# Writing remainder of data file...")
  #fid = files['data']

  # WRITE OUT THE ATOMS SECTION
  #fid.write("\n{}\n\n".format("Atoms"))
  atid = 0
  molid= 0
  for molkey,the_mol in MOLS.items():
    for molnum in range(the_mol['nmol']):
      cntr_crds = sys['crds'][molid][0]
      the_crds  =  cntr_crds + sys['crds'][molid][1].copy()
      for i, icrds in enumerate(the_crds):
        atid  +=1
        scrds = scale_coordinates(sys['box'],sys['box0'],icrds)
   #     fid.write("{:<10}{:>5} {:>15.10f} {:>15.10f} {:15.10f} 0 0 0\n".\
          #format(atid,molid+1,scrds[0],scrds[1],scrds[2]))
      molid+=1

  # WRITE OUT BONDS,ANGLES, and DIHEDRALS SECTIONS
  bondlist = []
  keys = ['Bonds']
  for ptype in keys:
    if (number[ptype.lower()]) > 0:
      #fid.write("\n{}\n\n".format(ptype))
      cntr  = 0
      shift = 0
      for molkey in MOLS:
        for j in range(MOLS[molkey]['nmol']):
          for pi in MOLS[molkey][ptype.lower()]:
            for subpi in pi[0]:
              cntr += 1
    #          fid.write("{:<2} {}".format(cntr,subpi))
              minibond = []
              for pij in pi[1]:
                minibond.append(pij+shift)
      #          fid.write(" {}".format(pij+shift))
              bondlist.append(sorted(minibond))
     #         fid.write("\n")
          shift += len(MOLS[molkey]['atoms'])
  megabondlist = []
  if bondlist == []:
      maxbondindex = 0
  else:
      maxbondindex = max([max(i) for i in bondlist])
  for i in range(1,maxbondindex+1):
      indexbond = [i]
      for j in bondlist:
          if i in j:
              neigh = [n for n in j if n != i][0]
              indexbond.append(neigh)
      indexbond = [indexbond[0]] + sorted(indexbond[1:])
      megabondlist.append(indexbond)
  fid   = open('init.pdb',"a")
  for i in megabondlist:
      fid.write("CONECT")
      for j in i:
          fid.write("{:>5}".format(j))
      fid.write("\n")
      #if len(i) == 2:
      #    fid.write("CONECT{:>5}{:>5}\n".format(i[0],i[1]))
      
  fid.close()
  return None

#==================================================================
#  AUX: gen_data_header
#=================================================================
# writes the initial portion of the data file up to coordinate specifciation
def gen_data_header(sys,MOLS,files):

#  print("# Writing header for data file...")
  #fid = files['data']
  #fid.write("#LAMMPS data file generated on {} via gen_data.py, version {} {}\n\n".\
  #     format(the_date,version_num,version_date))

  # AGGREGATE TOTAL/TYPE NUMBER OF ATOMS, BONDS, ANGLES, DIHEDRALS EXPECTED
  number = {'atoms': 0,'bonds': 0,'angles': 0,'dihedrals': 0}
  ntypes = {'atoms': 0,'bonds': 0,'angles': 0,'dihedrals': 0}
  for key in number:
    if key == 'atoms' or sys['excludes'][key] == False:
      for molkey in MOLS:
        if key == 'atoms':
          number[key] += MOLS[molkey]['nmol']*len(MOLS[molkey][key])
        elif key == 'bonds':
          number[key] += MOLS[molkey]['nmol']*sum([len(i[0]) for i in MOLS[molkey][key]])
  #fid.write("{} bonds\n".format(number['bonds']))
  #fid.write("{} bond types\n".format(ntypes['bonds']))
  #fid.write("{} angles\n".format(number['angles']))
  #fid.write("{} angle types\n".format(ntypes['angles']))
  #fid.write("{} dihedrals\n".format(number['dihedrals']))
  #fid.write("{} dihedral types\n\n".format(ntypes['dihedrals']))
  #fid.write("\n")

  # BOX INFORMATION
  if (sys['contain'] == 'no'):
    donothing=1 #added by eser
#    print("# Container flag set to 'no'\n# Using prescribed box...\n")
    #fid.write("{:<25.16e}{:<25.16e} xlo xhi\n".format(0.0,sys['box'][0]))
    #fid.write("{:<25.16e}{:<25.16e} ylo yhi\n".format(0.0,sys['box'][1]))
    #fid.write("{:<25.16e}{:<25.16e} zlo zhi\n\n".format(0.0,sys['box'][2]))
  elif (sys['contain'] == 'yes'):
    print("# Container flag set to 'yes'\n# Using minimal containing box...\n")
    the_max = []
    the_min = []
    for the_crds in sys['crds']:
      center   = the_crds[0]
      mol_crds = center + the_crds[1].copy()
      the_max.append(list(np.array(mol_crds).max(axis=0)))
      the_min.append(list(np.array(mol_crds).min(axis=0)))
    pos_box = list(np.array(the_max).max(axis=0))
    neg_box = list(np.array(the_min).min(axis=0))
    #fid.write("{:<25.16e}{:<25.16e} xlo xhi\n"  .format(floor(neg_box[0]-1.0),ceil(pos_box[0]+1.0)))
    #fid.write("{:<25.16e}{:<25.16e} ylo yhi\n"  .format(floor(neg_box[1]-1.0),ceil(pos_box[1]+1.0)))
    #fid.write("{:<25.16e}{:<25.16e} zlo zhi\n\n".format(floor(neg_box[2]-1.0),ceil(pos_box[2]+1.0)))

  # MASSES

  # PAIR COEFFICIENTS
  #fid.write("\nPair Coeffs\n\n")
  #for i,atmi in enumerate(params['atoms']):
  #  fid.write("{:<2} ".format(atmi[0][0]))
  #  params = atmi[1]  # dictionary of available parameters for the atom type
  #  for pi in ['eps','sig','e14','s14']:
  #    try:
  #      fid.write("{:>8.5f} ".format(params[pi]))
  #    except KeyError:
  #      if pi == "sig" and "r" in params.keys():
  #        sig = 2*params['r'] / 2.0**(1.0/6.0) # rmin = 2^(1/6)*sigma
  #        fid.write("{:>8.5f} ".format(sig))
  #      elif pi =="s14" and "r14" in params.keys():
  #        s14 = 2*params['r14'] / 2.0**(1.0/6.0) # rmin = 2^(1/6)*sigma
  #        fid.write("{:>8.5f}".format(s14))
  #  fid.write("\n")

  return number


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  MAIN: _main
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def main(argv):
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # PARSE ARGUMENTS
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  t0 = time.time()
  parser = create_parser()
  args                         = parser.parse_args()
  (nmol_type,MOLS,files,sys) =\
                                 convert_args(args)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # INTERPRET MOLECULE STRUCTURE
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # MOLS will contain references for each molecule .pdb for
  # the geometry and connectivity, separated in terms of bonds,
  # angles, and dihedrals.
  # MOLS keys for each molecules: 'atoms','bonds','angles','dihedrals','pdb','nmol'
  MOLS = get_mol_spec(MOLS)
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # LOOK UP PARAMETERS FOR ALL INTERACTION TYPES
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Following this, MOLS is updated to include a numerical
  # type reference for the atoms, bonds, angles, and dihedrals
  # In the 'atoms' field, this is added as through the 'type' key
  # for each atom index. In all other fields, it precedes a list
  # of the affected atom indices in a tuple

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # CREATE SYSTEM GEOMETRY
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  sys = make_system_geometry(sys,MOLS)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # GENERATE DATA FILE HEADER
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  number = gen_data_header(sys,MOLS,files)
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # GENERATE SETTINGS FILE
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # FINISH WRITING THE DATA FILE
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  finish_data(sys,MOLS,files,number)

#  print("The elapsed time is {} seconds.".format(time.time()-t0))
  return

#==================================================================
#  RUN PROGRAM
#==================================================================
if __name__ == "__main__":
  main(sys.argv[1:])

