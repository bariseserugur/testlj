# the following initializes a dictionary that will contain various atom cap types
# each cap type will itself be a mimic of the molecule structure in the main file
# this code is probably better suited to making all the molecules their own class,
# but for consistency, we will stick with the dictionary structure.
import numpy as np

caps = {}
caplbl = ["tail","head"]
def add_generic_cap(capType):
  caps[capType] = {}
  lbls  = [capType]
  crds  = [np.array([0.,0.,0.])]
  adjs  = [[]]
  atoms  = {}
  for i,(lbl,crd,adj) in enumerate(zip(lbls,crds,adjs)):
    atoms[i+1] = {'name': lbl,'crds': crd, 'adj': adj}
  caps[capType] = atoms

# SIMPLE HYDROGEN CAP GROUP
capType = 'hydrogen'
caps[capType] = {}
lbls  = ['H']
crds  = [np.array([0.,0.,0.])]
adjs  = [[]]
atoms  = {}
for i,(lbl,crd,adj) in enumerate(zip(lbls,crds,adjs)):
  atoms[i+1] = {'name': lbl,'crds': crd, 'adj': adj}
caps[capType] = atoms

# SIMPLE METHYL CAP GROUP
capType = 'methyl'
caps[capType] = {}
lbls  = ['C','H','H','H']
crds  = [np.array([0.,0.,0.]),np.array([0.5,0.7,0.7]),np.array([0.5,-0.8,0]),np.array([0.2,0.5,-0.96]) ]
adjs  = [[2,3,4],[1],[1],[1]]
atoms  = {}
for i,(lbl,crd,adj) in enumerate(zip(lbls,crds,adjs)):
  atoms[i+1] = {'name': lbl,'crds': crd, 'adj': adj}
caps[capType] = atoms

# SIMPLE HYDROXYL GROUP
capType = 'hydroxy'
caps[capType] = {}
lbls  = ['O','H']
crds  = [np.array([0.,0.,0.]),np.array([0.71,0.0,0.71])]
adjs  = [[2],[1]]
atoms  = {}
for i,(lbl,crd,adj) in enumerate(zip(lbls,crds,adjs)):
  atoms[i+1] = {'name': lbl,'crds': crd, 'adj': adj}
caps[capType] = atoms

# SIMPLE METHOXY GROUP
capType = 'methoxy'
caps[capType] = {}
lbls  = ['C','H','H','O','H']
crds  = [np.array([0.,0.,0.]),np.array([0.6,0.6,0.6]),np.array([0.6,-0.6,0.6]),np.array([1.2,0.0,-1.2]),np.array([1.9,0.0,-0.5])]
adjs  = [[2,3,4],[1],[1],[1,5],[4]]
atoms  = {}
for i,(lbl,crd,adj) in enumerate(zip(lbls,crds,adjs)):
  atoms[i+1] = {'name': lbl,'crds': crd, 'adj': adj}
caps[capType] = atoms

# SIMPLE ETHYL GROUP
capType = 'ethyl'
caps[capType] = {}
lbls  = ['C','H','H','C','H','H','H']
crds  = [np.array([0.,0.,0.]),np.array([0.6,0.6,0.6]),np.array([0.6,-0.6,0.6]),np.array([1.2,0.0,-1.2]),np.array([1.9,0.0,-0.5]),\
        np.array([1.8,0.6,-1.8]),np.array([1.8,-0.6,-1.8])]
adjs  = [[2,3,4],[1],[1],[1,5],[4],[4],[4]]
atoms  = {}
for i,(lbl,crd,adj) in enumerate(zip(lbls,crds,adjs)):
  atoms[i+1] = {'name': lbl,'crds': crd, 'adj': adj}
caps[capType] = atoms

# SIMPLE ETHOXY GROUP
capType = 'ethoxy'
caps[capType] = {}
lbls  = ['C','H','H','C','O','H','H','H']
crds  = [np.array([0.,0.,0.]),np.array([0.6,0.6,0.6]),np.array([0.6,-0.6,0.6]),np.array([1.2,0.0,-1.2]),np.array([2.0,0.0,0.0]),\
        np.array([1.8,0.6,-1.8]),np.array([1.8,-0.6,-1.8]),np.array([2.7,0,-0.7])]
adjs  = [[2,3,4],[1],[1],[1,5],[4,8],[4],[4],[5]]
atoms  = {}
for i,(lbl,crd,adj) in enumerate(zip(lbls,crds,adjs)):
  atoms[i+1] = {'name': lbl,'crds': crd, 'adj': adj}
caps[capType] = atoms
