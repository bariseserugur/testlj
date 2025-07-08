import numpy as np
import pandas
from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
import time
import copy
sys.path.append('/scratch/gpfs/WEBB/bu9134/lj_test/simulation_files')
from add_LJ_force import custom_LJ_force,system_from_topology,add_polymer_bonds,exclude_12_13_14

for i in open('../mixed_parameters.txt','r').readlines():
    exec(i)
for i in open('../simulation_files/simulation_settings.txt','r').readlines():
    exec(i)

T_HIGH = 1000

# Simulation parameters
PDB = PDBFile('init.pdb')
TOPOLOGY = PDB.topology

force = custom_LJ_force(TOPOLOGY,epsilon_matrix,sigma_matrix,cutoff) # custom nonbonded force with given epsilon and sigma parameters
force = exclude_12_13_14(force,TOPOLOGY) # within polymer, exclude 1-2, 1-3, and 1-4 interactions

system = system_from_topology(TOPOLOGY, mass_dict)
system = add_polymer_bonds(system,TOPOLOGY) # add constraints between polymer beads

force_index = system.addForce(force)

# Define the ensemble to be simulated in.
integrator = LangevinIntegrator(T_HIGH, 1 / picosecond, TIMESTEP * picoseconds)
barostat = MonteCarloBarostat(PRESSURE, T_HIGH, 1000000000)
system.addForce(barostat)

# Create simulation
simulation = Simulation(topology=TOPOLOGY, system=system, integrator=integrator, platform=platform, platformProperties=platform_properties)
simulation.context.setPositions(PDB.positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(T_HIGH)

# Add reporters
simulation.reporters.append(StateDataReporter(False,'cubic_pellet.avg', THERMO_FREQ, 
step=True, 
time=True, 
density=True, 
totalEnergy=True, 
kineticEnergy=True, 
volume=True, 
potentialEnergy=True, 
temperature=True))

simulation.reporters.append(PDBReporter('cubic_pellet.lammpstrj',PRINT_VEL,COORDS_FREQ))

#Run simulation
tinit = time.time()

TIME_IN_NS = 10
TOTAL_STEPS = int(TIME_IN_NS*1000/TIMESTEP)
barostat.setFrequency(25)

simulation.step(TOTAL_STEPS)

for T in np.linspace(T_HIGH,300,100):
    integrator.setTemperature(T)
    simulation.step(TOTAL_STEPS//100)

simulation.context.setParameter(MonteCarloBarostat.Temperature(), 300*kelvin)

TIME_IN_NS = 100
TOTAL_STEPS = int(TIME_IN_NS*1000/TIMESTEP)

for i in range(100):
    simulation.step(TOTAL_STEPS//100)

tfinal = time.time()
stdout.write(f"Simulation time: {tfinal - tinit:.2f} seconds\n")
Restart.save_simulation('cubic_pellet.save',simulation,'classical')
