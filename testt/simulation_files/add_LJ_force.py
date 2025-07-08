import numpy as np
import pandas
from openmm import *
from openmm.app import *
from openmm.unit import *
#from scipy.spatial import cKDTree
#from numba import njit

def custom_LJ_force(TOPOLOGY,epsilon_matrix,sigma_matrix,cutoff):
    TOPOLOGY_ATOMS = list(TOPOLOGY.atoms())
    natom = TOPOLOGY.getNumAtoms()

    # Define the force field as a "force" object to be added to the system.
    force = CustomNonbondedForce("4*epsilon_matrix(type1, type2)*((sigma/r)^12 - (sigma/r)^6);sigma = sigma_matrix(type1, type2)")
    force.addPerParticleParameter("type")  # Particle type as an integer
    species_list = [TOPOLOGY_ATOMS[atom_ix].residue.name for atom_ix in range(natom)]
    species_map = {'AAA': 0, 'WWW': 1, 'BBB': 2}
    for s in species_list:
        force.addParticle([species_map[s]])
    
    # Create epsilon lookup table
    n_types = 3
    epsilon_lookup = np.zeros((n_types, n_types))
    for (a, b), eps in epsilon_matrix.items():
        eps = eps.value_in_unit(kilojoule_per_mole)  # Convert to kJ/mol
        i, j = species_map[a], species_map[b]
        epsilon_lookup[i][j] = eps
        epsilon_lookup[j][i] = eps  # Ensure symmetry
    force.addTabulatedFunction("epsilon_matrix", Discrete2DFunction(n_types, n_types, epsilon_lookup.flatten()))
    
    sigma_lookup = np.zeros((n_types, n_types))
    for (a, b), sig in sigma_matrix.items():
        i, j = species_map[a], species_map[b]
        sigma_lookup[i][j] = sig
        sigma_lookup[j][i] = sig  # Ensure symmetry
    force.addTabulatedFunction("sigma_matrix", Discrete2DFunction(n_types, n_types, sigma_lookup.flatten()))
    # force.addInteractionGroup(range(len(species_list)), range(len(species_list)))
    force.setCutoffDistance(cutoff)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(0.9 * cutoff)
    force.setUseLongRangeCorrection(True)
    force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)

    return force

def system_from_topology(TOPOLOGY,mass_dict):
    TOPOLOGY_ATOMS = list(TOPOLOGY.atoms())
    natom = TOPOLOGY.getNumAtoms()

    system = System()
    system.setDefaultPeriodicBoxVectors(*TOPOLOGY.getPeriodicBoxVectors())
    for atom_ix in range(natom):
        atom_residue = TOPOLOGY_ATOMS[atom_ix].residue
        system.addParticle(mass_dict[atom_residue.name])
        
    cmm_remover = CMMotionRemover()
    system.addForce(cmm_remover)
    return system

def add_polymer_bonds(system,molecules):
    for residue in list(molecules):
        residue_atoms = residue 
        pairs = [(residue_atoms[i], residue_atoms[i+1]) for i in range(len(residue_atoms) - 1)]
        for pair in pairs:
            system.addConstraint(pair[0], pair[1], 5.0 * angstrom)
    return system

def exclude_12_13_14(force,TOPOLOGY):
    for residue in list(TOPOLOGY.residues()):
        if residue.name not in ['AAA','BBB']:
            continue
        residue_atoms = [atom.index for atom in residue.atoms()]
        exclusion_pairs = [(residue_atoms[i], residue_atoms[i+1]) for i in range(len(residue_atoms) - 1)] + \
                          [(residue_atoms[i], residue_atoms[i+2]) for i in range(len(residue_atoms) - 2)] + \
                          [(residue_atoms[i], residue_atoms[i+3]) for i in range(len(residue_atoms) - 3)]
        for pair in exclusion_pairs:
            force.addExclusion(pair[0], pair[1])
    return force

def molecule_count(V_nm2):
    avogadro_number = AVOGADRO_CONSTANT_NA._value
    molar_mass = 18.01528
    density_g_per_cm3 = 1.0
    molecules_per_nm3 = (density_g_per_cm3 / molar_mass) * avogadro_number / 1e21
    return int(np.ceil(V_nm2 * molecules_per_nm3))

def run_21step(simulation,barostat,integrator,NVT_FRQ=100000000,BAROSTAT_FRQ=25,TARGET_T=300,HEAT_T=2000, Pmax=200):
    TIMESTEP = simulation.integrator.getStepSize().value_in_unit(picosecond)

    #Step 1: NVT, High Temperature
    barostat.setFrequency(NVT_FRQ)
    integrator.setTemperature(HEAT_T*kelvin)
    simulation.step(int(0.5*1000/TIMESTEP))

    #Step 2: NVT, Target Temperature
    integrator.setTemperature(TARGET_T*kelvin)
    simulation.step(int(0.25*1000/TIMESTEP))

    #Step 3: NPT, 0.02*Pmax, Target Temperature
    simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.02*Pmax*bar)
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
    integrator.setTemperature(TARGET_T*kelvin)
    barostat.setFrequency(BAROSTAT_FRQ)
    simulation.step(int(0.5*1000/TIMESTEP))

    #Step 4: NVT High Temperature
    barostat.setFrequency(NVT_FRQ)
    integrator.setTemperature(HEAT_T*kelvin)
    simulation.step(int(0.25*1000/TIMESTEP))

    #Step 5: NVT, Target Temperature
    integrator.setTemperature(TARGET_T*kelvin)
    simulation.step(int(0.5*1000/TIMESTEP))

    #Step 6: NPT, 0.6*Pmax, Target Temperature
    simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.6*Pmax*bar)
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
    integrator.setTemperature(TARGET_T*kelvin)
    barostat.setFrequency(BAROSTAT_FRQ)
    simulation.step(int(0.25*1000/TIMESTEP))

    #Step 7: NVT High Temperature
    barostat.setFrequency(NVT_FRQ)
    integrator.setTemperature(HEAT_T*kelvin)
    simulation.step(int(0.5*1000/TIMESTEP))

    #Step 8: NVT, Target Temperature
    integrator.setTemperature(TARGET_T*kelvin)
    simulation.step(int(0.5*1000/TIMESTEP))

    #Step 9: NPT, Pmax, Target Temperature
    simulation.context.setParameter(MonteCarloBarostat.Pressure(), Pmax*bar)
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
    integrator.setTemperature(TARGET_T*kelvin)
    barostat.setFrequency(BAROSTAT_FRQ)
    simulation.step(int(0.5*1000/TIMESTEP))

    #Step 10: NVT High Temperature
    barostat.setFrequency(NVT_FRQ)
    integrator.setTemperature(HEAT_T*kelvin)
    simulation.step(int(0.25*1000/TIMESTEP))

    #Step 11: NVT, Target Temperature
    integrator.setTemperature(TARGET_T*kelvin)
    simulation.step(int(0.5*1000/TIMESTEP))

    #Step 12: NPT, 0.5*Pmax, Target Temperature
    simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.5*Pmax*bar)
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
    integrator.setTemperature(TARGET_T*kelvin)
    barostat.setFrequency(BAROSTAT_FRQ)
    simulation.step(int(0.05*1000/TIMESTEP))

    #Step 13: NVT High Temperature
    barostat.setFrequency(NVT_FRQ)
    integrator.setTemperature(HEAT_T*kelvin)
    simulation.step(int(0.025*1000/TIMESTEP))

    #Step 14: NVT, Target Temperature
    integrator.setTemperature(TARGET_T*kelvin)
    simulation.step(int(0.05*1000/TIMESTEP))

    #Step 15: NPT, 0.1*Pmax, Target Temperature
    simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.1*Pmax*bar)
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
    integrator.setTemperature(TARGET_T*kelvin)
    barostat.setFrequency(BAROSTAT_FRQ)
    simulation.step(int(0.025*1000/TIMESTEP))

    #Step 16: NVT High Temperature
    barostat.setFrequency(NVT_FRQ)
    integrator.setTemperature(HEAT_T*kelvin)
    simulation.step(int(0.025*1000/TIMESTEP))

    #Step 17: NVT, Target Temperature
    integrator.setTemperature(TARGET_T*kelvin)
    simulation.step(int(0.05*1000/TIMESTEP))

    #Step 18: NPT, 0.01*Pmax, Target Temperature
    simulation.context.setParameter(MonteCarloBarostat.Pressure(), 0.01*Pmax*bar)
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
    integrator.setTemperature(TARGET_T*kelvin)
    barostat.setFrequency(BAROSTAT_FRQ)
    simulation.step(int(0.025*1000/TIMESTEP))

    #Step 19: NVT High Temperature
    barostat.setFrequency(NVT_FRQ)
    integrator.setTemperature(HEAT_T*kelvin)
    simulation.step(int(0.025*1000/TIMESTEP))

    #Step 20: NVT, Target Temperature
    integrator.setTemperature(TARGET_T*kelvin)
    simulation.step(int(0.05*1000/TIMESTEP))

    #Step 21: NPT, 1 bar, Target Temperature
    simulation.context.setParameter(MonteCarloBarostat.Pressure(), 1*bar)
    simulation.context.setParameter(MonteCarloBarostat.Temperature(), TARGET_T*kelvin)
    integrator.setTemperature(TARGET_T*kelvin)
    barostat.setFrequency(BAROSTAT_FRQ)

    Restart.save_simulation('20step.save',simulation,'classical')

    # simulation.step(int(10*1000/TIMESTEP))

    # Restart.save_simulation('21step.save',simulation,'classical')
    
    return simulation
