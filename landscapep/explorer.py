def exploration_1(pdb_filename, temperature, steps_segment, n_segments):

    from openmm import app
    import openmm as mm
    from openmm import unit as u
    import sys
    import mdtraj as md
    import numpy as np
    from tqdm import tqdm

    pdb = PDBFile(pdb_filename)
    forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,  # Sin tratamiento de fronteras periódicas
        constraints=HBonds         # Restringe enlaces de hidrógeno para un paso de integración más largo
    )

    # Configurar el integrador (Langevin dynamics)
    temperature = 300 * u.kelvin
    friction = 1 / u.picosecond
    timestep = 2 * u.femtoseconds
    integrator = LangevinIntegrator(temperature, friction, timestep)

    # Crear una plataforma para ejecutar la simulación (e.g., CUDA si tienes GPU)
    platform = Platform.getPlatformByName("CUDA")  # Usa "CPU" si no tienes GPU

    # Configurar la simulación
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    # Minimización de energía
    # print("Minimizando energía...")
    simulation.minimizeEnergy()

    md_topology = md.Topology.from_openmm(simulation.topology)

    traj_inh_db = []
    unique_db = []

    for ii in tqdm(range(n_segments)):
        simulation.step(steps_segment)
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        simulation.minimizeEnergy()
        min_state = simulation.context.getState(getEnergy=True, getPositions=True)
        min_energy = min_state.getPotentialEnergy()
        min_positions = min_state.getPositions(asNumpy=True)
        traj = md.Trajectory(min_positions / u.nanometer, md_topology)
        phis = md.compute_phi(traj)[1]
        psis = md.compute_psi(traj)[1]
        dihed_angs=np.concatenate((phis[0],psis[0]))
        visitado = False
        for unique_index in range(len(unique_db)):
            aux = es_el_mismo(dihed_angs, unique_db[unique_index])
            if aux == True:
                visitado = True
                traj_inh_db.append(unique_index)
                break
        if visitado == False:
            traj_inh_db.append(len(unique_db))
            unique_db.append(dihed_angs)

    return unique_db, traj_inh_db

def _es_el_mismo(angs1, angs2):
    if np.max(np.abs(angs1-angs2))<0.1:
        return True
    else:
        return False

