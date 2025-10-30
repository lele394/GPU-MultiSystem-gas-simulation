from md_lj.forces import force_energy

def velocity_verlet(positions, velocities, forces, dt, box_length, mass=1.0):
    """
    Performs one step of the velocity Verlet algorithm.
    """
    # Update positions
    positions += velocities * dt + 0.5 * forces / mass * dt**2

    # Store old forces
    forces_old = forces.copy()

    # Calculate new forces
    forces, potential_energy, virial = force_energy(positions, box_length)

    # Update velocities
    velocities += 0.5 * (forces + forces_old) / mass * dt

    return positions, velocities, forces, potential_energy, virial