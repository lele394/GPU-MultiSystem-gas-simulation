import numpy as np

def initialize_positions(n_particles, box_length):
    """
    Places particles on a simple cubic lattice.
    """
    n_dim = int(round(n_particles**(1/3)))
    if n_dim**3 != n_particles:
        raise ValueError("Number of particles must be a perfect cube for lattice initialization.")
    
    spacing = box_length / n_dim
    positions = np.zeros((n_particles, 3))
    idx = 0
    for i in range(n_dim):
        for j in range(n_dim):
            for k in range(n_dim):
                positions[idx] = np.array([i, j, k]) * spacing
                idx += 1
    return positions

def initialize_velocities(n_particles, temp_target, mass=1.0, k_B=1.0):
    """
    Initializes velocities from a Maxwell-Boltzmann distribution.
    """
    # Draw from a normal distribution
    velocities = np.random.normal(0.0, np.sqrt(k_B * temp_target / mass), (n_particles, 3))

    # Remove center-of-mass velocity
    velocities -= np.mean(velocities, axis=0)

    # Rescale to the target temperature
    current_temp = np.sum(mass * velocities**2) / (3 * n_particles * k_B)
    scaling_factor = np.sqrt(temp_target / current_temp)
    velocities *= scaling_factor
    
    return velocities

def apply_periodic_boundaries(positions, box_length):
    """
    Applies periodic boundary conditions.
    """
    return positions % box_length

def minimum_image_distance(delta, box_length):
    """
    Computes the minimum image distance.
    """
    return delta - box_length * np.round(delta / box_length)