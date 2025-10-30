import numpy as np
from md_lj.utils import minimum_image_distance

def force_energy(positions, box_length, r_cutoff=2.5, epsilon=1.0, sigma=1.0):
    """
    Calculates forces, potential energy, and virial contribution using a shifted Lennard-Jones potential.
    """
    n_particles = len(positions)
    forces = np.zeros((n_particles, 3))
    potential_energy = 0.0
    virial = 0.0
    
    r_cutoff_sq = r_cutoff**2
    
    # Potential shift
    sr_inv_6_c = (sigma / r_cutoff)**6
    sr_inv_12_c = sr_inv_6_c**2
    u_cutoff = 4.0 * epsilon * (sr_inv_12_c - sr_inv_6_c)

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            delta_r = positions[i] - positions[j]
            delta_r = minimum_image_distance(delta_r, box_length)
            
            r_sq = np.sum(delta_r**2)

            if r_sq < r_cutoff_sq:
                r_inv_2 = 1.0 / r_sq
                r_inv_6 = r_inv_2**3
                
                term1 = (sigma**2 * r_inv_2)**3
                term2 = term1**2

                # Lennard-Jones potential and force
                potential_energy += 4.0 * epsilon * (term2 - term1) - u_cutoff
                force_magnitude = 24.0 * epsilon * r_inv_2 * (2.0 * term2 - term1)
                
                force_vector = force_magnitude * delta_r
                forces[i] += force_vector
                forces[j] -= force_vector
                
                virial += np.dot(force_vector, delta_r)

    return forces, potential_energy, virial