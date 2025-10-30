import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, maxwell

def calculate_temperature(velocities, mass=1.0, k_B=1.0):
    """
    Calculates the instantaneous temperature.
    """
    n_particles = len(velocities)
    kinetic_energy = 0.5 * mass * np.sum(velocities**2)
    return (2.0 / (3.0 * n_particles * k_B)) * kinetic_energy

def calculate_pressure(temperature, n_particles, box_volume, virial, k_B=1.0):
    """
    Calculates the pressure using the virial theorem.
    """
    return (n_particles * k_B * temperature / box_volume) + (virial / (3.0 * box_volume))

def radial_distribution(positions, box_length, n_bins=100, r_max=None):
    """
    Calculates the radial distribution function g(r).
    """
    if r_max is None:
        r_max = box_length / 2.0
    
    n_particles = len(positions)
    box_volume = box_length**3
    particle_density = n_particles / box_volume
    
    bin_width = r_max / n_bins
    bins = np.linspace(0, r_max, n_bins + 1)
    bin_centers = bins[:-1] + bin_width / 2.0
    
    g_r = np.zeros(n_bins)
    
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            delta_r = positions[i] - positions[j]
            delta_r -= box_length * np.round(delta_r / box_length)
            r = np.sqrt(np.sum(delta_r**2))
            
            if r < r_max:
                bin_index = int(r / bin_width)
                g_r[bin_index] += 2

    # Normalize
    shell_volumes = 4.0 * np.pi * bin_centers**2 * bin_width
    ideal_gas_counts = particle_density * shell_volumes
    
    g_r /= (n_particles * ideal_gas_counts)
    
    return bin_centers, g_r

def plot_results(data):
    """
    Generates all the required plots from the simulation data.
    """
    # Total Energy vs. Time
    plt.figure()
    plt.plot(data['time'], data['total_energy'])
    plt.xlabel('Time (reduced units)')
    plt.ylabel('Total Energy (reduced units)')
    plt.title('Total Energy Conservation')
    plt.grid(True)
    plt.show()

    # Velocity Component Distribution
    plt.figure()
    vx = np.array(data['velocities'])[:, :, 0].flatten()
    plt.hist(vx, bins=50, density=True, label='Simulation Vx')
    
    mean_temp = np.mean(data['temperature'])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, 0, np.sqrt(mean_temp))
    plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
    plt.xlabel('Velocity Component (reduced units)')
    plt.ylabel('Probability Density')
    plt.title('Velocity Component Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Speed Distribution
    plt.figure()
    speeds = np.linalg.norm(np.array(data['velocities']).reshape(-1, 3), axis=1)
    plt.hist(speeds, bins=50, density=True, label='Simulation Speeds')
    
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = maxwell.pdf(x, scale=np.sqrt(mean_temp))
    plt.plot(x, p, 'k', linewidth=2, label='Maxwell-Boltzmann')
    plt.xlabel('Speed (reduced units)')
    plt.ylabel('Probability Density')
    plt.title('Speed Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Radial Distribution Function
    plt.figure()
    g_r_avg = np.mean(data['g_r'], axis=0)
    plt.plot(data['r_bins'], g_r_avg)
    plt.xlabel('r (reduced units)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.grid(True)
    plt.show()