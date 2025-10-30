import numpy as np
from md_lj.utils import initialize_positions, initialize_velocities, apply_periodic_boundaries
from md_lj.forces import force_energy
from md_lj.integrator import velocity_verlet
from md_lj.analysis import calculate_temperature, calculate_pressure, radial_distribution, plot_results

def run_simulation(params):
    """
    Runs a single molecular dynamics simulation.
    """
    # Initialization
    positions = initialize_positions(params['n_particles'], params['box_length'])
    velocities = initialize_velocities(params['n_particles'], params['initial_temp'])
    forces, _, _ = force_energy(positions, params['box_length'])

    # Data storage
    data = {
        'time': [],
        'potential_energy': [],
        'kinetic_energy': [],
        'total_energy': [],
        'temperature': [],
        'pressure': [],
        'positions': [],
        'velocities': [],
        'g_r': [],
        'r_bins': None
    }

    # Main loop
    for step in range(params['total_steps']):
        positions, velocities, forces, potential_energy, virial = velocity_verlet(
            positions, velocities, forces, params['dt'], params['box_length']
        )
        positions = apply_periodic_boundaries(positions, params['box_length'])

        # Sampling
        if step % params['sampling_interval'] == 0 and step >= params['equilibration_steps']:
            kinetic_energy = 0.5 * np.sum(velocities**2)
            temperature = calculate_temperature(velocities)
            pressure = calculate_pressure(temperature, params['n_particles'], params['box_length']**3, virial)
            
            data['time'].append(step * params['dt'])
            data['potential_energy'].append(potential_energy)
            data['kinetic_energy'].append(kinetic_energy)
            data['total_energy'].append(potential_energy + kinetic_energy)
            data['temperature'].append(temperature)
            data['pressure'].append(pressure)
            
            data['positions'].append(positions.copy())
            data['velocities'].append(velocities.copy())
            
            r_bins, g_r = radial_distribution(positions, params['box_length'])
            if data['r_bins'] is None:
                data['r_bins'] = r_bins
            data['g_r'].append(g_r)
    
    return data

def main():
    """
    Main function to run multiple simulations and analyze the results.
    """
    params = {
        'n_particles': 27,
        'box_length': 4.0, # Adjusted for a reasonable density
        'initial_temp': 1.0,
        'dt': 0.002,
        'total_steps': 600,
        'sampling_interval': 10,
        'equilibration_steps': 100 
    }
    
    n_simulations = 50
    all_data = {
        'time': [],
        'potential_energy': [],
        'kinetic_energy': [],
        'total_energy': [],
        'temperature': [],
        'pressure': [],
        'positions': [],
        'velocities': [],
        'g_r': [],
        'r_bins': None
    }
    
    for i in range(n_simulations):
        print(f"Running simulation {i+1}/{n_simulations}")
        sim_data = run_simulation(params)
        for key in sim_data:
            if key != 'r_bins':
                all_data[key].extend(sim_data[key])
            else:
                all_data[key] = sim_data[key]
    
    # Numerical averages
    avg_temp = np.mean(all_data['temperature'])
    avg_pressure = np.mean(all_data['pressure'])
    avg_total_energy = np.mean(all_data['total_energy'])
    
    print(f"\nAverage Temperature: {avg_temp:.4f}")
    print(f"Average Pressure: {avg_pressure:.4f}")
    print(f"Average Total Energy: {avg_total_energy:.4f}")

    # Plotting
    plot_results(all_data)

if __name__ == '__main__':
    main()