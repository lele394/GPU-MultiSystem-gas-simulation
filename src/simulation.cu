#include <cstdio> // For printf in the sanity check
#include "simulation.cuh"
#include "particle_system.h"
#include "interactions.h"
#include "integrators.h"
#include "boundaries.h"
#include "simulation_defs.h" // fancy umbrellas

// =========================================================================
//                             KER NEL S
// =========================================================================

// --- KERNEL 1: For Calculating Initial Forces [Required for Leapfrog] ---
template <typename T, typename InteractionModel>
__global__ void initial_force_kernel(
    Particle<T>* all_particles_global,
    int particles_per_system,
    InteractionModel interaction) 
{
    const int system_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ Particle<T> system_particles_shared[];

    // Cooperatively load all particles
    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        system_particles_shared[p_idx] = all_particles_global[system_id * particles_per_system + p_idx];
    }
    __syncthreads();

    // Each thread calculates the initial force for its assigned particles
    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        Particle<T> p = system_particles_shared[p_idx];
        Vec2<T> initial_accel = interaction.calculate_acceleration(p, p_idx, system_particles_shared, particles_per_system);
        all_particles_global[system_id * particles_per_system + p_idx].acceleration = initial_accel;
    }
}

// --- KERNEL 2: For ALL Single-Step Integrators (e.g., Euler) ---
template <typename T, typename IntegratorModel, typename InteractionModel>
__global__ void single_step_simulation_kernel(
    Particle<T>* all_particles_global,
    int particles_per_system,
    Vec2<T> box_min, Vec2<T> box_max, T dt,
    int steps_to_run,
    InteractionModel interaction)
{
    const int system_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ Particle<T> shared_mem[];
    Particle<T>* read_buffer = &shared_mem[0];
    Particle<T>* write_buffer = &shared_mem[particles_per_system];

    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        read_buffer[p_idx] = all_particles_global[system_id * particles_per_system + p_idx];
    }
    __syncthreads();

    IntegratorModel integrator;   // Instantiate the chosen integrator

    for (int step = 0; step < steps_to_run; ++step) {
        for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
            Particle<T> p = read_buffer[p_idx];
            
            // This single line now encapsulates all the logic!
            integrator.integrate(p, p_idx, read_buffer, particles_per_system, interaction, dt);
            
            apply_mirror_boundaries(p, box_min, box_max);
            write_buffer[p_idx] = p;
        }
        __syncthreads();
        Particle<T>* temp = read_buffer; read_buffer = write_buffer; write_buffer = temp;
    }

    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        all_particles_global[system_id * particles_per_system + p_idx] = read_buffer[p_idx];
    }
}

// --- KERNEL 3: Specifically for the Leapfrog Integrator ---
template <typename T, typename InteractionModel>
__global__ void leapfrog_simulation_kernel(
    Particle<T>* all_particles_global,
    int particles_per_system,
    Vec2<T> box_min, Vec2<T> box_max, T dt,
    int steps_to_run,
    InteractionModel interaction)
{
    const int system_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ Particle<T> system_particles[];
    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        system_particles[p_idx] = all_particles_global[system_id * particles_per_system + p_idx];
    }
    __syncthreads();
    
    LeapfrogIntegrator<T> integrator;

    for (int step = 0; step < steps_to_run; ++step) {
        // Part 1
        for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
            integrator.pre_force_update(system_particles[p_idx], dt);
        }
        __syncthreads();

        // Part 2
        for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
            Particle<T> p = system_particles[p_idx];
            Vec2<T> new_accel = interaction.calculate_acceleration(p, p_idx, system_particles, particles_per_system);
            integrator.post_force_update(system_particles[p_idx], new_accel, dt);
        }
        __syncthreads();
        
        // Boundaries
        for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
            apply_mirror_boundaries(system_particles[p_idx], box_min, box_max);
        }
        __syncthreads();
    }

    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        all_particles_global[system_id * particles_per_system + p_idx] = system_particles[p_idx];
    }
}


// =========================================================================
//            The Single "Smart" Launcher & Initial Force Launcher
// =========================================================================

// Launcher for the one-off initial force calculation
template <typename T>
void run_initial_force_calculation(
    Particle<T>* d_particles, int num_systems, int particles_per_system, InteractionType interaction_type, cudaStream_t stream)
{
    const int threads_per_block = 256;
    dim3 blocks_per_grid(num_systems);
    dim3 threads_per_block_dim(threads_per_block);
    size_t shared_mem_size = particles_per_system * sizeof(Particle<T>);

    if (interaction_type == InteractionType::Gravity) {
        Gravity<T> interaction(1.0f, 0.05f); 
        initial_force_kernel<T, Gravity<T>>
            <<<blocks_per_grid, threads_per_block_dim, shared_mem_size, stream>>>
            (d_particles, particles_per_system, interaction);
    } 
    else if (interaction_type == InteractionType::RepulsiveForce) {
        RepulsiveForce<T> interaction(1.0f, 0.1f);
        initial_force_kernel<T, RepulsiveForce<T>>
            <<<blocks_per_grid, threads_per_block_dim, shared_mem_size, stream>>>
            (d_particles, particles_per_system, interaction);
    }
    // Add other interactions like LennardJones if you want to use them with Leapfrog
}

// The main "Smart" Launcher
template <typename T>
void run_simulation_steps(
    Particle<T>* d_particles, int total_particles, int num_systems, int particles_per_system,
    IntegratorType integrator_type,
    InteractionType interaction_type,
    Vec2<T> box_min, Vec2<T> box_max, T dt, int steps_to_run, cudaStream_t stream)
{
    const int threads_per_block = 256;
    dim3 blocks_per_grid(num_systems);
    dim3 threads_per_block_dim(threads_per_block);

    // --- Path for Euler and other single-step integrators ---
    if (integrator_type == IntegratorType::Euler) {
        size_t shared_mem_size = 2 * particles_per_system * sizeof(Particle<T>);
        
        if (interaction_type == InteractionType::Gravity) {
            Gravity<T> interaction(1.0f, 0.05f);
            single_step_simulation_kernel<T, EulerIntegrator<T>, Gravity<T>>
                <<<blocks_per_grid, threads_per_block_dim, shared_mem_size, stream>>>
                (d_particles, particles_per_system, box_min, box_max, dt, steps_to_run, interaction);
        } else if (interaction_type == InteractionType::RepulsiveForce) {
            RepulsiveForce<T> interaction(1.0f, 0.1f);
            single_step_simulation_kernel<T, EulerIntegrator<T>, RepulsiveForce<T>>
                <<<blocks_per_grid, threads_per_block_dim, shared_mem_size, stream>>>
                (d_particles, particles_per_system, box_min, box_max, dt, steps_to_run, interaction);
        }

    // --- Path for the Leapfrog integrator ---
    } else if (integrator_type == IntegratorType::Leapfrog) {
        size_t shared_mem_size = particles_per_system * sizeof(Particle<T>);
        
        if (interaction_type == InteractionType::Gravity) {
            Gravity<T> interaction(1.0f, 0.05f);
            leapfrog_simulation_kernel<T, Gravity<T>>
                <<<blocks_per_grid, threads_per_block_dim, shared_mem_size, stream>>>
                (d_particles, particles_per_system, box_min, box_max, dt, steps_to_run, interaction);
        } else if (interaction_type == InteractionType::RepulsiveForce) {
            RepulsiveForce<T> interaction(1.0f, 0.1f);
            leapfrog_simulation_kernel<T, RepulsiveForce<T>>
                <<<blocks_per_grid, threads_per_block_dim, shared_mem_size, stream>>>
                (d_particles, particles_per_system, box_min, box_max, dt, steps_to_run, interaction);
        }
    }
}

// =========================================================================
//                   Explicit Template Instantiations
// =========================================================================
template void run_initial_force_calculation<float>(Particle<float>*, int, int, InteractionType, cudaStream_t);
template void run_simulation_steps<float>(Particle<float>*, int, int, int, IntegratorType, InteractionType, Vec2<float>, Vec2<float>, float, int, cudaStream_t);