#include "simulation.cuh"
#include "particle_system.h"
#include "interactions.h"
#include "integrators.h"
#include "boundaries.h"



// ===========================================================
//            Basic 1 System Kernel with Internal Loop
// ===========================================================

// CUDA kernel with an internal loop, avoid starting overhead each step
template <typename T>
__global__ void simulation_n_steps_kernel(Particle<T>* particles, int num_particles, 
                                          Vec2<T> box_min, Vec2<T> box_max, T dt, 
                                          int steps_to_run) { // New parameter
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int num_of_systems = 24;
    // int system_size = num_particles / num_of_systems;

    if (idx >= num_particles) {
        return;
    }

    // Load particle data from global memory just once at the beginning
    Particle<T> p = particles[idx];

    // THE MAIN LOOP IS NOW INSIDE THE KERNEL
    for (int i = 0; i < steps_to_run; ++i) {
        // --- Interaction Force Calculation ---
        IdealGas<T> interaction;
        Vec2<T> acceleration = interaction.calculate_acceleration(p);

        // --- Integration ---
        euler_integrator(p, acceleration, dt);

        // --- Boundary Conditions ---
        apply_mirror_boundaries(p, box_min, box_max);
    }

    // Write the final particle data back to global memory just once at the end
    particles[idx] = p;
}

// Host-callable function that launches the kernel
template <typename T>
void run_n_simulation_steps(Particle<T>* d_particles, int num_particles, 
                            Vec2<T> box_min, Vec2<T> box_max, T dt, 
                            int steps_to_run, cudaStream_t stream) {

    int threads_per_block = 256; // Hardcoded, cores/SM * 2 => seems to be ok since we're aiming for 4-6k particles
    int blocks_per_grid = 24; // 24 because we run 24 different systems, one per SM. Old 1system computation :  (num_particles + threads_per_block - 1) / threads_per_block;

    simulation_n_steps_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_particles, num_particles, box_min, box_max, dt, steps_to_run
    );
}

// Explicit template instantiation for float. This is required because this function
// is defined in a .cu file and called from a .cpp file.
template void run_n_simulation_steps<float>(Particle<float>*, int, Vec2<float>, Vec2<float>, float, int, cudaStream_t);











// ===========================================================
//       New Kernel for Multiple Systems with LJ Interaction    
// ===========================================================
template <typename T>
__global__ void multi_system_lj_kernel(
    Particle<T>* all_particles_global,
    int particles_per_system,
    Vec2<T> box_min, Vec2<T> box_max, T dt,
    int steps_to_run) 
{
    const int system_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    // --- 1. Set up Shared Memory Double Buffers ---
    // We allocate enough shared memory for TWO copies of our system.
    extern __shared__ Particle<T> shared_mem[];
    Particle<T>* read_buffer = &shared_mem[0];
    Particle<T>* write_buffer = &shared_mem[particles_per_system];

    // --- 2. Cooperative Loading (Grid-Stride Loop) ---
    // The team of 'num_threads' (256) cooperatively loads ALL 'particles_per_system' (e.g., 4000).
    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        int global_idx = system_id * particles_per_system + p_idx;
        read_buffer[p_idx] = all_particles_global[global_idx];
    }
    __syncthreads();

    // --- 3. Main Simulation Loop ---
    LennardJones<T> interaction(0.1f, 0.1f, 9.0f);

    for (int step = 0; step < steps_to_run; ++step) {
        // --- A. Process Particles (Grid-Stride Loop) ---
        for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
            // Get particle from the consistent READ buffer.
            Particle<T> p = read_buffer[p_idx];

            // Calculate acceleration by reading ALL particles from the READ buffer.
            // This guarantees no race conditions.
            Vec2<T> acceleration = interaction.calculate_acceleration(p, p_idx, read_buffer, particles_per_system);

            euler_integrator(p, acceleration, dt);
            apply_mirror_boundaries(p, box_min, box_max);

            // Write the new state to the separate WRITE buffer.
            write_buffer[p_idx] = p;
        }

        // --- B. Synchronize ---
        // Wait for all threads to finish writing for this time step.
        __syncthreads();

        // --- C. Swap Buffers for Next Iteration ---
        Particle<T>* temp = read_buffer;
        read_buffer = write_buffer;
        write_buffer = temp;
    }

    // --- 4. Cooperative Writing (Grid-Stride Loop) ---
    // The final, correct data is in the `read_buffer` (due to the last swap).
    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        int global_idx = system_id * particles_per_system + p_idx;
        all_particles_global[global_idx] = read_buffer[p_idx];
    }
}

// =========================================================================
//                The Correct "Smart" Launcher Function
// =========================================================================
template <typename T>
void run_lj_simulation_steps(
    Particle<T>* d_particles, int total_particles, int num_systems, int particles_per_system,
    Vec2<T> box_min, Vec2<T> box_max, T dt, int steps_to_run, cudaStream_t stream)
{
    // 1. Define the optimal hardware configuration here, hidden from main.cpp
    const int threads_per_block = 256;

    // 2. Calculate shared memory requirement for the double-buffer strategy
    size_t shared_mem_size = 2 * particles_per_system * sizeof(Particle<T>);

    // 3. Sanity Check against hardware limits
    int max_shared_mem_per_block;
    cudaDeviceGetAttribute(&max_shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (shared_mem_size > max_shared_mem_per_block) {
        printf("FATAL ERROR: Requested shared memory size (%zu bytes) for %d particles exceeds the device limit of %d bytes.\n",
               shared_mem_size, particles_per_system, max_shared_mem_per_block);
        printf("-> Reduce particles_per_system.\n");
        return;
    }

    // 4. Launch the kernel
    dim3 blocks_per_grid(num_systems);
    dim3 threads_per_block_dim(threads_per_block);
    
    multi_system_lj_kernel<<<blocks_per_grid, threads_per_block_dim, shared_mem_size, stream>>>(
        d_particles, particles_per_system,
        box_min, box_max, dt, steps_to_run
    );
}

// Ensure the explicit template instantiation matches the simplified signature
template void run_lj_simulation_steps<float>(Particle<float>*, int, int, int,
    Vec2<float>, Vec2<float>, float, int, cudaStream_t);