#include <cstdio> // For printf in the sanity check
#include "simulation.cuh"
#include "particle_system.h"
#include "interactions.h"
#include "integrators.h"
#include "boundaries.h"
#include "simulation_defs.h" // fancy umbrellas

#include "settings.h"


// I hate that file
//  I hate that file so much
//   Look at those kernel launches
//    It's art 
//     Beautiful but horrible art

// Shouldv've named this hell.cu


// I have a smart launcher now
// So I can hide all that code away here

// Split kernels for Leapfrog and other integrators
// Because Leapfrog needs that initial force calculation
//  and has that weird two-part update

// I have no idea if this is efficient or not
// But it works and is way cleaner than before
// And supposedly should be easier to add RK4(5) later on
//   ^ this might be a lie



// =========================================================================
//                             KER NEL S
// =========================================================================

// --- KERNEL 1: For Calculating Initial Forces [Required for Leapfrog] ---
// That's leapfrog precomputed initial acceleration
template <typename T, typename InteractionModel>
__global__ void initial_force_kernel(
    Particle<T>* all_particles_global,
    int particles_per_system,
    InteractionModel interaction) 
{

    // Identify system and thread
    const int system_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    extern __shared__ Particle<T> system_particles_shared[]; // Shared memory for all particles in the system
                                                             // Secret sauce for L1 caching?

    // Cooperatively load all particles
    // This should do L1 caching too hopefully (Please I beg)
    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        system_particles_shared[p_idx] = all_particles_global[system_id * particles_per_system + p_idx];
    }
    __syncthreads();

    // calculates the initial force for its assigned particles
    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        Particle<T> p = system_particles_shared[p_idx];
        Vec2<T> initial_accel = interaction.calculate_acceleration(p, p_idx, system_particles_shared, particles_per_system);
        all_particles_global[system_id * particles_per_system + p_idx].acceleration = initial_accel;
    }
}

// --- KERNEL 2: For ALL Single-Step Integrators (e.g., Euler) ---
// Concerns : integrator recreation every nstep
// There's so much destruction and creation going on here it's insane
//  Might be an issue? (definitly)
template <typename T, typename IntegratorModel, typename InteractionModel>
__global__ void single_step_simulation_kernel(
    Particle<T>* all_particles_global,
    int particles_per_system,
    Vec2<T> box_min, Vec2<T> box_max, T dt,
    int steps_to_run,
    InteractionModel interaction)
{

    // Cuda stuff like before
    const int system_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    // buffers
    extern __shared__ Particle<T> shared_mem[];
    Particle<T>* read_buffer = &shared_mem[0];
    Particle<T>* write_buffer = &shared_mem[particles_per_system];

    // L1 cache magic again 
    for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
        read_buffer[p_idx] = all_particles_global[system_id * particles_per_system + p_idx];
    }
    __syncthreads();

    IntegratorModel integrator;   // Instantiate the integrator (again)

    // steps
    for (int step = 0; step < steps_to_run; ++step) {
        for (int p_idx = thread_id; p_idx < particles_per_system; p_idx += num_threads) {
            Particle<T> p = read_buffer[p_idx];
            
            // integrator magic
            integrator.integrate(p, p_idx, read_buffer, particles_per_system, interaction, dt);
            
            apply_mirror_boundaries(p, box_min, box_max); // bouncy stuff
            write_buffer[p_idx] = p;
        }
        __syncthreads();
        Particle<T>* temp = read_buffer; read_buffer = write_buffer; write_buffer = temp; //ping pong
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
//            The "Smart" Launcher & Initial Force Launcher 
// =========================================================================
// I wanna refactor with switch case later probably


// Launcher for the one-off initial force calculation
// Needed for Leapfrog integrator
template <typename T>
void run_initial_force_calculation(
    Particle<T>* d_particles, int num_systems, int particles_per_system, InteractionType interaction_type, cudaStream_t stream, Settings settings)
{   // Should I pass the interaction model directly? avoids the switch case, case avoids destroying it later
    // const int threads_per_block = 256;
    dim3 blocks_per_grid(num_systems);
    dim3 threads_per_block_dim(settings.threads_per_block);
    size_t shared_mem_size = particles_per_system * sizeof(Particle<T>);

    // Interaction switch. ACTUALLY a switch case would be cleaner here
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
    // Add other interactions like LennardJones here if new ones are impleemented
    // Note : I have hard coded defaults here. Might not be good.
}

// The main "Smart" Launcher
template <typename T>
void run_simulation_steps(
    Particle<T>* d_particles, int total_particles, int num_systems, int particles_per_system,
    IntegratorType integrator_type,
    InteractionType interaction_type,
    Vec2<T> box_min, Vec2<T> box_max, T dt, int steps_to_run, cudaStream_t stream, Settings settings)
{   // Pass interaction model directly? Avoid destroying and recreating it every time.
    // I don't like it but it works for now
    // Perfs improvement possible later
    // Also enables creation of the interaction in main, can change currently hardcoded params

    // Not sure I can pass the integrator directly, however might be possible to 
    // bundle the call in the integrator struct itself? Needs thought
    // ~~I really don't like that if-else/switch-case mess though~~
    // ~~Too heavy to read~~
    // I did the switch case

    // Need to check cuz the integrator kernel depends on the interaction type
    // I smell perfs issues here too

    // I really wanna nuke that switch case to oblivion later
    // unneeded added complexity
    // Yeah but how tho


    // const int threads_per_block = 256;
    dim3 blocks_per_grid(num_systems);
    dim3 threads_per_block_dim(settings.threads_per_block);

    // --- Path for Euler and other single-step integrators ---
    // USE SWITCH CASE???? ANYONE??????

    // --- 2. Calculate Required Shared Memory and Challenge Sizes ---
    int max_shmem_bytes;
    cudaDeviceGetAttribute(&max_shmem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    size_t requested_shmem_size = 0;
    int max_particles_for_integrator = 0;
    const size_t particle_size = sizeof(Particle<T>);

    // Determine requested memory and max possible particles based on integrator choice
    switch (integrator_type) {
        case IntegratorType::Euler:
            requested_shmem_size = 2 * particles_per_system * particle_size;
            max_particles_for_integrator = max_shmem_bytes / (2 * particle_size);
            break;
        case IntegratorType::Leapfrog:
            requested_shmem_size = particles_per_system * particle_size;
            max_particles_for_integrator = max_shmem_bytes / particle_size;
            break;
        default:
            printf("FATAL ERROR: Unknown IntegratorType specified in launcher.\n");
            return;
    }

    // With quick if toogle, technical printout for debugging
    // I'll keep it like that, if you're here that means you're digging into kernel launch
    
    if(settings.KernelLogsEnabled)
    {
        printf("\n=== KERNEL LOGS ===\n");
        printf("\n--- SHARED MEMORY CONFIGURATION ---\n");
        printf("Max available shared memory per SM : %d bytes (%.1f KB)\n", max_shmem_bytes, (float)max_shmem_bytes / 1024.0f);
        printf("Size of one Particle               : %zu bytes\n", particle_size);
        printf("Integrator selected                : %s\n", (integrator_type == IntegratorType::Leapfrog ? "Leapfrog" : "Euler"));
        printf(" -> Max possible particles/system   : %d\n", max_particles_for_integrator);
        printf(" -> Current particles/system        : %d\n", particles_per_system);
        printf(" -> Requested shared memory         : %zu bytes\n", requested_shmem_size);
        printf("-----------------------------------\n\n");

    }

    if (requested_shmem_size > max_shmem_bytes) {
        printf("FATAL ERROR: Request EXCEEDS device limit.\n");
        printf("-> Reduce particles_per_system to %d or less for this integrator.\n", max_particles_for_integrator);
        return;
    }

    // --- 3. Dispatch, with switches (finally) ---
    switch (integrator_type) {

        // ============================
        case IntegratorType::Euler:
            switch (interaction_type) {


                case InteractionType::Gravity: {
                    cudaFuncSetAttribute(single_step_simulation_kernel<T, EulerIntegrator<T>, Gravity<T>>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes);
                    Gravity<T> interaction(settings.G, settings.gravity_smoothing);
                    single_step_simulation_kernel<T, EulerIntegrator<T>, Gravity<T>>
                        <<<blocks_per_grid, threads_per_block_dim, requested_shmem_size, stream>>>
                        (d_particles, particles_per_system, box_min, box_max, dt, steps_to_run, interaction);
                    break;
                }
                
                
                
                case InteractionType::RepulsiveForce: {
                    cudaFuncSetAttribute(single_step_simulation_kernel<T, EulerIntegrator<T>, RepulsiveForce<T>>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes);
                    RepulsiveForce<T> interaction(settings.epsilon, settings.sigma);
                    single_step_simulation_kernel<T, EulerIntegrator<T>, RepulsiveForce<T>>
                        <<<blocks_per_grid, threads_per_block_dim, requested_shmem_size, stream>>>
                        (d_particles, particles_per_system, box_min, box_max, dt, steps_to_run, interaction);
                    break;
                }
                
                
                
                // Add other like IdealGas here...
                default:
                    printf("FATAL ERROR: Unknown InteractionType for Euler integrator.\n");
                    break;
            }
            break; // End of Euler case




        // ============================
        case IntegratorType::Leapfrog:
            switch (interaction_type) {



                case InteractionType::Gravity: {
                    cudaFuncSetAttribute(leapfrog_simulation_kernel<T, Gravity<T>>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes);
                    Gravity<T> interaction(settings.G, settings.gravity_smoothing);
                    leapfrog_simulation_kernel<T, Gravity<T>>
                        <<<blocks_per_grid, threads_per_block_dim, requested_shmem_size, stream>>>
                        (d_particles, particles_per_system, box_min, box_max, dt, steps_to_run, interaction);
                    break;
                }




                case InteractionType::RepulsiveForce: {
                    cudaFuncSetAttribute(leapfrog_simulation_kernel<T, RepulsiveForce<T>>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_bytes);
                    RepulsiveForce<T> interaction(settings.epsilon, settings.sigma);
                    leapfrog_simulation_kernel<T, RepulsiveForce<T>>
                        <<<blocks_per_grid, threads_per_block_dim, requested_shmem_size, stream>>>
                        (d_particles, particles_per_system, box_min, box_max, dt, steps_to_run, interaction);
                    break;
                }
                // Add other like IdealGas here...
                default:
                    printf("FATAL ERROR: Unknown InteractionType for Leapfrog integrator.\n");
                    break;
            }
            break; // End of Leapfrog case
    }
}

// =========================================================================
//                   Explicit Template Instantiations
// =========================================================================
// This is bad, I'm gonna have to change that when adding mixed precision
template void run_initial_force_calculation<float>(Particle<float>*, int, int, InteractionType, cudaStream_t, Settings);
template void run_simulation_steps<float>(Particle<float>*, int, int, int, IntegratorType, InteractionType, Vec2<float>, Vec2<float>, float, int, cudaStream_t, Settings);