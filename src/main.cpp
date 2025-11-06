#include <iostream>
#include <vector>
#include <random>

#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>   

#include "simulation.cuh"
#include "particle_system.h"

int main() {

    // ---- LOGGING STUFF ----
    // Create data directory since I nuke it every compile/run
    std::filesystem::create_directory("dat");


    // --- Simulation Parameters ---
    // - Physics - 
    const float dt = 1.0e-5f;
    // const InteractionType interaction_type = InteractionType::RepulsiveForce;
    const InteractionType interaction_type = InteractionType::Gravity;
    const Vec2<float> box_max = {1.0f, 1.0f}; // Simulation Box size
    const Vec2<float> box_min = {-1.0f, -1.0f}; // Simulation Box size

    // - Simulation Control -
    const int total_steps = 10000;
    const int steps_between_recordings = 10;
    const int num_recordings = total_steps / steps_between_recordings;


    // MS tests
    const int num_systems = 24;
    const int particles_per_system = 2048;  // MAX 2048 for Leapfrog, due to __shared__ I believe. Kernel not started (?)    
                                            // MAX 1024 for Euler due to double buffering (Most likely since 50% of LF)

    const int num_particles = num_systems * particles_per_system; // This is now a calculated value // New: Number of independent systems

    const size_t total_particles_size = num_particles * sizeof(Particle<float>);

    const IntegratorType integrator_type = IntegratorType::Leapfrog;

    // --- CUDA Stream Setup ---
    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    cudaStream_t transfer_stream;
    CUDA_CHECK(cudaStreamCreate(&transfer_stream));

    // --- Host Data Initialization ---
    std::vector<Particle<float>> h_particles(num_particles);

    // - RNG and distributions - 
    std::mt19937 rng(1234);
    // Pos
    // std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> pos_dist(-0.5f, 0.5f);
    // Dist
    std::uniform_real_distribution<float> vel_dist(-0.5f, 0.5f);

    // Actual array initialization
    for (int i = 0; i < num_particles; ++i) {
        h_particles[i].position = {pos_dist(rng), pos_dist(rng)};
        h_particles[i].velocity = {vel_dist(rng), vel_dist(rng)};
    }

    // --- Device Memory Allocation ---
    // Ping pong won't work since a copy is needed. D-D copy
    size_t particles_size = num_particles * sizeof(Particle<float>); // Dynamic scaling
    Particle<float>* d_particles_sim = nullptr;     // pointer to device simulation buffer
    Particle<float>* d_particles_record = nullptr;  // pointer to device recording buffer
    CUDA_CHECK(cudaMalloc(&d_particles_sim, particles_size));
    CUDA_CHECK(cudaMalloc(&d_particles_record, particles_size));

    // --- DOUBLE BUFFER: Host Memory for Receiving Data (PINNED) ---
    // Create two buffers to ping-pong between.
    // Ping pong so we can have D-H transfer overlap with treatment (disk write).
    // Is it really useful here? Maybe for larger systems / longer writes.
    // Anyway, good to have I guess
    Particle<float>* h_pinned_buffers[2];
    CUDA_CHECK(cudaMallocHost(&h_pinned_buffers[0], particles_size)); // cudaMallocHost makes it pinned memory
    CUDA_CHECK(cudaMallocHost(&h_pinned_buffers[1], particles_size));

    // --- Initial Copy to GPU ---
    CUDA_CHECK(cudaMemcpy(d_particles_sim, h_particles.data(), particles_size, cudaMemcpyHostToDevice));

    // --- Initial Force Calculation for Leapfrog Integrator ---
    // Since leapfrog is our standard, I didn't add a check for Euler here
    // Euler do be going nuts with "high" dt tho
    run_initial_force_calculation<float>(
        d_particles_sim,
        num_systems,
        particles_per_system,
        interaction_type, 
        compute_stream
    );





    // --- Main Simulation Loop (Achieving Overlap) ---
    for (int i = 0; i < num_recordings; ++i) {
        int current_buffer_idx = i % 2;
        int previous_buffer_idx = (i + 1) % 2;

        // --- PROCESS DATA FROM PREVIOUS ITERATION (if it exists) ---
        if (i > 0) {
            // Wait for the PREVIOUS transfer to finish before writing
            CUDA_CHECK(cudaStreamSynchronize(compute_stream)); 

            int previous_step = i * steps_between_recordings;
            std::cout << "Processing data from step " << previous_step << std::endl;
            // std::cout << "  - Particle 0 position: (" 
            //           << h_pinned_buffers[previous_buffer_idx][0].position.x << ", " 
            //           << h_pinned_buffers[previous_buffer_idx][0].position.y << ")" << std::endl;
            
            // ==========================================================
            // UNFORMATTED SNAPSHOT SAVE
            // Dumping it raw, because that's how python likes it ;)
            // ==========================================================
            
            // Gemini was there^tm

            // 1. Create the filename
            std::stringstream ss;
            ss << "dat/" << previous_step << "_step.bin";
            std::string filename = ss.str();

            // 2. Open the file in binary mode
            std::ofstream outFile(filename, std::ios::binary);
            if (outFile.is_open()) {
                // 3. Write the raw byte data
                // We cast our Particle pointer to a char pointer for the write function
                outFile.write(reinterpret_cast<const char*>(h_pinned_buffers[previous_buffer_idx]),
                            total_particles_size); // total_particles_size is the size in bytes
                outFile.close();
                std::cout << "  - Saved frame to " << filename << std::endl;
            } else {
                std::cerr << "  - Error: Could not open file for writing: " << filename << std::endl;
            }
            // ==========================================================



        }

        // Run N steps of simulation
        // That should be non blocking cuz NVIDIA STREAMS BABY
        run_simulation_steps<float>(
            d_particles_sim,
            num_particles, num_systems, particles_per_system,
            integrator_type,      
            interaction_type,      
            box_min, box_max, dt,
            steps_between_recordings, compute_stream
        );

        // --- ASYNC DATA TRANSFER SETUP ---
        // Note on stream : First copy the data to a staging area on the GPU using the compute stream then launch D-H transfer on transfer stream
        CUDA_CHECK(cudaMemcpyAsync(d_particles_record, d_particles_sim, particles_size, cudaMemcpyDeviceToDevice, compute_stream));
        
        // Gemini says to refactor this and use events because StreamSnchronize is a hard block 
        // Can apparently hook into cuda events and avoid that. Would it corrupt data tho? 
        // I dunno. Needs testing.
        // For now, this works ¯\_(ツ)_/¯ (I copy pasted that shrug)
        CUDA_CHECK(cudaStreamSynchronize(compute_stream)); // Finish D-D copy
        CUDA_CHECK(cudaStreamSynchronize(transfer_stream)); // Ensure previous D-H is done before starting a new one, don't make data go badonkers


        // Start async transfer into the CURRENT buffer (cf ping pong)
        CUDA_CHECK(cudaMemcpyAsync(h_pinned_buffers[current_buffer_idx], d_particles_record, particles_size, cudaMemcpyDeviceToHost, transfer_stream));
    }

    // --- FINAL SYNCHRONIZATION AND PROCESSING ---
    // Same as above, but for the last batch
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    int final_step = num_recordings * steps_between_recordings;
    std::cout << "Processing data from final step " << final_step << std::endl;
    // std::cout << "  - Particle 0 position: (" 
    //           << h_pinned_buffers[(num_recordings - 1) % 2][0].position.x << ", " 
    //           << h_pinned_buffers[(num_recordings - 1) % 2][0].position.y << ")" << std::endl;


    // --- Cleanup ---
    // Driver passing the mop on the absolute mess we made of memory
    CUDA_CHECK(cudaFree(d_particles_sim));
    CUDA_CHECK(cudaFree(d_particles_record));
    CUDA_CHECK(cudaFreeHost(h_pinned_buffers[0]));
    CUDA_CHECK(cudaFreeHost(h_pinned_buffers[1]));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(transfer_stream));

    std::cout << "\nSimulation finished successfully." << std::endl;
    return 0; // Graceful exit in style *sparklesss*
}