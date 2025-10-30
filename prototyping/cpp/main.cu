#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// A simple macro to check for CUDA errors
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_code) \
                  << " in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// =========================================================================
// 1. DATA STRUCTURES
// =========================================================================

// A simple 2D vector type
template <typename T>
struct Vec2 {
    T x, y;
};

// Particle data structure
template <typename T>
struct Particle {
    Vec2<T> position;
    Vec2<T> velocity;
};

// =========================================================================
// 2. SIMULATION LOGIC (DEVICE CODE)
// =========================================================================

// --- Boundary Conditions ---
// Reverses velocity component when a particle hits a wall.
template <typename T>
__device__ void apply_mirror_boundaries(Particle<T>& p, const Vec2<T>& box_min, const Vec2<T>& box_max) {
    // Check and reflect on X-axis
    if (p.position.x < box_min.x) {
        p.position.x = box_min.x;
        p.velocity.x = -p.velocity.x;
    } else if (p.position.x > box_max.x) {
        p.position.x = box_max.x;
        p.velocity.x = -p.velocity.x;
    }

    // Check and reflect on Y-axis
    if (p.position.y < box_min.y) {
        p.position.y = box_min.y;
        p.velocity.y = -p.velocity.y;
    } else if (p.position.y > box_max.y) {
        p.position.y = box_max.y;
        p.velocity.y = -p.velocity.y;
    }
}

// --- Integrator ---
// Updates position and velocity using the simple Euler method.
template <typename T>
__device__ void euler_integrator(Particle<T>& p, const Vec2<T>& acceleration, T dt) {
    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
    p.velocity.x += acceleration.x * dt;
    p.velocity.y += acceleration.y * dt;
}

// =========================================================================
// 3. CUDA KERNEL
// =========================================================================

template <typename T>
__global__ void simulation_step_kernel(Particle<T>* particles, int num_particles, Vec2<T> box_min, Vec2<T> box_max, T dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles) {
        return;
    }

    // Load particle data from global memory into registers
    Particle<T> p = particles[idx];

    // --- Interaction Force Calculation ---
    // For an ideal gas, there are no interactions, so acceleration is zero.
    // In a more complex simulation (like Lennard-Jones), you would loop 
    // through all other particles here to calculate the total force.
    Vec2<T> acceleration = {0.0f, 0.0f};

    // --- Integration ---
    // Update the particle's state over the time step.
    euler_integrator(p, acceleration, dt);

    // --- Boundary Conditions ---
    // Check for and handle collisions with the container walls.
    apply_mirror_boundaries(p, box_min, box_max);

    // Write the updated particle data back to global memory
    particles[idx] = p;
}


// =========================================================================
// 4. HOST CODE (MAIN)
// =========================================================================

int main() {
    // --- Simulation Parameters ---
    const int num_particles = 20000000;
    const float dt = 0.001f; // Time step
    const int num_steps = 1000000;
    const Vec2<float> box_min = {-1.0f, -1.0f};
    const Vec2<float> box_max = {1.0f, 1.0f};

    std::cout << "Simulating " << num_particles << " particles for " << num_steps << " steps." << std::endl;

    // --- Initialization on the Host ---
    std::vector<Particle<float>> h_particles(num_particles);
    std::mt19937 rng(1234); // Mersenne Twister for random numbers, seeded for reproducibility
    std::uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> vel_dist(-0.5f, 0.5f);

    for (int i = 0; i < num_particles; ++i) {
        h_particles[i].position = {pos_dist(rng), pos_dist(rng)};
        h_particles[i].velocity = {vel_dist(rng), vel_dist(rng)};
    }

    // --- Allocate Memory on the GPU Device ---
    Particle<float>* d_particles = nullptr;
    size_t particles_size = num_particles * sizeof(Particle<float>);
    CUDA_CHECK(cudaMalloc(&d_particles, particles_size));

    // --- Copy Data from Host to Device ---
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles.data(), particles_size, cudaMemcpyHostToDevice));

    // --- Configure and Launch Kernel ---
    int threads_per_block = 256;
    int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching kernel with " << blocks_per_grid << " blocks and " 
              << threads_per_block << " threads per block." << std::endl;

    for (int step = 0; step < num_steps; ++step) {
        simulation_step_kernel<<<blocks_per_grid, threads_per_block>>>(
            d_particles, num_particles, box_min, box_max, dt
        );
    }
    
    // Ensure the kernel is finished before we proceed
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copy Results from Device to Host ---
    std::vector<Particle<float>> h_final_particles(num_particles);
    CUDA_CHECK(cudaMemcpy(h_final_particles.data(), d_particles, particles_size, cudaMemcpyDeviceToHost));

    // --- Print a Sample of the Results ---
    std::cout << "\nFinal positions of the first 5 particles:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Particle " << i << ": Pos=(" 
                  << h_final_particles[i].position.x << ", " 
                  << h_final_particles[i].position.y << "), Vel=("
                  << h_final_particles[i].velocity.x << ", "
                  << h_final_particles[i].velocity.y << ")" << std::endl;
    }

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_particles));

    std::cout << "\nSimulation finished successfully." << std::endl;

    return 0;
}