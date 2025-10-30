#pragma once

// Include the main CUDA runtime header to define types like cudaError_t
#include <cuda_runtime.h>
// Include for fprintf and exit
#include <cstdio>
#include <cstdlib>

#include "particle_system.h"

// A simple macro to check for CUDA errors from any file
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s in file %s at line %d\n", \
                cudaGetErrorString(err_code), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Declaration of the host-callable wrapper function.
// The implementation is in simulation.cu
// cudaStream_t for async execution, we're messing with streams here.
template <typename T>
void run_n_simulation_steps(Particle<T>* d_particles, int num_particles, 
                            Vec2<T> box_min, Vec2<T> box_max, T dt, 
                            int steps_to_run, cudaStream_t stream);


// Add a declaration for the new launcher of LJ
template <typename T>
void run_lj_simulation_steps(
    Particle<T>* d_particles, int total_particles, int num_systems, int particles_per_system,
    Vec2<T> box_min, Vec2<T> box_max, T dt, int steps_to_run, cudaStream_t stream);