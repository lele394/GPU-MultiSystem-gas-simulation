#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "particle_system.h"
#include "simulation_defs.h" // <-- Make sure this is included!

// A simple macro to check for CUDA errors from any file
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s in file %s at line %d\n", \
                cudaGetErrorString(err_code), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Declaration for the one-off initial force calculation
template <typename T>
void run_initial_force_calculation(
    Particle<T>* d_particles, 
    int num_systems, 
    int particles_per_system, 
    InteractionType interaction_type, 
    cudaStream_t stream);

// Declaration for the main "Smart" Launcher
template <typename T>
void run_simulation_steps(
    Particle<T>* d_particles, 
    int total_particles, 
    int num_systems, 
    int particles_per_system,
    IntegratorType integrator_type,
    InteractionType interaction_type,
    Vec2<T> box_min, 
    Vec2<T> box_max, 
    T dt, 
    int steps_to_run, 
    cudaStream_t stream);