#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "particle_system.h"
#include "simulation_defs.h"

#include "settings.h"

// A simple macro to check for CUDA errors from any file (Thanks Gemini for this one)
#define CUDA_CHECK(err) { \
    cudaError_t err_code = err; \
    if (err_code != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s in file %s at line %d\n", \
                cudaGetErrorString(err_code), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Below templated calls for simulation steps
// That's actually pretty cool

// Declaration for the one-off initial force calculation
// I'm gonna need to change that too when adding mixed precision
template <typename T>
void run_initial_force_calculation(
    Particle<T>* d_particles, 
    int num_systems, 
    int particles_per_system, 
    InteractionType interaction_type, 
    cudaStream_t stream,
    Settings settings);

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
    cudaStream_t stream,
    Settings settings);