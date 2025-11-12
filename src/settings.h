#pragma once
#include "simulation_defs.h"
#include "particle_system.h"
#include <random>

// Enable or disable kernel logs
const bool KernelLogsEnabled = true; 
const bool Timing_Profiling = true; // If you want to display timing info (computed anyway) 
const bool Benchmarking_Mode = true; // If true, displays benchmarking infos

// Perf logs filename
const std::string perf_filename = "../perfs.csv";


// Simulation settings
const int num_systems = 24;
const int particles_per_system = 224;  


const int total_steps = 100;
const int steps_between_recordings = 10;

const float dt = 1.0e-3f;
const InteractionType interaction_type = InteractionType::RepulsiveForce;
// const InteractionType interaction_type = InteractionType::Gravity;

// Integrator settings
const IntegratorType integrator_type = IntegratorType::Leapfrog;


const Vec2<float> box_max = {1.0f, 1.0f}; // Simulation Box size
const Vec2<float> box_min = {-1.0f, -1.0f}; // Simulation Box size

// Interactions settings
// Repulsive Force parameters
const float epsilon = 1.0f;
const float sigma = 0.01f;

// Gravity parameters
const float G = 9.81f;
const float gravity_smoothing = 0.05f;






//RNG settings
const unsigned int RNG_Seed = 1234;
// Soooooo I need to move the distribs here but I'm not yet sure how



// CUDA stuff
const int threads_per_block = 256;

