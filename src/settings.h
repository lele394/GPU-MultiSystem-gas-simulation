#pragma once
#include "simulation_defs.h"
#include "particle_system.h"
#include <string>
#include <cstdlib> // For std::getenv
#include <iostream> // For std::cout/cerr
#include <iomanip> // For std::setw to format the output nicely


const std::string RESET       = "\033[0m";
const std::string BLACK       = "\033[30m";
const std::string RED         = "\033[31m";
const std::string GREEN       = "\033[32m";
const std::string YELLOW      = "\033[33m";

const std::string T_DEFAULT   = YELLOW + "[DEFAULT]" + RESET;
const std::string T_ENV_VAR   = RED + "[ENV VAR]" + RESET;



// Mostly Gemini wrote that since it's boiler plate code for settings management
// I'm not sure if there's some fancy way to do that?
// Using env vars might be peak stupidity but hey, it works for now ¯\_(ツ)_/¯




struct Settings
{
    // These members are now non-const so they can be modified at runtime
    bool KernelLogsEnabled = true; 
    bool Timing_Profiling = true;
    bool Benchmarking_Mode = true;

    std::string perf_filename = "../perfs.csv";

    // Simulation settings
    int num_systems = 24;
    int particles_per_system = 2048;  
    int total_steps = 100;
    int steps_between_recordings = 10;
    float dt = 1.0e-3f;
    
    InteractionType interaction_type = InteractionType::RepulsiveForce;
    IntegratorType integrator_type = IntegratorType::Leapfrog;

    Vec2<float> box_max = {1.0f, 1.0f};
    Vec2<float> box_min = {-1.0f, -1.0f};

    // Interactions settings
    float epsilon = 1.0f;
    float sigma = 0.01f;
    float G = 9.81f;
    float gravity_smoothing = 0.05f;

    unsigned int RNG_Seed = 1234;
    int threads_per_block = 256;
};



// Helper function to safely parse a float from an environment variable with logging
inline float get_env_float(const char* var_name, float default_val) {
    const char* env_val = std::getenv(var_name);
    
    // Set up formatting for aligned output
    const int label_width = 10;
    const int name_width = 30;

    if (env_val == nullptr) {
        // The environment variable was not found, use the default
        std::cout << std::left << std::setw(label_width) << T_DEFAULT
                  << std::setw(name_width) << var_name 
                  << "= " << default_val << std::endl;
        return default_val;
    }
    
    try {
        // Try to convert the string value to a float
        float value = std::stof(env_val);
        std::cout << std::left << std::setw(label_width) << T_ENV_VAR
                  << std::setw(name_width) << var_name 
                  << "= " << value << std::endl;
        return value;
    } catch (const std::invalid_argument& ia) {
        // The conversion failed (e.g., the value was "hello")
        std::cerr << "Warning: Invalid float value for " << var_name << ". Using default.\n";
        std::cout << std::left << std::setw(label_width) << T_DEFAULT
                  << std::setw(name_width) << var_name 
                  << "= " << default_val << " (parse error)" << std::endl;
        return default_val;
    }
}

// Helper function to safely parse an int from an environment variable with logging
inline int get_env_int(const char* var_name, int default_val) {
    const char* env_val = std::getenv(var_name);
    const int label_width = 10;
    const int name_width = 30;

    if (env_val == nullptr) {
        std::cout << std::left << std::setw(label_width) << T_DEFAULT
                  << std::setw(name_width) << var_name 
                  << "= " << default_val << std::endl;
        return default_val;
    }
    
    try {
        int value = std::stoi(env_val);
        std::cout << std::left << std::setw(label_width) << T_ENV_VAR
                  << std::setw(name_width) << var_name 
                  << "= " << value << std::endl;
        return value;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Warning: Invalid integer value for " << var_name << ". Using default.\n";
        std::cout << std::left << std::setw(label_width) << T_DEFAULT
                  << std::setw(name_width) << var_name 
                  << "= " << default_val << " (parse error)" << std::endl;
        return default_val;
    }
}

// Helper function to safely parse an InteractionType from an environment variable with logging
inline InteractionType get_env_interaction(const char* var_name, InteractionType default_val) {
    const char* env_val_str = std::getenv(var_name);
    const int label_width = 10;
    const int name_width = 30;

    auto log_value = [&](const std::string& source, InteractionType val) {
        std::string val_str;
        switch(val) {
            case InteractionType::Gravity: val_str = "Gravity"; break;
            case InteractionType::RepulsiveForce: val_str = "RepulsiveForce"; break;
            case InteractionType::IdealGas: val_str = "IdealGas"; break;
        }
        std::cout << std::left << std::setw(label_width) << source
                  << std::setw(name_width) << var_name 
                  << "= " << val_str << std::endl;
    };

    if (env_val_str == nullptr) {
        log_value(T_DEFAULT, default_val);
        return default_val;
    }

    std::string val_str(env_val_str);
    InteractionType value = default_val;
    bool found = false;

    if (val_str == "Gravity") { value = InteractionType::Gravity; found = true; }
    else if (val_str == "RepulsiveForce") { value = InteractionType::RepulsiveForce; found = true; }
    else if (val_str == "IdealGas") { value = InteractionType::IdealGas; found = true; }

    if (found) {
        log_value(T_ENV_VAR, value);
        return value;
    } else {
        std::cerr << "Warning: Invalid value '" << val_str << "' for " << var_name << ". Using default.\n";
        log_value(T_DEFAULT, default_val);
        return default_val;
    }
}


// The main function to get the settings.
// It now needs argc and argv from main() to check for the flag.
inline Settings GetSettings(int argc, char* argv[])
{

    Settings s; // Start with default values

    // Check if the "--env" or "-e" flag is present
    bool use_env = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--env" || arg == "-e") {
            use_env = true;
            break;
        }
    }

    if (use_env) {
        std::cout << "--- Loading settings from Environment Variables ---" << std::endl;
        // If the flag is found, try to load each parameter from its environment variable
        
        // Simulation settings
        s.num_systems = get_env_int("SIM_NUM_SYSTEMS", s.num_systems);
        s.particles_per_system = get_env_int("SIM_PARTICLES_PER_SYSTEM", s.particles_per_system);
        s.total_steps = get_env_int("SIM_TOTAL_STEPS", s.total_steps);
        s.steps_between_recordings = get_env_int("SIM_STEPS_BETWEEN_RECORDINGS", s.steps_between_recordings);
        s.dt = get_env_float("SIM_DT", s.dt);

        // Enum settings
        s.interaction_type = get_env_interaction("SIM_INTERACTION", s.interaction_type);
        // Add get_env_integrator(...) here if I ever need to use something else than Leapfrog

        // Interactions settings
        s.epsilon = get_env_float("SIM_EPSILON", s.epsilon);
        s.sigma = get_env_float("SIM_SIGMA", s.sigma);
        s.G = get_env_float("SIM_GRAVITY_G", s.G);
        s.gravity_smoothing = get_env_float("SIM_GRAVITY_SMOOTHING", s.gravity_smoothing);

        // RNG settings
        s.RNG_Seed = get_env_int("SIM_RNG_SEED", s.RNG_Seed);
        std::cout << "-----------------------------------------------" << std::endl;
    } else {
        std::cout << "--- Using default compile-time settings ---" << std::endl;
    }

    return s;
}




