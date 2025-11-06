#pragma once
#include "particle_system.h"
#include <cmath>

template <typename T>
struct IdealGas {
    // IdealGas requires no parameters, so the default constructor is fine.

    __host__ __device__ IdealGas() {}


    __device__ Vec2<T> calculate_acceleration(
        const Particle<T>& my_particle,
        int my_particle_id,
        const Particle<T>* system_particles,
        int system_size
    ) const {
        // By definition, in an ideal gas, there are no forces between
        // particles. The acceleration is therefore always zero.
        return {0.0, 0.0};
    }
};






// ==========================================================
//            Where is Lennard-Jones???
// ==========================================================
// Lennard-Jones interaction is currently not functioning correctly and has been removed
// And attempts resulted in explosions.

// Might wanna come back to it later 





// ==========================================================
//            Gravity Interaction because LJ is borkedddd
// ==========================================================
template <typename T>
struct Gravity {
    const T G; // Gravitational constant, use reduce unit, set to 1, unless special stuff
    const T smoothing_factor_sq; // To prevent division by zero at r=0

    __host__ __device__ Gravity(T g_constant, T smoothing) 
        : G(g_constant), smoothing_factor_sq(smoothing * smoothing) {}
    __device__ Gravity() : G(1.0f), smoothing_factor_sq(0.01f*0.01f) {}

    __device__ Vec2<T> calculate_acceleration(
        const Particle<T>& my_particle,
        int my_particle_id,
        const Particle<T>* system_particles,
        int system_size
    ) const {
        Vec2<T> total_acceleration = {0.0, 0.0};

        for (int j = 0; j < system_size; ++j) {
            if (my_particle_id == j) {
                continue;
            }

            // distance
            Vec2<T> r_ij = {system_particles[j].position.x - my_particle.position.x,
                            system_particles[j].position.y - my_particle.position.y};
            
            // squared distance cached
            T r_sq = r_ij.x * r_ij.x + r_ij.y * r_ij.y;

            // softening factor
            T r_sq_smooth = r_sq + smoothing_factor_sq;

            // Actual inverse interaction 
            T inv_r_cubed = 1.0f / (r_sq_smooth * sqrt(r_sq_smooth));
            // Note: On a GPU, `rsqrt(r_sq_smooth)` might be faster but less precise. (gemini says it at least)

            // Magnitude if G ain't 1
            T acceleration_magnitude = G * 1.0f * inv_r_cubed; // mass = 1.0 for simplicity

            // update
            total_acceleration.x += r_ij.x * acceleration_magnitude;
            total_acceleration.y += r_ij.y * acceleration_magnitude;
        }

        return total_acceleration;
    }
};





















// ==========================================================
//     Purely Repulsive 1/x2 Interaction
// ==========================================================
template <typename T>
struct RepulsiveForce {
    const T epsilon; // Strength of the repulsion
    const T sigma;   // The "diameter" of the particle // Cutoff distance

    __host__ __device__ RepulsiveForce(T e, T s) : epsilon(e), sigma(s) {}

    __device__ Vec2<T> calculate_acceleration(
        const Particle<T>& my_particle,
        int my_particle_id,
        const Particle<T>* system_particles,
        int system_size) const
    {
        Vec2<T> total_acc = {0.0, 0.0};
        const T sigma2 = sigma * sigma; // squared for cutoff (should short it since not used anywhere else?)

        for (int j = 0; j < system_size; ++j) {
            if (j == my_particle_id) continue;

            Vec2<T> r_ij = {
                system_particles[j].position.x - my_particle.position.x,
                system_particles[j].position.y - my_particle.position.y
            };

            T r_sq = r_ij.x * r_ij.x + r_ij.y * r_ij.y;

            // Cutoff distance
            if (r_sq >= sigma2) continue;
            
            // Prevent singularity
            if (r_sq < 1e-6) r_sq = 1e-6;

            T scalar = 1.0f / r_sq;
 

            //  Had issues with time step and euler going badonkers
            // Leaving it here in case need to tweak again
            // const T max_acc_scalar = 1.0e10; 
            // if (scalar > max_acc_scalar) {
            //     scalar = max_acc_scalar;
            // }

            total_acc.x -= scalar * r_ij.x;
            total_acc.y -= scalar * r_ij.y;
        }
        return total_acc;
    }
};