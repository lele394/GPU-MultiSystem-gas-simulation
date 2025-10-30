#pragma once
#include "particle_system.h"
#include <cmath>


// Functor for Ideal Gas (No Interaction)
// The acceleration is always zero.
template <typename T>
struct IdealGas {
    __device__ Vec2<T> calculate_acceleration(const Particle<T>& p) const {
        return {0.0, 0.0};
    }
};



// Functor for Lennard-Jones Interaction
template <typename T>
struct LennardJones {
    const T epsilon; // Depth of the potential well
    const T sigma; // Distance where potential is zero
    const T cutoff_sq; // To avoid calculating forces for distant particles
    // The calculation function now takes the particle we are focused on,
    // its ID, the array of all particles in its system, and the system size.

    // What kind of magic is that
    __device__ LennardJones(T e, T s, T c_sq) : epsilon(e), sigma(s), cutoff_sq(c_sq) {}


    __device__ Vec2<T> calculate_acceleration(
        const Particle<T>& my_particle,
        int my_particle_id,
        const Particle<T>* system_particles, // Pointer to the array of particles (in shared memory)
        int system_size
    )  const {
        Vec2<T> total_acceleration = {0.0, 0.0};

        // Loop over every other particle in THIS system
        for (int j = 0; j < system_size; ++j) {
            if (my_particle_id == j) {
                continue; // Don't interact with self
            }

            // Calculate distance vector
            Vec2<T> r_ij = {system_particles[j].position.x - my_particle.position.x,
                            system_particles[j].position.y - my_particle.position.y};
            
            T r_sq = r_ij.x * r_ij.x + r_ij.y * r_ij.y;

            // Optional: Use a cutoff to improve performance <= we consider every particles here, added for future use
            // if (r_sq > cutoff_sq) {
            //     continue;
            // }

            // square root smoothing to avoid singularities
            r_sq += 1.0e-6;

            T sigma_sq = sigma * sigma;
            T r2_inv = sigma_sq / r_sq;
            T r6_inv = r2_inv * r2_inv * r2_inv;
            T r12_inv = r6_inv * r6_inv;
            
            // Lennard-Jones force formula (simplified for F/m = a)
            T force_magnitude = 24.0 * epsilon / r_sq * (2.0 * r12_inv - r6_inv);

            total_acceleration.x += r_ij.x * force_magnitude;
            total_acceleration.y += r_ij.y * force_magnitude;
        }

        return total_acceleration;
        // return {1000.0f, 1000.0f}; // ideal gas debug
    }

};







































// ==========================================================
//            Gravity Interaction because LJ is borkedddd
// ==========================================================
template <typename T>
struct Gravity {
    const T G; // Gravitational constant
    const T smoothing_factor_sq; // To prevent division by zero at r=0

    __device__ Gravity(T g_constant, T smoothing) 
        : G(g_constant), smoothing_factor_sq(smoothing * smoothing) {}

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

            // Calculate distance vector from me to particle j
            Vec2<T> r_ij = {system_particles[j].position.x - my_particle.position.x,
                            system_particles[j].position.y - my_particle.position.y};
            
            T r_sq = r_ij.x * r_ij.x + r_ij.y * r_ij.y;

            // The force on 'me' (particle i) is F = G * m_i * m_j / r^2
            // The acceleration is a = F / m_i
            // So, a = G * m_j / r^2.  My own mass cancels out!
            
            // The numerically stable formula for the acceleration vector is:
            // a_vec = (G * m_j * r_vec) / (r^2 + smoothing^2)^(3/2)

            // 1. Apply the softening factor
            T r_sq_smooth = r_sq + smoothing_factor_sq;

            // 2. Calculate the inverse cube of the smoothed distance
            T inv_r_cubed = 1.0f / (r_sq_smooth * sqrt(r_sq_smooth));
            // Note: On a GPU, `rsqrt(r_sq_smooth)` might be faster but less precise.

            // 3. Calculate the magnitude of the acceleration contribution
            T acceleration_magnitude = G * 1.0f * inv_r_cubed; // mass = 1.0 for simplicity

            // 4. Add the acceleration vector
            total_acceleration.x += r_ij.x * acceleration_magnitude;
            total_acceleration.y += r_ij.y * acceleration_magnitude;
        }

        return total_acceleration;
    }
};