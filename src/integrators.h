#pragma once
#include "particle_system.h"
#include "interactions.h" // Needed for InteractionModel

// ==========================================================
//                 Euler Integrator (Single-Step)
// ==========================================================
// Does the whole job
template <typename T>
struct EulerIntegrator {
    template <typename InteractionModel>
    __device__ void integrate(
        Particle<T>& p,
        int p_idx,
        const Particle<T>* read_buffer,
        int system_size,
        const InteractionModel& interaction,
        T dt) const 
    {

        // I need to refactor this, the acceleration array already exists in the particle
        //  ^ No you don't, you basically dropped that feature


        // Boom, interactions in the integrator
        Vec2<T> acceleration = interaction.calculate_acceleration(p, p_idx, read_buffer, system_size);

        // Good ol' Euler update
        p.position.x += p.velocity.x * dt;
        p.position.y += p.velocity.y * dt;
        p.velocity.x += acceleration.x * dt;
        p.velocity.y += acceleration.y * dt;
    }
};

// ==========================================================
//              Leapfrog Integrator (Multi-Step)
// ==========================================================
template <typename T>
struct LeapfrogIntegrator {

    // LEAP
    // Part 1: Update positions using OLD acceleration a(t)
    __device__ void pre_force_update(Particle<T>& p, T dt) const {
        p.velocity.x += 0.5f * p.acceleration.x * dt;
        p.velocity.y += 0.5f * p.acceleration.y * dt;
        p.position.x += p.velocity.x * dt;
        p.position.y += p.velocity.y * dt;
    }

    // FROG
    // Part 2: Update velocities using NEW acceleration a(t+dt)
    __device__ void post_force_update(Particle<T>& p, const Vec2<T>& new_acceleration, T dt) const {

        // Can I get rid of new_acceleration and just use p.acceleration?
        //  ^ No, because p.acceleration is still a(t), we need a(t+dt) here

        p.velocity.x += 0.5f * new_acceleration.x * dt;
        p.velocity.y += 0.5f * new_acceleration.y * dt;
        p.acceleration = new_acceleration; // Store for next iteration
    }
};