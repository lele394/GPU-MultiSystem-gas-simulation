#pragma once
#include "particle_system.h"

// Updates position and velocity using the simple Euler method.
template <typename T>
__device__ void euler_integrator(Particle<T>& p, const Vec2<T>& acceleration, T dt) {
    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
    p.velocity.x += acceleration.x * dt;
    p.velocity.y += acceleration.y * dt;
}