#pragma once
#include "particle_system.h"

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