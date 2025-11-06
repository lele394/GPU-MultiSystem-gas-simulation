#pragma once

// 2D vector structure
template <typename T>
struct Vec2 {
    T x, y;
};

// Particle data structure
// Template should allow to change elements precision later on
template <typename T>
struct Particle {
    Vec2<T> position;
    Vec2<T> velocity;
    Vec2<T> acceleration; // Soo this is bad, memory footprint +33%
};