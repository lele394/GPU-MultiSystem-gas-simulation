#pragma once

// 22D vector structure
template <typename T>
struct Vec2 {
    T x, y;
};

// Particle data structure
template <typename T>
struct Particle {
    Vec2<T> position;
    Vec2<T> velocity;
};