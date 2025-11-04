// Umbrella integrator, because refactoring all this code every time is a pain
// And having multiple integrator is cool anyway
#pragma once

enum class IntegratorType {
    Euler,
    Leapfrog
};


// Interaction model types umbrella 
enum class InteractionType {
    Gravity,
    RepulsiveForce,
    IdealGas
};
// Nuked LJ to oblivion, obviously didn't work as intended

// Fckin Umbrella academy type shi in this file