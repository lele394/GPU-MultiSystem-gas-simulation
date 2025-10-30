
# Section 1

Language : C++
Framework : CUDA

RTX 4060 important specs :
    - L1 cache : 128kb/SM
    - L2 cache : 24MB

    - 24 SM => 128 shading units
               4 tensor
               1 RT


Any use to texture units? starting conditions?


# Section 2

Words :

The goal here is to use multiple simulated gas systems to verify Statistical Physics properties.
Using different precisions we can check how it affects the statistical validity of the simulations.
For performance we want to try system sizes that are multiple of the number of shading units per SM, and/or fit inside L1 cache of each SM.


Main idea :

    Recommended :
        - Use template for easily modifiable precision (`float`, `double`)

    - For each SM, dispatch a simulation of a system with a number of particles Nx128 particles.
    - Record status of the particles every N steps 
    - Build known properties from Statistical Physics from simulation results

Additional ideas :
    - Size system to fit inside L1 cache => 128kb, localized to each SM, avoid L1<=>L2 transfer
        - Check performance diffs with systems of 128x2 kb and see if there is a perf draw



Initialization :
    - Uniform distribution (default)
    - Center skewed gaussian weighted?

Data recording :
    - Copy back position and velocity arrays directly



Delivrables :

- Simulation code with the following :
    - Adaptive number of particles per system
    - Templated system for precision manipulation
- Python analysis notebook 
    - Confirmation of simulation accuracy ¹
    - Reconstruct Statistical Physics properties such as boltzman

Notes :
- Seeding could be done on the CPU, could be nice messing with hash based RNGs too while at it.
- L1 cache (per SM) is 128kb on Ada lovelace (targetted here)
- Max amount of particles needs to be tuned to leave room for variables used in the integrator

¹ Can use one "dry run" that records arrays every step, compute energy conservation.

# Section 3 - Physics

Interactions : 
    - Ideal Gas (no interaction)
    - Lennard-Jones 12-6 (see Jeanna's course)
    - Coulomb interaction

Boundary condition :
    - Mirror velocity compnent on exit (ie p.x>boundary.x => v.x = -v.x)

Integrator :
    - Euler to start (Easiest, *dt)
    - Leapfrog (E conservation)
    - RK4(5) (maybe but most likely no)

Time :
    - Fixed dt (non adaptive)


Notes :
- Sampling every N steps, since the N-th step can be considered as new starting conditions for a new system


# Section 4 - Perfs

- Check L1 residency against L2 residency, against L3 residency (ie scale up systems)
- Check single system residency

Check for async transfer by duplicating data to VRAM and initiating transfer while running computation.

# Section 5 - Validation

- Previously mentionned dry run with 1-step snapshots
    - Using the previous dry run, can recover Stat Phys properties, energy conservation etc.