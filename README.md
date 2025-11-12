Pretty cool N-Body simulation that can do a bunch of stuff.


> There isn't any kind of forces computation optimization, ie we don't use the interactions symmetry 
> > Might be usefull since I run on a single SM, CF fortran N-body to check if that damn race condition still mess around

> Can add different kind of interactions

> Euler and Leapfrog integrators are available. Might add RK4(5) but ain't for now.


Went with a simple integrator instead of doing object oriented with `p1.interact(p2)`


# Mixed Precision analysis

## Gonna need to parametrize :




### `particle_system.h`

`position`, `velocity`, `acceleration` should be added to the template
=> No issue with `Vec2` 


### `boundaries.h` 
> Need to pass particle precisions
`box_max` and `box_min` most likely don't really matter, maybe check what comparison is best depending on `Particle.position` precision is?
> ie is float>double worse than double>double or something ........


### `interactions.h`
#### `RepulsiveForce`

> Need to pass particle precisions

Could be nice to add `epsilon` and `sigma` to see if precision here would impact anything (best bet is we could go wayyyyyyyy down, float8 or something even?)
I'm sure there's some dumpster diving to do in this one. 


#### `IdealGas`

> Need to pass particle precisions

Not much to do 


#### `Gravity`

> Need to pass particle precisions

Try to parametrize `G` and `smoothing_factor_sq`
> Might be easier to just change it in the function tbh


#### `integrators.h`

> Need to pass particle precisions

# `EulerIntegrator`

Should depend on particle precisions *when* I refactor it
> Spoiler, I probably won't because Euler sucks

#### `LeapfrogIntegrator`

> Need to pass particle precisions

Parametrize `dt` too? Seems actually duper important


#### `Simulation.cu`

> Need to pass particle precisions

F my sanity

There's so much to do here

I need to get rid of the switch and make it cleaner

#### `Simulation.cuh`
> Need to pass particle precisions

Update templated declarations ;(


#### `main.c`

Tough nut, will change during implementation 