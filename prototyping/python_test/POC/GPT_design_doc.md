

## 1  Overview

Goal:
Simulate a small 3-D gas of identical particles interacting via the Lennard-Jones (LJ) potential, using **microcanonical (NVE) molecular dynamics**.
From the trajectory, measure fundamental equilibrium properties and compare their empirical distributions with the theoretical predictions of statistical physics.

---

## 2  Physical Model

* **Particles:**
  (N = 32 \text{–} 64), identical, mass (m) (choose (m=1) for convenience).
* **Simulation box:**
  Cube of side (L = 1).
  Volume (V = L^3 = 1).
* **Boundary conditions:**
  3-D **periodic boundaries** with **minimum-image convention**.

### 2.1 Lennard-Jones potential

[
U(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12}
- \left(\frac{\sigma}{r}\right)^6 \right]
]

* Use **reduced LJ units**: set (\varepsilon = 1), (\sigma = 1), (m = 1), (k_B = 1).
* Introduce a **cutoff** (r_c = 2.5\sigma) and shift the potential so (U(r_c)=0):
  [
  U_\text{shifted}(r) = U(r) - U(r_c)
  ]
  for (r \le r_c); zero otherwise.

---

## 3  Equations of Motion

For each particle (i):
[
m \frac{d^2 \mathbf{r}*i}{dt^2} = \sum*{j\neq i} \mathbf{F}*{ij},\qquad
\mathbf{F}*{ij} = -\nabla U_\text{shifted}(r_{ij})
]
with pair separation (\mathbf{r}_{ij}) computed under PBC.

**Integrator:** velocity Verlet
[
\begin{aligned}
\mathbf{r}_i(t+\Delta t) &= \mathbf{r}_i(t) + \mathbf{v}_i(t),\Delta t
+ \tfrac{1}{2m}\mathbf{F}_i(t),\Delta t^2,\
\mathbf{v}_i(t+\Delta t) &= \mathbf{v}_i(t) + \tfrac{1}{2m}\left[
\mathbf{F}_i(t) + \mathbf{F}_i(t+\Delta t)\right]\Delta t.
\end{aligned}
]

---

## 4  Initialization

1. **Positions**

   * Place particles on a simple cubic lattice or by random sequential addition avoiding overlap (e.g. min separation 0.8 σ).

2. **Velocities**

   * Draw each Cartesian component from a normal distribution of mean 0 and variance (k_B T_0/m) for an initial target temperature (T_0).
   * Subtract the center-of-mass velocity to set total momentum to zero.
   * Rescale so that total kinetic energy equals (\tfrac{3}{2} N k_B T_0).

3. **Parameters**

   * Typical reduced temperature (T_0) ≈ 1.0.
   * Time step (\Delta t) ≈ 0.001–0.005 (reduced units).
   * Run length: ~50 000–100 000 steps (discard first 10 % as equilibration).

---

## 5  Algorithm Outline

1. **Neighbor list (optional for small N):**
   O(N²) force loop is fine for N ≲ 64.

2. **Main loop:**

   * Integrate positions/velocities with velocity Verlet.
   * Apply periodic boundaries.
   * Every `sample_interval` steps record:

     * positions (for structural observables)
     * velocities (for kinetic/temperature/velocity distribution)
     * instantaneous potential energy U and kinetic energy K.

---

## 6  Quantities to Compute & Compare

All in reduced LJ units ((\varepsilon=\sigma=k_B=1)).

| Quantity                            | Formula                                                                                 | Comparison                                                              |   |                                                                                                     |
| ----------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | - | --------------------------------------------------------------------------------------------------- |
| **Instantaneous temperature**       | ( T = \tfrac{2}{3N} K )                                                                 | Time-average constant if energy conserved.                              |   |                                                                                                     |
| **Velocity component distribution** | Histogram of (v_x) (and (v_y, v_z))                                                     | Gaussian: ( f(v) = \sqrt{\frac{m}{2\pi k_BT}} e^{-m v^2 / (2 k_BT)} ).  |   |                                                                                                     |
| **Speed distribution**              | Histogram of (                                                                          | \mathbf{v}                                                              | ) | Maxwell–Boltzmann: ( f(v)=4\pi \left( \frac{m}{2\pi k_BT}\right)^{3/2} v^2 e^{- m v^2/(2 k_B T)} ). |
| **Total energy**                    | ( E=K+U )                                                                               | Should remain constant (numerical drift is a stability check).          |   |                                                                                                     |
| **Radial distribution (g(r))**      | Histogram of pair distances normalized by ideal-gas shell                               | Shows fluid structure; compare to g(r)=1 for ideal gas at same density. |   |                                                                                                     |
| **Pressure (virial)**               | ( P = \frac{N k_B T}{V} + \frac{1}{3V}\sum_{i<j} \mathbf{r}*{ij}\cdot \mathbf{F}*{ij} ) | Check vs. known LJ equation of state (qualitative match).               |   |                                                                                                     |

Optional dynamic property (longer runs):

* **Mean squared displacement** → diffusion coefficient (D = \lim_{t\to\infty} \frac{1}{6t}\langle |\mathbf{r}_i(t)-\mathbf{r}_i(0)|^2 \rangle).

---

## 7  Analysis & Plotting

* After discarding equilibration, pool samples across particles and times.
* Use **numpy / matplotlib**:

  * Velocity component histogram + Gaussian overlay.
  * Speed histogram + Maxwell–Boltzmann curve.
  * Radial distribution g(r) (bin size Δr ≈ 0.01–0.02).
  * Time series of total energy for energy-conservation check.
* For statistical confidence, you can bootstrap histogram counts.

---

## 8  Practical Notes

* **Scaling:** N ≈ 64 is small; O(N²) force evaluation is trivial in Python.
* **Energy conservation:** monitor drift; if large, reduce Δt.
* **Finite size effects:** distributions will match theory within sampling noise; small N broadens fluctuations—expected.

---

## 9  Suggested File/Function Structure (Python)

```
md_lj/
 ├── main.py            # driver
 ├── integrator.py      # velocity-Verlet
 ├── forces.py          # LJ potential & forces (abstracted for later swap)
 ├── analysis.py        # histograms, g(r), plots
 └── utils.py           # PBC utilities, initialization
```

*Keep interaction in a single `force_energy(positions)` function so alternative potentials can be dropped in later.*

---

### Minimal Simulation Parameters (good starting point)

| Parameter         | Value         |
| ----------------- | ------------- |
| N                 | 64            |
| Box length L      | 1.0           |
| σ, ε, m           | 1.0 (reduced) |
| Time step Δt      | 0.002         |
| Total steps       | 100 000       |
| Sampling interval | 10 steps      |
| Initial T         | 1.0           |

---

### Deliverables

* A Python script producing:

  * **Plots:** velocity Gaussian check, speed MB check, g(r), total-energy vs time.
  * Numerical averages of T, P, energy.
* All physics formulas (forces, virial pressure) are implemented exactly as written above.

