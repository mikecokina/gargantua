# Gargantua

Experimental ray tracing project focused on black-hole lensing.

The goal of this project is to understand and implement photon trajectories
around a black hole, starting from simple Newtonian bending and progressing
toward physically correct General Relativity (Schwarzschild null geodesics).

The codebase is designed to be backend-agnostic (NumPy / CuPy) and modular,
with the long-term goal of extending everything from 2D to full 3D.

---

## What’s implemented

- Backend-agnostic math (NumPy / CuPy)
- 2D camera and ray generation
- Newtonian-style ray bending (toy model)
- Schwarzschild null geodesics in 2D (GR)
- Runge–Kutta (RK4) integration
- Simple 2D visualization with Matplotlib

---

## What’s next

- Full 3D camera model
- 3D Schwarzschild null geodesics
- Better numerical stability and adaptive stepping
- Accretion disk / background rendering

---

## Running the demo

```bash
python demo_2d_gr.py
python demo_2d_newt.py
```
