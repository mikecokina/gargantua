# Gargantua

Experimental ray tracing and spacetime visualization project focused on
black-hole lensing and photon trajectories.

The goal of Gargantua is to **understand and implement light propagation**
around massive objects, starting from simple Newtonian bending and progressing
toward physically correct General Relativity (Schwarzschild null geodesics),
with an emphasis on *clarity, correctness, and debuggability* rather than raw
rendering performance.

The project is built incrementally, validating each physical and numerical
assumption with explicit visualizations.

---

## Core ideas

- Start simple, then add physics
- Prefer explicit geometry over implicit magic
- Visualize everything that can go wrong
- Keep math backend-agnostic (NumPy / CuPy)
- Treat cameras, rays, and spacetime as first-class objects

---

## What’s implemented

### Architecture
- Backend-agnostic math layer (NumPy / CuPy via `get_array_module`)
- Modular separation of camera, geometry, physics, and ray marching
- Shared infrastructure between Newtonian and GR solvers

### Camera & Rays
- 2D and 3D camera models
- Camera defined by **position + forward vector + up**
- Rectilinear (pinhole) and equirectangular projections
- Explicit control over resolution and horizontal FOV (`fov_x`)
- Ray grid generation in both image space and physical slice planes

### Physics
- Newtonian-style ray bending (toy force-field model)
- Schwarzschild null geodesics (GR)
- Runge–Kutta 4 (RK4) integration
- Configurable step sizes, escape radius, and capture conditions
- Event-based handling of horizon capture

### Rendering & Debugging
- Image-space ray marching
- Exact segment–sphere intersection to avoid tunneling
- True physical **2D slice-plane visualization** through camera and BH center
- World-space XZ overview for sanity checking camera motion and orientation
- Extensive debug plots to diagnose:
  - camera basis errors
  - UV-plane mirroring
  - FOV distortion
  - stepping artifacts

---

## Demos

The repository contains several self-contained demos used both for validation
and for debugging geometry, camera conventions, and numerical behavior.

### 2D demos

- `demo_2d_newt.py`  
  Newtonian-style ray bending in 2D.  
  Useful as a baseline toy model and for sanity checking camera and stepping logic.

- `demo_2d_schw_01.py`  
  Schwarzschild null geodesics in 2D (top-down XZ projection).  
  Focused on correct GR bending and horizon behavior.

- `demo_2d_schw_02.py`  
  Variant of the 2D Schwarzschild demo with alternative parameterization and
  visualization choices for comparison and experimentation.

### 3D demos

- `demo_3d_newt.py`  
  3D Newtonian ray bending with:
  - image rendering
  - true physical 2D slice-plane visualization
  - world-space XZ overview for camera and basis debugging

- `demo_3d_schw.py`  
  3D Schwarzschild null geodesics using the same camera and visualization
  conventions as the Newtonian demo:
  - image-space rendering
  - slice-plane ray visualization through camera and BH center
  - world XZ overview
  This demo prioritizes correctness and diagnostics over visual polish.

- `demo_3d_steps_debug.py`  
  Interactive step-by-step debugger for a single Schwarzschild geodesic.
  Designed to inspect:
  - step size selection
  - horizon capture
  - camera-plane projections
  - numerical stability issues
