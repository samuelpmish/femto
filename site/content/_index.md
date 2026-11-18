---
title: femto
---

# femto

**femto** is a small C++20 finite element library. Problems are written close to
their weak forms: you create fields on a mesh, write an integrand with `dot(...)`
over test and trial functions, and `integrate(...)` it over a domain to get a
matrix or residual vector.

```cpp
Field<Family::H1> V = create_field<Family::H1>(mesh, 1, 1);
BasisFunction phi(V);
sparse_matrix K = integrate(dot(grad(phi), eps_q, grad(phi)), domain);
```

## What it provides

- **Meshes and elements** — triangles, quads, tetrahedra and hexahedra, with
  H1 and H(curl) spaces of arbitrary polynomial degree.
- **Quadrature-point kernels** — material laws are written as small functors
  invoked through `forall(...)`, so the same code runs threaded on the CPU or on a GPU.
- **Sparse direct solvers** — MKL or PaStiX on the CPU, cuDSS on the GPU.
- **WebAssembly** — every example in this site is the same C++ source,
  compiled with emscripten and running in your browser.

## Where to go next

- [Building]({{< relref "build" >}}) femto and its build options.
- Browse the [examples]({{< relref "examples" >}}): electrostatics, heat conduction,
  finite-deformation elasticity, modal analysis, magnetostatics, incompressible
  flow and wave propagation, each with the physics, the femto implementation,
  and a live version to play with.
