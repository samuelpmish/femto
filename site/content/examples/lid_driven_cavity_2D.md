---
title: "Lid-driven cavity"
short: "Lid-driven cavity"
app: lid_driven_cavity_2D
weight: 7
description: "2D incompressible Navier-Stokes with Taylor-Hood elements, Newton to the steady state."
---

## Try it

{{< example >}}

## Physics

A square box of fluid is driven by its top wall sliding at speed $U$; the other walls are
no-slip. The steady incompressible Navier–Stokes equations are

\[ (\mathbf{u}\cdot\nabla)\mathbf{u} - \nu\Delta\mathbf{u} + \nabla p = 0, \qquad \nabla\cdot\mathbf{u}=0, \]

and the Reynolds number $\mathrm{Re} = UL/\nu$ sets the shape of the central vortex.

## Implementation in femto

Taylor–Hood elements (Q2 velocity, Q1 pressure) with the steady equations solved directly
by Newton's method on the coupled system, rather than time-marching to a steady state.
The first step from rest is the Stokes solution; $\mathrm{Re}$ is then raised in stages so
every Newton solve starts close to its answer. The Jacobian blocks are each one `integrate`:

```cpp
Kuu = integrate(dot(grad(phi), nu_q,    grad(phi)), domain);
Cuu = integrate(dot(grad(phi), conv_q,  phi),       domain);
Gup = integrate(dot(psi,       gradp_q, grad(phi)), domain);
```

They are scattered into one unsymmetric matrix over $[\mathbf{u};p]$, factorized with LU,
and the factorization is `update()`d rather than rebuilt at each step.
