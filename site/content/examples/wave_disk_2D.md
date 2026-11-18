---
title: "Pressure pulse in a disk"
short: "Absorbing boundaries"
app: wave_disk_2D
weight: 8
description: "2D acoustics against the analytic solution: hard wall, absorbing condition, or a PML ring."
---

## Try it

{{< example >}}

Click to add pulses.

## Physics

A pressure pulse set off inside the unit disk should spread out and leave, as it does in
the analytic solution of the wave equation

\[ \partial_t^2 u = c^2\,\Delta u \]

on the whole plane. A finite mesh has to stop somewhere, and what happens at its rim is
the point of the example. Three treatments are compared:

- **hard wall**, the natural boundary condition, which echoes the pulse straight back;
- the first-order **absorbing boundary condition** of Bayliss and Turkel, which lets the
  parts arriving head-on leave and reflects the rest;
- a ring of **perfectly matched layer** elements, which absorbs nearly everything.

The reference is the Hankel transform of the Gaussian pulse, a sum of Bessel modes with
$\cos(ckt)$ time dependence.

## Implementation in femto

P1 triangles or Q1 quadrilaterals from `Mesh::disk`, with the PML ring extruded from the
rim. The mass matrix is lumped and time stepping is SSP-RK3, so each stage is just a
residual evaluation:

```cpp
r = integrate(dot(flux_q, grad(phi)), domain);
```

The PML is `ScalarPML<2>` from `materials/pml.hpp`, using the Grote–Sim auxiliary system
with one auxiliary vector per quadrature point.
