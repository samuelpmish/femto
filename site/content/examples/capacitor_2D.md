---
title: "Parallel-plate capacitor"
short: "Capacitor"
app: capacitor_2D
weight: 2
description: "2D electrostatics: capacitance from the field energy, fringing fields included."
---

## Try it

{{< example >}}

## Physics

Electrostatics: the potential $V$ satisfies

\[ -\nabla\cdot(\varepsilon\,\nabla V) = 0 \]

between the plates, with $V$ fixed on each. A dielectric block fills part of the
gap. The stored energy is

\[ W = \tfrac12 \int \varepsilon\,|\nabla V|^2 \, dA, \]

and the capacitance follows as $C = 2W/\Delta V^2$. Because the whole box of air is
solved, fringing fields are included, which is what separates the result from the
textbook $\varepsilon A/d$.

Lengths are in mm and permittivities relative to $\varepsilon_0$, so $C$ comes out in
units of $\varepsilon_0$ per unit depth.

## Implementation in femto

The plates are a material with an enormous permittivity, so each one floats to a
single voltage; one node per plate is pinned. The system is one `integrate`:

```cpp
BasisFunction phi(V);
sparse_matrix K = integrate(dot(grad(phi), eps_q, grad(phi)), domain);
```

where `eps_q` is the per-quadrature-point permittivity. The energy is then
$\tfrac12 V^T K V$.
