---
title: "Coffee mug vibration modes"
short: "Coffee mug modes"
app: coffee_cup_modal
weight: 5
description: "3D modal analysis of a procedurally generated mug."
---

## Try it

{{< example >}}

## Physics

The free vibration modes of an elastic body satisfy the generalized eigenproblem

\[ K\,\mathbf{x} = \omega^2 M\,\mathbf{x}, \]

with stiffness matrix $K$ and mass matrix $M$; the lowest few eigenpairs give the
mug's resonant frequencies $f = \omega/2\pi$ and mode shapes. SI units throughout.

## Implementation in femto

The mug is generated procedurally. Both matrices are single `integrate` calls:

```cpp
K = sparse_matrix<>(integrate(dot(grad(phi), C_q, grad(phi)), domain));
M = sparse_matrix<>(integrate(dot(phi, rho_q, phi), domain));
```

and the lowest modes come from a shift-inverted Lanczos iteration in the $M$-inner
product (see `mug_analysis.hpp`).
