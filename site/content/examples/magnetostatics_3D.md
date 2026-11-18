---
title: "Induction coil and ferrite core"
short: "Induction coil"
app: magnetostatics_3D
weight: 6
description: "3D magnetostatics with H(curl) elements on a tetrahedral mesh of copper, ferrite and air."
---

## Try it

{{< example >}}

## Physics

An induction-heating work coil, six turns of thick copper wound around a cylindrical
ferrite core, in a box of air. The vector potential $\mathbf{A}$ solves

\[ \nabla\times(\nu\,\nabla\times\mathbf{A}) = \mathbf{J}, \qquad \mathbf{A}\times\mathbf{n}=0 \text{ on the box}, \]

with $\nu = 1/\mu$ per element and $\mathbf{J}$ the current along the wire. The helix is open
at both ends, so $\mathbf{J}$ is not divergence-free on the mesh: it is projected,
$\mathbf{J}-\nabla\psi$, with a scalar Poisson solve for $\psi$, and a small mass term
$\epsilon\mathbf{A}$ fixes the gauge of the otherwise singular curl-curl operator.
Neither changes $\mathbf{B}=\nabla\times\mathbf{A}$.

## Implementation in femto

$\mathbf{A}$ lives in the lowest-order $H(\text{curl})$ space on tetrahedra:

```cpp
A = create_field<Family::Hcurl>(mesh, 1, 1);
sparse_matrix K = integrate(dot(curl(phi), nu_q, curl(phi)), domain);
sparse_matrix M = integrate(dot(phi, eps_q, phi), domain);
Residual<Family::Hcurl> f = integrate(dot(J_q, phi), domain);
```

The mesh, `data/meshes/induction_coil.msh`, is built by `tools/generate_example_meshes.py`.
