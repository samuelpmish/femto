---
title: "Hall effect current sensor"
short: "Hall current sensor"
app: hall_sensor_2D
weight: 1
description: "2D magnetostatics: a gapped ferrite core reads the current in a wire passing through it and screens out its neighbor."
---

## Try it

{{< example >}}

Drag the wires, or click to place one. The measured wire sits in the core's
window, the external one outside it. Toggle the core to see how much of the
reading it is responsible for.

## Physics

A Hall current sensor measures the current in a wire without touching it. The
wire passes through the window of a ferrite core; by Ampère's law the core
gathers the circulating field and drives it across a narrow gap, where a Hall
element reads it:

\[ \oint \mathbf{H}\cdot d\mathbf{l} = I_{\text{enc}}. \]

With a high-permeability core, nearly all of the drop in $\int \mathbf{H}\cdot d\mathbf{l}$ happens
across the gap of width $g$, so the field there is about

\[ B \approx \frac{\mu_0 I}{g}, \]

independent of where the wire sits in the window, and a second wire outside
the core contributes almost nothing, since its field loops around the core
without enclosing it. Without the core neither is true.

In two dimensions the out-of-plane vector potential $A$ satisfies

\[ -\nabla\cdot\left(\nu\,\nabla A\right) = J, \qquad \mathbf{B} = \nabla\times(A\,\mathbf{e}_z), \]

with reluctivity $\nu = 1/\mu$ ($\mu_r = 1000$ in the core). Each wire is a line
current, $J = I\,\delta(\mathbf{x}-\mathbf{x}_w)$. The air region is a disk with an
absorbing rim, $\partial A/\partial r + A/R = 0$, which a line current's field
satisfies up to a constant and its dipole part exactly, so the disk can stay small.

The Hall element is modeled on a real part (an AKM HW-101A, about 5.8 mV/mT at
1 V drive); its output drives a center-zero analog meter, drawn under the core.

## Implementation in femto

The mesh is generated on the fly by [Triangle](https://www.cs.cmu.edu/~quake/triangle.html)
from the outlines of the core, sensor and rim, with a size function that refines
to 0.4 mm across the gap. Regional attributes tag each triangle as air, core or sensor.

The potential is quadratic (P2). The stiffness matrix is one `integrate` over the
domain, with the reluctivity evaluated per element by a `forall` functor, and the
absorbing rim adds a boundary mass term on the same sparsity pattern:

```cpp
BasisFunction phi(A);
nd::cpu_array<double, 3> nu_q = forall(Reluctivity{mu_r}, at_quadrature_points(region, total(domain.num_qpts) / nt));
sparse_matrix K = integrate(dot(grad(phi), nu_q, grad(phi)), domain);
// ... + int phi phi / R over boundary_of(mesh)
invK = inv(K);
```

Because the wires are point sources, moving one never changes the matrix: the
factorization `invK` is computed once per core setting, and each wire position is a
back-substitution with a load of $\mu_0 I \phi_i(\mathbf{x})$ on the nodes of the element
containing the wire.

```cpp
A_primary.v()  = potential(primary);
A_external.v() = potential(external);
signal       = sensor_reading(A_primary);
interference = sensor_reading(A_external);
```

Solving each wire separately splits the reading into the signal from the measured
wire and the interference from the other, which is what the page reports.

A quadratic field drawn at the mesh vertices looks no better than a linear one, so the
example evaluates $\mathbf{B}$ on a second `Domain` whose triangle rule is a grid of
reference points, and the page draws the small triangles between them.
