---
title: "Notched fracture specimen"
short: "Fracture specimen"
app: fracture_2D
weight: 4
description: "2D finite-deformation elasticity: a notch opened in load steps, solved by Newton's method."
---

## Try it

{{< example >}}

## Physics

A square coupon with a sharp notch cut in from its left edge and a pin hole above and
below it. The pins are pulled apart in load steps, opening the notch. The material is
isotropic linear elastic or neo-Hookean, in plane strain or plane stress. With first
Piola–Kirchhoff stress $P(F)$ the equilibrium residual is

\[ r(u) = \int P(F)\!:\!\nabla\phi \, dA = 0, \qquad F = I + \nabla u. \]

Each load step is solved by Newton's method with the consistent tangent
$\partial P/\partial F$, and the reaction force at the upper pin is recorded per step.

## Implementation in femto

Displacement is quadratic on straight-sided triangles; constant-strain elements are far
too stiff to say anything useful about a notch tip. The residual and tangent are one
`integrate` each, with $P$ and $\partial P/\partial F$ evaluated at quadrature points:

```cpp
r_u = integrate(dot(P_q, grad(phi)), domain);
sparse_matrix K = integrate(dot(grad(phi), dP_dF_q, grad(phi)), domain);
```
