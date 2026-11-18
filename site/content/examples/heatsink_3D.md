---
title: "TO-220 regulator on a heatsink"
short: "TO-220 heatsink"
app: heatsink_3D
weight: 3
description: "3D heat conduction from a silicon die through the package and heatsink to the air."
---

## Try it

{{< example >}}

## Physics

Steady heat conduction with a volumetric source in the silicon die:

\[ -\nabla\cdot(k\,\nabla T) = q, \]

and every exposed surface loses heat to the surrounding air by Newton's law of cooling,

\[ -k\,\partial_n T = h\,(T - T_{\text{air}}). \]

The package is modeled as it is built: a copper tab with its mounting hole, the epoxy
body molded around it, the silicon die soldered to the tab, and three copper leads.
Three meshes compare the bare package against two heatsinks commonly clipped or bolted
to its tab. Lengths are in mm, power in W, temperature in °C.

## Implementation in femto

The convective term is a Robin condition: a mass-like matrix integrated over the
boundary facets, added to the conduction stiffness.

```cpp
sparse_matrix K  = integrate(dot(grad(phi), k_q, grad(phi)), domain);
sparse_matrix Kb = integrate(dot(phi, h_q, phi), bdr);     // h (T - T_air), the T part
Residual<Family::H1> f  = integrate(dot(q_q, phi), domain);
Residual<Family::H1> fb = integrate(dot(hT_q, phi), bdr);  // ... and the T_air part
```

The tetrahedral meshes come from `tools/generate_example_meshes.py`, and elements are
sorted into copper, epoxy, silicon and aluminum by the same box tests the generator used.
