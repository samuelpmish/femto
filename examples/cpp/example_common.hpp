#pragma once

// helpers shared by the physics examples: Dirichlet elimination for the
// linear systems they assemble, mesh export for the browser viewer, and the
// typed-array conversions the emscripten bindings use.  Each example is one
// .cpp with a simulation object, a main() for the native build, and
// EMSCRIPTEN_BINDINGS exposing the same object to javascript.

#include <vector>
#include <cstdint>
#include <algorithm>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"
#include "linear_algebra/sparse_matrix.hpp"
#include "linear_algebra/sparse_direct.hpp"
#include "forall.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/val.h>
#endif

namespace femto {

// K u = f with u[dofs] prescribed: the constrained columns move to the right
// hand side, the constrained rows and columns become identity, and the
// factorization is kept for repeated solves (load steps, time steps).  For a
// Newton step pass the residual as f and the constraint error as values
struct ConstrainedSystem {

  std::vector<int> dofs;
  std::vector<int> order;  // sparse_matrix slices sort their columns, so Kc's columns are dofs[order]
  sparse_matrix<> Kc;      // the constrained columns
  sparse_matrix<> Kmod;    // identity rows/columns at the constraints
  sparse_factorization invK;
  bool factorized = false;

  void set_matrix(const sparse_matrix<> & K, bool spd = true) {
    order.resize(dofs.size());
    for (size_t k = 0; k < order.size(); k++) { order[k] = int(k); }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return dofs[a] < dofs[b]; });
    sparse_matrix<> K0 = K;
    Kc = K0({}, dofs);
    Kmod = K0;
    Kmod({}, dofs) = [](int i, int j){ return double(i == j); };
    Kmod(dofs, {}) = [](int i, int j){ return double(i == j); };
    Kmod.symmetry = Symmetry::Symmetric;
    Kmod.definiteness = spd ? Definiteness::PositiveDefinite : Definiteness::Indefinite;
    if (factorized) { invK.update(Kmod); } else { invK = inv(Kmod); factorized = true; }
  }

  vector solve(vector f, const vector & values) {
    FEMTO_ASSERT(values.sz == dofs.size(), "one prescribed value per constrained dof");
    FEMTO_ASSERT(f.sz == Kmod.nrows, "right hand side must match the matrix");
    vector sorted_values(uint32_t(dofs.size()));
    for (size_t k = 0; k < dofs.size(); k++) { sorted_values[k] = values[order[k]]; }
    f = f - dot(Kc, sorted_values);
    f[dofs] = values;
    return dot(invK, f);
  }

};

// interleaved vertex coordinates, sdim per vertex
inline std::vector<double> vertex_coordinates(const Mesh<> & mesh) {
  uint32_t nv = mesh.vert.shape[0], sdim = mesh.spatial_dimension;
  std::vector<double> out(size_t(nv) * sdim);
  for (uint32_t i = 0; i < nv; i++) {
    for (uint32_t c = 0; c < sdim; c++) { out[size_t(i) * sdim + c] = mesh.X.data(i, c); }
  }
  return out;
}

// the vertex ids of every cell of one geometry, flattened
inline std::vector<uint32_t> cell_vertices(const Mesh<> & mesh, Geometry g) {
  uint32_t n = mesh[g].shape[0];
  uint32_t nv = (g == Geometry::Triangle) ? 3 : (g == Geometry::Quadrilateral) ? 4 : (g == Geometry::Tetrahedron) ? 4 : (g == Geometry::Hexahedron) ? 8 : 2;
  std::vector<uint32_t> out(size_t(n) * nv);
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < nv; j++) { out[size_t(i) * nv + j] = mesh[g](i, j).index; }
  }
  return out;
}

// the boundary facets, as vertex ids: edges (2) of a 2D mesh, triangles (3)
// then quadrilaterals (4) of a 3D mesh, with the facet size in front so the
// viewer can split them
inline std::vector<uint32_t> surface_facets(const Mesh<> & mesh) {
  SubMesh<> bdr = boundary_of(mesh);
  std::vector<uint32_t> out;
  auto append = [&](Geometry g, const nd::array<uint32_t, 1, memory::space::cpu> & ids, uint32_t nv) {
    for (uint32_t k = 0; k < ids.shape[0]; k++) {
      out.push_back(nv);
      for (uint32_t j = 0; j < nv; j++) { out.push_back(mesh[g](ids(k), j).index); }
    }
  };
  if (mesh.geometry_dimension == 2) { append(Geometry::Edge, bdr.edge, 2); }
  if (mesh.geometry_dimension == 3) { append(Geometry::Triangle, bdr.tri, 3); append(Geometry::Quadrilateral, bdr.quad, 4); }
  return out;
}

// a per-cell value repeated at each of the cell's quadrature points, so a
// q-function can take it as an input
inline nd::cpu_array<int, 2> at_quadrature_points(const std::vector<int> & per_cell, uint32_t qpe) {
  nd::cpu_array<int, 2> out({uint32_t(per_cell.size()) * qpe, 1});
  for (uint32_t i = 0; i < out.shape[0]; i++) { out(i, 0) = per_cell[i / qpe]; }
  return out;
}

// the centroid of a cell, from its vertices
template < uint32_t dim >
vec<dim> centroid(const Mesh<> & mesh, Geometry g, uint32_t e) {
  uint32_t nv = (g == Geometry::Triangle) ? 3 : (g == Geometry::Quadrilateral) ? 4 : (g == Geometry::Tetrahedron) ? 4 : 8;
  vec<dim> c{};
  for (uint32_t j = 0; j < nv; j++) { c += load<vec<dim>>(mesh.X.data, mesh[g](e, j).index) / nv; }
  return c;
}

// the vertex nearest a point
template < uint32_t dim >
uint32_t nearest_vertex(const Mesh<> & mesh, vec<dim> x) {
  uint32_t best = 0;
  double best_d = 1e300;
  for (uint32_t i = 0; i < mesh.vert.shape[0]; i++) {
    double d = norm_squared(load<vec<dim>>(mesh.X.data, i) - x);
    if (d < best_d) { best_d = d; best = i; }
  }
  return best;
}

#ifdef __EMSCRIPTEN__
// javascript typed arrays holding copies of the data (a typed_memory_view
// alone would alias wasm memory that may move on growth)
inline emscripten::val to_f64(const std::vector<double> & v) {
  return emscripten::val::global("Float64Array").new_(emscripten::val(emscripten::typed_memory_view(v.size(), v.data())));
}
template < uint32_t rank >
inline emscripten::val to_f64(const nd::cpu_array<double, rank> & a) {
  return emscripten::val::global("Float64Array").new_(emscripten::val(emscripten::typed_memory_view(size_t(a.sz), a.data())));
}
inline emscripten::val to_u32(const std::vector<uint32_t> & v) {
  return emscripten::val::global("Uint32Array").new_(emscripten::val(emscripten::typed_memory_view(v.size(), v.data())));
}
inline emscripten::val to_i32(const std::vector<int> & v) {
  return emscripten::val::global("Int32Array").new_(emscripten::val(emscripten::typed_memory_view(v.size(), v.data())));
}
#endif

} // namespace femto
