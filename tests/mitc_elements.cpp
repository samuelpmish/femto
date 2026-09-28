// Prototype of the MITC3+ and MITC4+ shell elements as a femto Family, and their verification
// on the plate and shell problems used in the MITC papers (see docs/mitc_shells/mitc_shells.md,
// section 8.4).
//
// The three-phase paradigm, as this file exercises it:
//   E_q = mitc::evaluate(strain(u), shell);                    // phase 1: 3 x 3 strain tensors at the (in-plane x thickness) points
//   S_q = forall(material, E_q);                               // phase 2: pointwise 3D material law -> 3 x 3 stress tensors
//   r   = mitc::integrate(mitc::dot(S_q, strain(psi)), shell); // phase 3: generalized forces on the shell dofs
//   K   = mitc::integrate(mitc::dot(strain(psi), C_q, strain(psi)), shell);
// The strain tensors are expressed in a local Cartesian frame whose third axis is the director
// (mitc::local_frames returns the frames), and e_33 is zero by the shell kinematics, so the
// material law has to condense its normal stress out (mitc::PlaneStress does that for any 3D law).
// The drivers below are what src/field/evaluate.cpp and integrate_residual.cpp would need for
// this family: the same element loop, but the element kernel receives the element geometry
// (midsurface coordinates, directors, thickness) and a thickness quadrature rule.

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <functional>

#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"
#include "forall.hpp"
#include "linear_algebra/sparse_matrix.hpp"
#include "linear_algebra/sparse_direct.hpp"
#include "materials/neohookean.hpp"

using namespace femto;
using namespace fm;

namespace femto {
namespace mitc {

////////////////////////////////////////////////////////////////////////////////
// the shell domain: a surface mesh in R^3 with directors, nodal frames, a thickness and a
// thickness quadrature rule. femto's Domain supplies the element lists and the in-plane rule;
// its square Jacobian tables do not apply to a 2-manifold in R^3 and are not built.

struct ShellDomain {
  ShellDomain(const Mesh<> & mesh, MeshQuadratureRule rule, double thickness_, uint32_t num_thickness_points = 2)
      : domain(mesh, rule, false) {
    FEMTO_ASSERT(mesh.X.degree == 1, "the MITC prototype assumes straight-sided (degree 1) geometry");
    FEMTO_ASSERT(mesh.spatial_dimension == 3, "the MITC prototype expects a surface mesh in R^3");
    uint32_t nv = mesh.X.data.shape[0];
    director.resize({nv, 3});
    frame.resize({nv, 2, 3});
    thickness.resize({nv});
    for (uint32_t v = 0; v < nv; v++) { thickness(v) = thickness_; for (int i = 0; i < 3; i++) director(v, i) = 0.0; }

    // directors: area-weighted average of the element normals, in the orientation of the elements
    auto X = [&](uint32_t node) { return vec3{mesh.X.data(node, 0), mesh.X.data(node, 1), mesh.X.data(node, 2)}; };
    for (uint32_t e = 0; e < mesh.tri.shape[0]; e++) {
      uint32_t ids[3]; FiniteElement<Geometry::Triangle, Family::H1>{1}.indices(mesh.X.offsets, mesh.tri(e).data(), ids);
      vec3 n = cross(X(ids[1]) - X(ids[0]), X(ids[2]) - X(ids[0]));
      for (int k = 0; k < 3; k++) for (int i = 0; i < 3; i++) director(ids[k], i) += n[i];
    }
    for (uint32_t e = 0; e < mesh.quad.shape[0]; e++) {
      uint32_t ids[4]; FiniteElement<Geometry::Quadrilateral, Family::H1>{1}.indices(mesh.X.offsets, mesh.quad(e).data(), ids);
      vec3 n = cross(X(ids[3]) - X(ids[0]), X(ids[2]) - X(ids[1]));   // lexicographic order: the diagonals
      for (int k = 0; k < 4; k++) for (int i = 0; i < 3; i++) director(ids[k], i) += n[i];
    }
    for (uint32_t v = 0; v < nv; v++) {
      vec3 V = normalize(vec3{director(v, 0), director(v, 1), director(v, 2)});
      set_director(v, V);
    }

    // Gauss rule in the thickness coordinate z in [-1, 1]
    switch (num_thickness_points) {
      case 1: z = {0.0}; wz = {2.0}; break;
      case 2: z = {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)}; wz = {1.0, 1.0}; break;
      case 3: z = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)}; wz = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0}; break;
      default: FEMTO_ASSERT(false, "1, 2 or 3 thickness points");
    }
  }

  // set the director of a vertex, and the frame by Bathe's convention (see mitc::default_frame)
  void set_director(uint32_t v, vec3 V) {
    vec3 V1, V2;
    default_frame(V, V1, V2);
    set_director(v, V, V1, V2);
  }
  void set_director(uint32_t v, vec3 V, vec3 V1, vec3 V2) {
    for (int i = 0; i < 3; i++) { director(v, i) = V[i]; frame(v, 0, i) = V1[i]; frame(v, 1, i) = V2[i]; }
  }

  // points are enumerated (in-plane quadrature point, thickness point), thickness fastest
  uint32_t num_points() const { return total(domain.num_qpts) * uint32_t(z.size()); }

  Domain<> domain;
  nd::cpu_array<double, 2> director;   // (vertices, 3), unit
  nd::cpu_array<double, 3> frame;      // (vertices, 2, 3): V1, V2
  nd::cpu_array<double, 1> thickness;  // (vertices)
  std::vector<double> z, wz;
};

////////////////////////////////////////////////////////////////////////////////
// prototype drivers

// visit every triangle and quadrilateral of the shell: the element, its geometry, the dof node
// ids of a Family::MITC space with the given offsets, the in-plane quadrature points and
// weights in femto's ordering, and the global index of the element's first in-plane point
template < typename F >
void foreach_element(const ShellDomain & shell, const GeometryInfo & offsets, F && fn) {
  const Domain<> & domain = shell.domain;
  const Mesh<> & mesh = domain.mesh;
  uint32_t q_offset = 0;
  foreach_geometry([&](auto geom_tag) {
    constexpr Geometry geom = decltype(geom_tag){};
    if constexpr (geom == Geometry::Triangle || geom == Geometry::Quadrilateral) {
      nd::view<const int> elements = domain.active_elements[geom];
      if (elements.size() == 0) return;
      nd::view<const Connection, 2> conn = mesh[geom];
      FiniteElement<geom, Family::H1> X_el{1};
      FiniteElement<geom, Family::MITC> el{1};
      constexpr uint32_t nv = el.num_vertices;

      std::vector<vec2> xi;
      std::vector<double> w;
      if constexpr (geom == Geometry::Triangle) {
        const auto & r = domain.rule.tri;
        for (uint32_t i = 0; i < r.points.shape[0]; i++) { xi.push_back({r.points(i, 0), r.points(i, 1)}); w.push_back(r.weights(i)); }
      } else {
        const auto & r = domain.rule.quad;   // compact: a 1D rule, tensor product implied
        uint32_t q = r.points.shape[0];
        for (uint32_t iy = 0; iy < q; iy++)
          for (uint32_t ix = 0; ix < q; ix++) { xi.push_back({r.points(ix, 0), r.points(iy, 0)}); w.push_back(r.weights(ix) * r.weights(iy)); }
      }

      uint32_t n = el.num_nodes();
      std::vector<uint32_t> Xids(nv), ids(n);
      ElementGeometry g;
      g.nv = nv; g.n = n;
      for (uint32_t e = 0; e < elements.size(); e++) {
        const Connection * c = conn(elements(e)).data();
        X_el.indices(mesh.X.offsets, c, Xids.data());
        el.indices(offsets, c, ids.data());
        vec3 Vsum{};
        double tsum = 0.0;
        for (uint32_t k = 0; k < nv; k++) {
          uint32_t v = Xids[k];
          double half = 0.5 * shell.thickness(v);
          for (int i = 0; i < 3; i++) {
            g.X[k][i] = mesh.X.data(v, i);
            g.Vh[k][i] = half * shell.director(v, i);
            g.d1[k][i] = half * shell.frame(v, 0, i);
            g.d2[k][i] = half * shell.frame(v, 1, i);
            Vsum[i] += shell.director(v, i);
          }
          tsum += shell.thickness(v);
        }
        if (n > nv) {   // the MITC3+ bubble: the averaged director, mean thickness, default frame
          vec3 V = normalize(Vsum), V1, V2;
          default_frame(V, V1, V2);
          g.d1[nv] = (0.5 * tsum / nv) * V1;
          g.d2[nv] = (0.5 * tsum / nv) * V2;
        }
        fn(el, g, ids, xi, w, q_offset + e * uint32_t(xi.size()));
      }
      q_offset += elements.size() * xi.size();
    }
  });
}

// phase 1: strain tensors (points, 3, 3) in the local frames, from the strain operator at every point
inline nd::cpu_array<double, 3> evaluate(const FieldOp<DerivedQuantity::STRAIN, Family::MITC> & op, const ShellDomain & shell) {
  const Field<Family::MITC> & u = op.field;
  uint32_t nz = shell.z.size();
  nd::cpu_array<double, 3> E({shell.num_points(), 3, 3});
  for (uint32_t q = 0; q < E.shape[0]; q++) for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) E(q, a, b) = 0.0;
  foreach_element(shell, u.offsets, [&](const auto & el, const ElementGeometry & g, const std::vector<uint32_t> & ids,
                                        const std::vector<vec2> & xi, const std::vector<double> &, uint32_t q0) {
    uint32_t n = ids.size();
    nd::cpu_array<double, 2> B({ntensor, ncomp * n});
    for (uint32_t k = 0; k < xi.size(); k++)
      for (uint32_t iz = 0; iz < nz; iz++) {
        el.strain_matrix(xi[k], shell.z[iz], g, B);
        uint32_t q = (q0 + k) * nz + iz;
        for (uint32_t a = 0; a < ntensor; a++) {
          double v = 0.0;
          for (uint32_t j = 0; j < ncomp * n; j++) v += B(a, j) * u.data(ids[j / ncomp], j % ncomp);
          E(q, a / 3, a % 3) = v;
        }
      }
  });
  return E;
}

// the local frames Q = [e_1 e_2 e_3] (points, 3, 3) that the strain components refer to
inline nd::cpu_array<double, 3> local_frames(const ShellDomain & shell) {
  uint32_t nz = shell.z.size();
  nd::cpu_array<double, 3> Q({shell.num_points(), 3, 3});
  Field<Family::MITC> layout = create_field<Family::MITC>(shell.domain.mesh, 1, ncomp);
  foreach_element(shell, layout.offsets, [&](const auto & el, const ElementGeometry & g, const std::vector<uint32_t> &,
                                             const std::vector<vec2> & xi, const std::vector<double> &, uint32_t q0) {
    ShapeData s;
    for (uint32_t k = 0; k < xi.size(); k++) {
      el.shape(xi[k], s);
      for (uint32_t iz = 0; iz < nz; iz++) {
        mat3 F = local_frame(jacobian(g, s, shell.z[iz]));
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) Q((q0 + k) * nz + iz, a, b) = F(a, b);
      }
    }
  });
  return Q;
}

struct ForceIntegrand { const nd::cpu_array<double, 3> & S_q; BasisFunction<Family::MITC> psi; };
struct StiffnessIntegrand { BasisFunction<Family::MITC> phi; const nd::cpu_array<double, 5> & C_q; BasisFunction<Family::MITC> psi; };

inline ForceIntegrand dot(const nd::cpu_array<double, 3> & S_q, const BasisFunctionOp<DerivedQuantity::STRAIN, Family::MITC> & psi) { return {S_q, psi.function}; }
inline StiffnessIntegrand dot(const BasisFunctionOp<DerivedQuantity::STRAIN, Family::MITC> & phi, const nd::cpu_array<double, 5> & C_q,
                              const BasisFunctionOp<DerivedQuantity::STRAIN, Family::MITC> & psi) { return {phi.function, C_q, psi.function}; }

// the volume measure at a point: w_xi w_z |det J|
inline double measure(const ElementGeometry & g, const ShapeData & s, double z) { return std::fabs(det(jacobian(g, s, z))); }

// phase 3: generalized forces r_I = sum_q w |det J| B^T : S
inline Residual<Family::MITC> integrate(const ForceIntegrand & I, const ShellDomain & shell) {
  Residual<Family::MITC> r(I.psi.space, shell.domain.mesh);
  uint32_t nz = shell.z.size();
  foreach_element(shell, r.offsets, [&](const auto & el, const ElementGeometry & g, const std::vector<uint32_t> & ids,
                                        const std::vector<vec2> & xi, const std::vector<double> & w, uint32_t q0) {
    uint32_t n = ids.size();
    nd::cpu_array<double, 2> B({ntensor, ncomp * n});
    ShapeData s;
    for (uint32_t k = 0; k < xi.size(); k++) {
      el.shape(xi[k], s);
      for (uint32_t iz = 0; iz < nz; iz++) {
        el.strain_matrix(xi[k], shell.z[iz], g, B);
        uint32_t q = (q0 + k) * nz + iz;
        double scale = w[k] * shell.wz[iz] * measure(g, s, shell.z[iz]);
        for (uint32_t j = 0; j < ncomp * n; j++) {
          double v = 0.0;
          for (uint32_t a = 0; a < ntensor; a++) v += B(a, j) * I.S_q(q, a / 3, a % 3);
          r.data(ids[j / ncomp], j % ncomp) += scale * v;
        }
      }
    }
  });
  return r;
}

// phase 3, bilinear: K = sum_q w |det J| B^T C B, assembled into the node-to-node sparsity
inline sparse_matrix<> integrate(const StiffnessIntegrand & I, const ShellDomain & shell) {
  Residual<Family::MITC> layout(I.psi.space, shell.domain.mesh);   // for the dof offsets and count
  uint32_t ndof = layout.size();
  uint32_t nz = shell.z.size();
  std::vector<triplet> triplets;
  foreach_element(shell, layout.offsets, [&](const auto & el, const ElementGeometry & g, const std::vector<uint32_t> & ids,
                                             const std::vector<vec2> & xi, const std::vector<double> & w, uint32_t q0) {
    uint32_t n = ids.size(), m = ncomp * n;
    nd::cpu_array<double, 2> B({ntensor, m}), CB({ntensor, m});
    std::vector<double> Ke(size_t(m) * m, 0.0);
    ShapeData s;
    for (uint32_t k = 0; k < xi.size(); k++) {
      el.shape(xi[k], s);
      for (uint32_t iz = 0; iz < nz; iz++) {
        el.strain_matrix(xi[k], shell.z[iz], g, B);
        uint32_t q = (q0 + k) * nz + iz;
        double scale = w[k] * shell.wz[iz] * measure(g, s, shell.z[iz]);
        for (uint32_t a = 0; a < ntensor; a++)
          for (uint32_t j = 0; j < m; j++) {
            double v = 0.0;
            for (uint32_t b = 0; b < ntensor; b++) v += I.C_q(q, a / 3, a % 3, b / 3, b % 3) * B(b, j);
            CB(a, j) = v;
          }
        for (uint32_t i = 0; i < m; i++)
          for (uint32_t j = 0; j < m; j++) {
            double v = 0.0;
            for (uint32_t a = 0; a < ntensor; a++) v += B(a, i) * CB(a, j);
            Ke[size_t(i) * m + j] += scale * v;
          }
      }
    }
    for (uint32_t i = 0; i < m; i++)
      for (uint32_t j = 0; j < m; j++)
        triplets.emplace_back(int(ncomp * ids[i / ncomp] + i % ncomp), int(ncomp * ids[j / ncomp] + j % ncomp), Ke[size_t(i) * m + j]);
  });
  sparse_matrix<> K = sparse_matrix<>::from_triplets(triplets, ndof, ndof);
  K.symmetry = Symmetry::Symmetric;
  return K;
}

// consistent nodal forces of a load q per unit midsurface area: f_I = int h_I q dA
inline vector distributed_load(const ShellDomain & shell, const FunctionSpace & space, vec3 q) {
  Residual<Family::MITC> layout(space, shell.domain.mesh);
  vector f = zeros(int(layout.size()));
  foreach_element(shell, layout.offsets, [&](const auto & el, const ElementGeometry & g, const std::vector<uint32_t> & ids,
                                             const std::vector<vec2> & xi, const std::vector<double> & w, uint32_t) {
    ShapeData s;
    for (uint32_t k = 0; k < xi.size(); k++) {
      el.shape(xi[k], s);
      mat3 J = jacobian(g, s, 0.0);
      double dA = norm(cross(vec3{J(0, 0), J(1, 0), J(2, 0)}, vec3{J(0, 1), J(1, 1), J(2, 1)}));
      for (uint32_t j = 0; j < g.nv; j++)
        for (int i = 0; i < 3; i++) f[ncomp * ids[j] + i] += w[k] * dA * s.h[j] * q[i];
    }
  });
  return f;
}

////////////////////////////////////////////////////////////////////////////////
// phase 2: material laws in the local frame (e_3 the director)

using tensor4 = mat<3, 3, mat<3, 3> >;

// isotropic linear elasticity with sigma_33 condensed out in closed form
struct PlaneStressIsotropic {
  double E, nu;
  mat3 operator()(const mat3 & eps) const {
    double c = E / (1.0 - nu * nu), G = E / (2.0 * (1.0 + nu));
    mat3 s{};
    s(0, 0) = c * (eps(0, 0) + nu * eps(1, 1));
    s(1, 1) = c * (eps(1, 1) + nu * eps(0, 0));
    s(0, 1) = s(1, 0) = 2.0 * G * eps(0, 1);
    s(0, 2) = s(2, 0) = 2.0 * G * eps(0, 2);
    s(1, 2) = s(2, 1) = 2.0 * G * eps(1, 2);
    return s;
  }
  tensor4 tangent() const {
    double c = E / (1.0 - nu * nu), G = E / (2.0 * (1.0 + nu));
    tensor4 C{};
    C[0][0][0][0] = C[1][1][1][1] = c;
    C[0][0][1][1] = C[1][1][0][0] = c * nu;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        if (a == b || (a == 2 && b == 2)) continue;
        C[a][b][a][b] += G; C[a][b][b][a] += G;   // sigma_ab = 2 G eps_ab for a != b
      }
    return C;
  }
};

// any 3D law sigma(eps) with tangent jac(eps) = d sigma / d eps, made plane stress: eps_33 is
// found so that sigma_33 = 0 (Newton), and the tangent is condensed accordingly. The law
// receives the strain as a symmetric displacement gradient, so femto's finite-strain models
// (which take du_dX) work as-is for small strains.
template < typename Model >
struct PlaneStress {
  Model model;
  double condensed_e33(mat3 eps) const {
    for (int it = 0; it < 20; it++) {
      mat3 s = model(eps);
      double s33 = s(2, 2);
      if (std::fabs(s33) < 1e-14 * (1.0 + std::fabs(s(0, 0)) + std::fabs(s(1, 1)))) break;
      double C3333 = model.jac(eps)[2][2][2][2];
      eps(2, 2) -= s33 / C3333;
    }
    return eps(2, 2);
  }
  mat3 operator()(mat3 eps) const {
    eps(2, 2) = condensed_e33(eps);
    mat3 s = model(eps);
    s(2, 2) = 0.0;
    return s;
  }
  tensor4 tangent(mat3 eps) const {
    eps(2, 2) = condensed_e33(eps);
    tensor4 C = model.jac(eps), Cps{};
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 3; c++) for (int d = 0; d < 3; d++)
      Cps[a][b][c][d] = C[a][b][c][d] - C[a][b][2][2] * C[2][2][c][d] / C[2][2][2][2];
    return Cps;
  }
};

////////////////////////////////////////////////////////////////////////////////
// helpers for the tests

// solve K u = f with prescribed dofs, reducing to the free block (SPD) and factorizing it
inline vector solve(const sparse_matrix<> & K, const vector & f, const std::vector<int> & fixed, const std::vector<double> & values) {
  int n = int(K.nrows);
  vector u = zeros(n);
  std::vector<int> map(n, -1);
  for (size_t i = 0; i < fixed.size(); i++) { map[fixed[i]] = -2; u[fixed[i]] = values[i]; }
  std::vector<int> free;
  for (int i = 0; i < n; i++) if (map[i] == -1) { map[i] = int(free.size()); free.push_back(i); }
  vector Ku = dot(K, u);
  vector rhs = zeros(int(free.size()));
  for (size_t i = 0; i < free.size(); i++) rhs[i] = f[free[i]] - Ku[free[i]];
  std::vector<triplet> triplets;
  for (int i = 0; i < n; i++) {
    if (map[i] < 0) continue;
    for (int p = K.row_ptr[i]; p < K.row_ptr[i + 1]; p++) {
      int j = K.col_ind[p];
      if (map[j] >= 0) triplets.emplace_back(map[i], map[j], K.values[p]);
    }
  }
  sparse_matrix<> Kff = sparse_matrix<>::from_triplets(triplets, free.size(), free.size());
  Kff.symmetry = Symmetry::Symmetric;
  Kff.definiteness = Definiteness::PositiveDefinite;
  auto invK = inv(Kff);
  vector uf = dot(invK, rhs);
  for (size_t i = 0; i < free.size(); i++) u[free[i]] = uf[i];
  return u;
}

// coordinates of the nodes of a Family::MITC space (vertices, then the MITC3+ bubble nodes,
// which get the centroid of their element)
inline nd::cpu_array<double, 2> node_coordinates(const Mesh<> & mesh) {
  Field<Family::MITC> u = create_field<Family::MITC>(mesh, 1, ncomp);
  nd::cpu_array<double, 2> X({u.num_nodes(), 3});
  for (uint32_t v = 0; v < mesh.X.data.shape[0]; v++) for (int i = 0; i < 3; i++) X(u.offsets.vert + v, i) = mesh.X.data(v, i);
  for (uint32_t e = 0; e < mesh.tri.shape[0]; e++) {
    uint32_t ids[3]; FiniteElement<Geometry::Triangle, Family::H1>{1}.indices(mesh.X.offsets, mesh.tri(e).data(), ids);
    for (int i = 0; i < 3; i++) X(u.offsets.tri + e, i) = (mesh.X.data(ids[0], i) + mesh.X.data(ids[1], i) + mesh.X.data(ids[2], i)) / 3.0;
  }
  return X;
}

// the translation dofs of MITC3+ bubble nodes carry no stiffness and must be constrained
inline std::vector<int> unused_dofs(const Field<Family::MITC> & u) {
  std::vector<int> dofs;
  for (uint32_t node = u.offsets.tri; node < u.num_nodes(); node++) for (int c = 0; c < 3; c++) dofs.push_back(int(ncomp * node + c));
  return dofs;
}

// the stiffness of a shell with a constant tangent (linear problems)
inline sparse_matrix<> stiffness(const ShellDomain & shell, const Field<Family::MITC> & u, const tensor4 & C) {
  BasisFunction psi(u);
  nd::cpu_array<double, 3> E_q = evaluate(strain(u), shell);
  nd::cpu_array<double, 5> C_q = forall(std::function<tensor4(const mat3 &)>([&](const mat3 &) { return C; }), E_q);
  return integrate(dot(strain(psi), C_q, strain(psi)), shell);
}

} // namespace mitc
} // namespace femto

////////////////////////////////////////////////////////////////////////////////
// meshes: surfaces in R^3

// a structured (N x M) mesh of the unit square mapped by `map`, as quadrilaterals or as triangles
// (each cell split along the same diagonal, or alternating)
static Mesh<> surface_mesh(uint32_t N, uint32_t M, std::function<vec3(double, double)> map, bool triangles, bool alternating = false) {
  nd::cpu_array<double, 2> nodes({(N + 1) * (M + 1), 3});
  for (uint32_t j = 0; j <= M; j++) for (uint32_t i = 0; i <= N; i++) {
    vec3 x = map(double(i) / N, double(j) / M);
    for (int c = 0; c < 3; c++) nodes(i + (N + 1) * j, c) = x[c];
  }
  auto id = [&](uint32_t i, uint32_t j) { return i + (N + 1) * j; };
  if (!triangles) {
    nd::cpu_array<uint32_t, 2> tris({0, 0}), quads({N * M, 4});
    for (uint32_t j = 0; j < M; j++) for (uint32_t i = 0; i < N; i++) {
      uint32_t e = i + N * j;
      quads(e, 0) = id(i, j); quads(e, 1) = id(i + 1, j); quads(e, 2) = id(i + 1, j + 1); quads(e, 3) = id(i, j + 1);
    }
    return Mesh<>::create_2D(nodes, 1, tris, quads);
  }
  nd::cpu_array<uint32_t, 2> tris({2 * N * M, 3}), quads({0, 0});
  uint32_t e = 0;
  for (uint32_t j = 0; j < M; j++) for (uint32_t i = 0; i < N; i++) {
    uint32_t a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
    if (!alternating || (i + j) % 2 == 0) { tris(e, 0) = a; tris(e, 1) = b; tris(e, 2) = c; e++; tris(e, 0) = a; tris(e, 1) = c; tris(e, 2) = d; e++; }
    else                                   { tris(e, 0) = a; tris(e, 1) = b; tris(e, 2) = d; e++; tris(e, 0) = b; tris(e, 1) = c; tris(e, 2) = d; e++; }
  }
  return Mesh<>::create_2D(nodes, 1, tris, quads);
}

static Mesh<> square_mesh(uint32_t N, double L, bool triangles, bool alternating = false) {
  return surface_mesh(N, N, [L](double u, double v) { return vec3{L * u, L * v, 0.0}; }, triangles, alternating);
}

// a flat mesh from explicit (x, y) nodes and cyclically ordered cells
static Mesh<> flat_mesh(const std::vector<vec2> & P, const std::vector<std::array<uint32_t, 4>> & Q, bool triangles) {
  nd::cpu_array<double, 2> nodes({uint32_t(P.size()), 3});
  for (uint32_t k = 0; k < P.size(); k++) { nodes(k, 0) = P[k][0]; nodes(k, 1) = P[k][1]; nodes(k, 2) = 0.0; }
  if (!triangles) {
    nd::cpu_array<uint32_t, 2> tris({0, 0}), quads({uint32_t(Q.size()), 4});
    for (uint32_t e = 0; e < Q.size(); e++) for (int k = 0; k < 4; k++) quads(e, k) = Q[e][k];
    return Mesh<>::create_2D(nodes, 1, tris, quads);
  }
  nd::cpu_array<uint32_t, 2> tris({2 * uint32_t(Q.size()), 3}), quads({0, 0});
  for (uint32_t e = 0; e < Q.size(); e++) {
    tris(2 * e, 0) = Q[e][0]; tris(2 * e, 1) = Q[e][1]; tris(2 * e, 2) = Q[e][2];
    tris(2 * e + 1, 0) = Q[e][0]; tris(2 * e + 1, 1) = Q[e][2]; tris(2 * e + 1, 2) = Q[e][3];
  }
  return Mesh<>::create_2D(nodes, 1, tris, quads);
}

// the MacNeal-Harder patch used in the MITC papers (Ko, Lee, Bathe 2017, fig. 5)
static Mesh<> patch_mesh(bool triangles) {
  std::vector<vec2> P = {{0, 0}, {10, 0}, {10, 10}, {0, 10}, {2, 2}, {8, 3}, {8, 7}, {4, 7}};
  std::vector<std::array<uint32_t, 4>> Q = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};
  return flat_mesh(P, Q, triangles);
}

// an affine patch: a 2 x 2 grid sheared into parallelograms (every element's Jacobian is constant)
static Mesh<> affine_patch_mesh(bool triangles) {
  std::vector<vec2> P;
  for (int j = 0; j <= 2; j++) for (int i = 0; i <= 2; i++) { double X = 5.0 * i, Y = 5.0 * j; P.push_back({X + 0.35 * Y, 0.8 * Y + 0.1 * X}); }
  std::vector<std::array<uint32_t, 4>> Q;
  for (uint32_t j = 0; j < 2; j++) for (uint32_t i = 0; i < 2; i++) Q.push_back({i + 3 * j, i + 1 + 3 * j, i + 1 + 3 * (j + 1), i + 3 * (j + 1)});
  return flat_mesh(P, Q, triangles);
}

// back to the grid coordinates of the affine patch
static bool on_affine_patch_boundary(vec2 x) {
  double Y = (x[1] - 0.1 * x[0]) / (0.8 - 0.035), X = x[0] - 0.35 * Y;
  return X < 1e-9 || X > 10 - 1e-9 || Y < 1e-9 || Y > 10 - 1e-9;
}

////////////////////////////////////////////////////////////////////////////////
// reference solutions

// Navier series for the hard simply supported Mindlin plate under uniform load q = 1, E = 1;
// the shear stiffness is G t (k = 1), as a three-dimensional material law gives it
static double ss_plate_center(double L, double t, double nu, double k = 1.0, int terms = 199) {
  double D = t * t * t / (12.0 * (1.0 - nu * nu)), S = k * t / (2.0 * (1.0 + nu));
  double w = 0.0;
  for (int m = 1; m <= terms; m += 2) for (int n = 1; n <= terms; n += 2) {
    double lam = M_PI * M_PI * (m * m + n * n) / (L * L);
    double sgn = ((m - 1) / 2 + (n - 1) / 2) % 2 == 0 ? 1.0 : -1.0;
    w += 16.0 / (M_PI * M_PI * m * n) * sgn * (1.0 / (D * lam * lam) + 1.0 / (S * lam));
  }
  return w;
}

////////////////////////////////////////////////////////////////////////////////
// a plate solve on a given flat mesh

struct PlateResult { vector u; double w_center; };

enum class Support { HardSimple, Clamped };

// solves the bending problem (in-plane dofs fixed everywhere, load q = 1 along +z) and returns the
// deflection at the node closest to the given point
static PlateResult solve_plate(const Mesh<> & mesh, double E, double nu, double t, Support support, vec2 probe,
                               std::function<bool(vec2)> on_boundary, std::function<bool(vec2, int)> boundary_free_rotation = nullptr) {
  Field<Family::MITC> u = create_field<Family::MITC>(mesh, 1, mitc::ncomp);
  mitc::ShellDomain shell(mesh, MeshQuadratureRule(3), t);
  nd::cpu_array<double, 2> X = mitc::node_coordinates(mesh);

  sparse_matrix<> K = mitc::stiffness(shell, u, mitc::PlaneStressIsotropic{E, nu}.tangent());
  vector f = mitc::distributed_load(shell, FunctionSpace{Family::MITC, 1, mitc::ncomp}, vec3{0.0, 0.0, 1.0});

  std::vector<int> fixed = mitc::unused_dofs(u);
  for (uint32_t node = 0; node < u.num_nodes(); node++) {
    fixed.push_back(int(mitc::ncomp * node + 0));
    fixed.push_back(int(mitc::ncomp * node + 1));
    if (node >= u.offsets.tri) continue;   // bubble nodes: rotations free
    vec2 x{X(node, 0), X(node, 1)};
    if (on_boundary(x)) {
      fixed.push_back(int(mitc::ncomp * node + 2));
      for (int c = 3; c < 5; c++) {
        bool fix = (support == Support::Clamped) || (boundary_free_rotation && !boundary_free_rotation(x, c));
        if (fix) fixed.push_back(int(mitc::ncomp * node + c));
      }
    }
  }
  std::sort(fixed.begin(), fixed.end()); fixed.erase(std::unique(fixed.begin(), fixed.end()), fixed.end());
  std::vector<double> values(fixed.size(), 0.0);
  vector sol = mitc::solve(K, f, fixed, values);

  uint32_t best = 0; double dmin = 1e300;
  for (uint32_t node = 0; node < u.offsets.tri; node++) {
    double d = std::hypot(X(node, 0) - probe[0], X(node, 1) - probe[1]);
    if (d < dmin) { dmin = d; best = node; }
  }
  return {sol, sol[mitc::ncomp * best + 2]};
}

static auto unit_square_boundary = [](vec2 x) { double e = 1e-9; return x[0] < e || x[0] > 1 - e || x[1] < e || x[1] > 1 - e; };
// hard simple support: the in-plane displacement tangential to the edge vanishes through the
// thickness, so the rotation dof whose axis of displacement is the tangent is fixed (on a flat
// plate with director e_z the frame is V1 = e_x, V2 = e_y: alpha moves along x, beta along y)
static auto unit_square_hard_ss = [](vec2 x, int c) {
  double e = 1e-9;
  bool on_x = x[0] < e || x[0] > 1 - e, on_y = x[1] < e || x[1] > 1 - e;   // edges x = const (tangent y), y = const (tangent x)
  bool free = true;
  if (on_x && c == 4) free = false;
  if (on_y && c == 3) free = false;
  return free;
};

////////////////////////////////////////////////////////////////////////////////
// tests

// evaluate() returns one 3 x 3 tensor per (in-plane point, thickness point), and the frames it
// refers to are orthonormal with the third axis along the director
TEST(MITC, evaluate_returns_one_tensor_per_thickness_point) {
  for (bool tri : {false, true}) {
    Mesh<> mesh = patch_mesh(tri);
    for (uint32_t nz : {1u, 2u, 3u}) {
      mitc::ShellDomain shell(mesh, MeshQuadratureRule(3), 0.1, nz);
      Field<Family::MITC> u = create_field<Family::MITC>(mesh, 1, mitc::ncomp);
      nd::cpu_array<double, 3> E_q = mitc::evaluate(strain(u), shell);
      EXPECT_EQ(E_q.shape[0], total(shell.domain.num_qpts) * nz);
      EXPECT_EQ(E_q.shape[1], 3u); EXPECT_EQ(E_q.shape[2], 3u);
      nd::cpu_array<double, 3> Q = mitc::local_frames(shell);
      double err = 0.0;
      for (uint32_t q = 0; q < Q.shape[0]; q++) {
        mat3 F; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) F(a, b) = Q(q, a, b);
        mat3 I = dot(transpose(F), F);
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) err = std::max(err, std::fabs(I(a, b) - (a == b)));
        err = std::max(err, std::fabs(F(2, 2) - 1.0));   // flat patch in the z = 0 plane: e_3 = e_z
      }
      EXPECT_LT(err, 1e-12);
    }
  }
}

// on a flat element the MITC4+ membrane strain equals the displacement-based one (Ko, Lee, Bathe
// 2017, eq. 22): the characteristic-vector construction is verified against that identity, at
// every thickness coordinate since the bending part is displacement-based
TEST(MITC, mitc4plus_membrane_reduces_to_displacement_based_when_flat) {
  FiniteElement<Geometry::Quadrilateral, Family::MITC> el{1};
  mitc::ElementGeometry g; g.nv = 4; g.n = 4;
  double P[4][2] = {{0, 0}, {1.3, 0.2}, {-0.1, 0.9}, {1.0, 1.1}};   // lexicographic node order
  for (int k = 0; k < 4; k++) {
    g.X[k] = vec3{P[k][0], P[k][1], 0.0};
    g.Vh[k] = vec3{0.0, 0.0, 0.05}; g.d1[k] = vec3{0.05, 0.0, 0.0}; g.d2[k] = vec3{0.0, 0.05, 0.0};
  }
  nd::cpu_array<double, 2> B({9, 20}), Bdi({9, 20});
  mitc::ShapeData s;
  mitc::dense cov;
  double err = 0.0;
  for (double z : {-1.0, 0.0, 0.7})
    for (vec2 xi : {vec2{0.2, 0.7}, vec2{0.9, 0.1}, vec2{0.5, 0.5}}) {
      el.strain_matrix(xi, z, g, B);
      el.shape(xi, s);
      mitc::covariant_rows(g, s, z, cov);
      mitc::to_local(cov, mitc::jacobian(g, s, z), Bdi);
      for (int a : {0, 1, 3, 4}) for (int j = 0; j < 20; j++) err = std::max(err, std::fabs(B(a, j) - Bdi(a, j)));
    }
  EXPECT_LT(err, 1e-12);
}

// constant-strain patch tests on the distorted patch (membrane, bending, shear), in two forms:
//  (i) reproduction: every node takes the exact field; the strain tensor at every point (in its
//      local frame) must be the prescribed one (the consistency requirement of the tying; the
//      quadratic bending deflection is not in the space, so its shear rows are not checked);
// (ii) equilibrium: boundary nodes prescribed, interior solved, for the membrane and bending
//      states (both elements pass both, on the affine and the distorted patch). A constant shear
//      with zero moment is not an equilibrium state without distributed couples.
static void patch_test(bool triangles, bool affine) {
  Mesh<> mesh = affine ? affine_patch_mesh(triangles) : patch_mesh(triangles);
  Field<Family::MITC> u = create_field<Family::MITC>(mesh, 1, mitc::ncomp);
  double t = 0.1;
  mitc::ShellDomain shell(mesh, MeshQuadratureRule(3), t);
  nd::cpu_array<double, 2> X = mitc::node_coordinates(mesh);
  nd::cpu_array<double, 3> Q = mitc::local_frames(shell);
  sparse_matrix<> K = mitc::stiffness(shell, u, mitc::PlaneStressIsotropic{1000.0, 0.3}.tangent());
  vector f = zeros(int(K.nrows));
  const char * label = triangles ? "MITC3+" : "MITC4+";

  // exact fields and their global strain tensor at height z (kinematics: u_inplane = z (alpha e_x + beta e_y))
  struct Mode { const char * name; std::function<void(vec2, double *)> exact; std::function<mat3(double)> eps; };
  const double a = 1e-3;
  std::vector<Mode> modes = {
    {"membrane", [=](vec2 x, double * d) { d[0] = a * (x[0] + 0.5 * x[1]); d[1] = a * (0.25 * x[0] + 2.0 * x[1]); d[2] = d[3] = d[4] = 0; },
     [=](double) { mat3 e{}; e(0, 0) = a; e(1, 1) = 2 * a; e(0, 1) = e(1, 0) = 0.375 * a; return e; }},
    {"bending", [=](vec2 x, double * d) { d[0] = d[1] = 0; d[2] = a * 0.5 * (x[0] * x[0] + 2.0 * x[0] * x[1] + 3.0 * x[1] * x[1]);
                                          d[3] = -a * (x[0] + x[1]); d[4] = -a * (x[0] + 3.0 * x[1]); },
     [=](double z) { mat3 e{}; e(0, 0) = -z * a; e(1, 1) = -3 * z * a; e(0, 1) = e(1, 0) = -z * a; return e; }},
    {"shear", [=](vec2 x, double * d) { d[0] = d[1] = 0; d[2] = a * (2.0 * x[0] + x[1]); d[3] = a * (-2.0 + 0.7); d[4] = a * (-1.0 + 0.4); },
     [=](double) { mat3 e{}; e(0, 2) = e(2, 0) = 0.35 * a; e(1, 2) = e(2, 1) = 0.2 * a; return e; }},
  };
  uint32_t nz = shell.z.size();
  for (const Mode & mode : modes) {
    bool bending = std::string(mode.name) == "bending";
    // (i) reproduction (bubble nodes take the exact rotation at their centroid, which makes the
    //     enriched rotation field reproduce a linear one)
    for (uint32_t node = 0; node < u.num_nodes(); node++) {
      double d[5]; mode.exact(vec2{X(node, 0), X(node, 1)}, d);
      for (int c = 0; c < 5; c++) u.data(node, c) = d[c];
    }
    nd::cpu_array<double, 3> E_q = mitc::evaluate(strain(u), shell);
    double rep_err = 0.0;
    for (uint32_t q = 0; q < E_q.shape[0]; q++) {
      double z = 0.5 * t * shell.z[q % nz];
      mat3 F; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) F(i, j) = Q(q, i, j);
      mat3 expected = dot(transpose(F), dot(mode.eps(z), F));
      for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        if (bending && (i == 2 || j == 2)) continue;
        rep_err = std::max(rep_err, std::fabs(E_q(q, i, j) - expected(i, j)) / a);
      }
    }
    EXPECT_LT(rep_err, 1e-9) << label << " " << mode.name << " reproduction";

    // (ii) equilibrium
    if (std::string(mode.name) == "shear") continue;
    std::vector<int> fixed = mitc::unused_dofs(u);
    std::vector<double> values(fixed.size(), 0.0);
    uint32_t nreal = u.offsets.tri;
    for (uint32_t node = 0; node < nreal; node++) {
      vec2 x{X(node, 0), X(node, 1)};
      bool boundary = affine ? on_affine_patch_boundary(x) : (x[0] < 1e-9 || x[0] > 10 - 1e-9 || x[1] < 1e-9 || x[1] > 10 - 1e-9);
      if (!boundary) continue;
      double d[5]; mode.exact(x, d);
      for (int c = 0; c < 5; c++) { fixed.push_back(int(mitc::ncomp * node + c)); values.push_back(d[c]); }
    }
    vector sol = mitc::solve(K, f, fixed, values);
    double node_err = 0.0;
    for (uint32_t node = 0; node < nreal; node++) {
      double d[5]; mode.exact(vec2{X(node, 0), X(node, 1)}, d);
      for (int c = 0; c < 5; c++) node_err = std::max(node_err, std::fabs(sol[mitc::ncomp * node + c] - d[c]) / a);
    }
    for (uint32_t i = 0; i < u.size(); i++) u.data(i / mitc::ncomp, i % mitc::ncomp) = sol[i];
    nd::cpu_array<double, 3> E_s = mitc::evaluate(strain(u), shell);
    double strain_err = 0.0;
    for (uint32_t q = 0; q < E_s.shape[0]; q++) {
      double z = 0.5 * t * shell.z[q % nz];
      mat3 F; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) F(i, j) = Q(q, i, j);
      mat3 expected = dot(transpose(F), dot(mode.eps(z), F));
      for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) strain_err = std::max(strain_err, std::fabs(E_s(q, i, j) - expected(i, j)) / a);
    }
    printf("  %-7s %-8s equilibrium patch test (%s patch): interior node error %.2e, strain error %.2e (relative to 1e-3)\n",
           label, mode.name, affine ? "affine" : "distorted", node_err, strain_err);
    EXPECT_LT(node_err, 1e-9) << label << " " << mode.name;
    EXPECT_LT(strain_err, 1e-9) << label << " " << mode.name;
  }
}

TEST(MITC, patch_tests_MITC4plus) { patch_test(false, true); patch_test(false, false); }
TEST(MITC, patch_tests_MITC3plus) { patch_test(true, true); patch_test(true, false); }

// rank of a single free element: exactly the six rigid body modes are zero-energy
static uint32_t nullity(std::vector<double> A, uint32_t n, double tol) {
  uint32_t rank = 0;
  double amax = 0; for (double v : A) amax = std::max(amax, std::fabs(v));
  for (uint32_t c = 0; c < n && rank < n; c++) {
    uint32_t piv = rank; double best = 0;
    for (uint32_t r = rank; r < n; r++) if (std::fabs(A[r * n + c]) > best) { best = std::fabs(A[r * n + c]); piv = r; }
    if (best < tol * amax) continue;
    for (uint32_t j = 0; j < n; j++) std::swap(A[rank * n + j], A[piv * n + j]);
    for (uint32_t r = rank + 1; r < n; r++) {
      double f = A[r * n + c] / A[rank * n + c];
      for (uint32_t j = c; j < n; j++) A[r * n + j] -= f * A[rank * n + j];
    }
    rank++;
  }
  return n - rank;
}

static void rank_test(bool triangles, uint32_t expected_nullity) {
  // a single distorted element, warped out of plane for the quadrilateral
  Mesh<> mesh = triangles ? flat_mesh({{0, 0}, {1.1, 0.1}, {0.3, 0.8}, {0, 0}}, {{0, 1, 2, 3}}, false)
                          : flat_mesh({{0, 0}, {1.2, 0.1}, {0.9, 1.1}, {-0.1, 0.8}}, {{0, 1, 2, 3}}, false);
  if (triangles) {
    nd::cpu_array<double, 2> nodes({3, 3}); double P[3][3] = {{0, 0, 0}, {1.1, 0.1, 0.05}, {0.3, 0.8, -0.1}};
    for (int k = 0; k < 3; k++) for (int i = 0; i < 3; i++) nodes(k, i) = P[k][i];
    nd::cpu_array<uint32_t, 2> tris({1, 3}), quads({0, 0}); tris(0, 0) = 0; tris(0, 1) = 1; tris(0, 2) = 2;
    mesh = Mesh<>::create_2D(nodes, 1, tris, quads);
  } else {
    nd::cpu_array<double, 2> nodes({4, 3}); double P[4][3] = {{0, 0, 0}, {1.2, 0.1, 0.1}, {0.9, 1.1, -0.05}, {-0.1, 0.8, 0.08}};
    for (int k = 0; k < 4; k++) for (int i = 0; i < 3; i++) nodes(k, i) = P[k][i];
    nd::cpu_array<uint32_t, 2> tris({0, 0}), quads({1, 4}); for (int k = 0; k < 4; k++) quads(0, k) = k;
    mesh = Mesh<>::create_2D(nodes, 1, tris, quads);
  }
  Field<Family::MITC> u = create_field<Family::MITC>(mesh, 1, mitc::ncomp);
  mitc::ShellDomain shell(mesh, MeshQuadratureRule(3), 0.1);
  sparse_matrix<> K = mitc::stiffness(shell, u, mitc::PlaneStressIsotropic{1.0, 0.3}.tangent());
  std::vector<int> skip = mitc::unused_dofs(u);
  std::vector<int> keep;
  for (uint32_t i = 0; i < K.nrows; i++) if (std::find(skip.begin(), skip.end(), int(i)) == skip.end()) keep.push_back(i);
  uint32_t n = keep.size();
  std::vector<double> A(size_t(n) * n, 0.0);
  for (uint32_t i = 0; i < n; i++)
    for (int q = K.row_ptr[keep[i]]; q < K.row_ptr[keep[i] + 1]; q++) {
      auto it = std::find(keep.begin(), keep.end(), K.col_ind[q]);
      if (it != keep.end()) A[size_t(i) * n + (it - keep.begin())] = K.values[q];
    }
  EXPECT_EQ(nullity(A, n, 1e-10), expected_nullity) << (triangles ? "MITC3+" : "MITC4+");
}

TEST(MITC, rank_MITC4plus) { rank_test(false, 6); }
TEST(MITC, rank_MITC3plus) { rank_test(true, 6); }

// the residual of the three-phase evaluation equals K u: integrate() is the adjoint of evaluate()
TEST(MITC, residual_is_adjoint_of_strain_evaluation) {
  for (bool tri : {false, true}) {
    Mesh<> mesh = surface_mesh(3, 2, [](double a, double b) { return vec3{2.0 * a, b, 0.3 * std::sin(3.0 * a) * b}; }, tri);   // a curved surface
    Field<Family::MITC> u = create_field<Family::MITC>(mesh, 1, mitc::ncomp);
    BasisFunction psi(u);
    mitc::ShellDomain shell(mesh, MeshQuadratureRule(3), 0.2);
    mitc::PlaneStressIsotropic law{100.0, 0.3};
    for (uint32_t i = 0; i < u.size(); i++) u.data(i / mitc::ncomp, i % mitc::ncomp) = std::sin(1.7 * i) + 0.3 * std::cos(0.4 * i * i);
    nd::cpu_array<double, 3> E_q = mitc::evaluate(strain(u), shell);
    nd::cpu_array<double, 3> S_q = forall(std::function<mat3(const mat3 &)>([&](const mat3 & E) { return law(E); }), E_q);
    Residual<Family::MITC> r = mitc::integrate(mitc::dot(S_q, strain(psi)), shell);
    nd::cpu_array<double, 5> C_q = forall(std::function<mitc::tensor4(const mat3 &)>([&](const mat3 &) { return law.tangent(); }), E_q);
    sparse_matrix<> K = mitc::integrate(mitc::dot(strain(psi), C_q, strain(psi)), shell);
    vector Ku = dot(K, u.v());
    double err = 0.0, scale = 0.0;
    for (uint32_t i = 0; i < u.size(); i++) { err = std::max(err, std::fabs(Ku[i] - r.v()[i])); scale = std::max(scale, std::fabs(Ku[i])); }
    EXPECT_LT(err, 1e-12 * scale) << (tri ? "MITC3+" : "MITC4+");
    // and K is symmetric
    double asym = 0.0;
    for (uint32_t i = 0; i < K.nrows; i++) for (int q = K.row_ptr[i]; q < K.row_ptr[i + 1]; q++) {
      int j = K.col_ind[q]; double kji = 0.0;
      for (int s = K.row_ptr[j]; s < K.row_ptr[j + 1]; s++) if (K.col_ind[s] == int(i)) kji = K.values[s];
      asym = std::max(asym, std::fabs(K.values[q] - kji));
    }
    EXPECT_LT(asym, 1e-12 * scale);
  }
}

// simply supported square plate under uniform load: thickness-independent accuracy against the
// Navier series (the shear-locking test of the MITC papers)
TEST(MITC, simply_supported_plate_no_locking) {
  double L = 1.0, nu = 0.3;
  struct Case { const char * name; bool tri; uint32_t N; double tol; };
  for (const Case & c : std::vector<Case>{{"MITC4+", false, 8, 0.01}, {"MITC3+", true, 8, 0.05}}) {
    std::vector<double> ratios;
    for (double t : {1e-2, 1e-3, 1e-4}) {
      Mesh<> mesh = square_mesh(c.N, L, c.tri);   // all diagonals alike: the hard pattern for MITC3
      PlateResult res = solve_plate(mesh, 1.0, nu, t, Support::HardSimple, vec2{0.5, 0.5}, unit_square_boundary, unit_square_hard_ss);
      double ratio = res.w_center / ss_plate_center(L, t, nu);
      ratios.push_back(ratio);
      printf("  %-7s N=%2u  t/L=%.0e  w_c / w_series = %.6f\n", c.name, c.N, t, ratio);
      EXPECT_NEAR(ratio, 1.0, c.tol) << c.name << " t=" << t;
    }
    // no locking: the ratio must not drift with the thickness
    EXPECT_NEAR(ratios[2] / ratios[0], 1.0, 3e-3) << c.name;
  }
}

// mesh convergence at t/L = 1/1000: second order in the deflection
TEST(MITC, simply_supported_plate_convergence) {
  double L = 1.0, nu = 0.3, t = 1e-3;
  double wref = ss_plate_center(L, t, nu);
  for (bool tri : {false, true}) {
    std::vector<double> errs;
    for (uint32_t N : {4u, 8u, 16u}) {
      Mesh<> mesh = square_mesh(N, L, tri);
      PlateResult res = solve_plate(mesh, 1.0, nu, t, Support::HardSimple, vec2{0.5, 0.5}, unit_square_boundary, unit_square_hard_ss);
      errs.push_back(std::fabs(res.w_center / wref - 1.0));
      printf("  %-7s N=%2u  relative error %.3e\n", tri ? "MITC3+" : "MITC4+", N, errs.back());
    }
    double order = std::log2(errs[1] / errs[2]);
    printf("  %-7s observed order %.2f\n", tri ? "MITC3+" : "MITC4+", order);
    EXPECT_GT(order, 1.7);
  }
}

// fully clamped square plate under uniform load (Ko, Lee, Bathe 2017, fig. 6): thin-plate reference
// w_c = 0.0012653 q L^4 / D
TEST(MITC, clamped_plate) {
  double L = 1.0, nu = 0.3, t = 1e-3;
  double D = t * t * t / (12.0 * (1.0 - nu * nu)), wref = 0.0012653 / D;
  struct Case { const char * name; bool tri; uint32_t N; double tol; };
  for (const Case & c : std::vector<Case>{{"MITC4+", false, 16, 0.02}, {"MITC3+", true, 16, 0.05}}) {
    Mesh<> mesh = square_mesh(c.N, L, c.tri, true);
    PlateResult res = solve_plate(mesh, 1.0, nu, t, Support::Clamped, vec2{0.5, 0.5}, unit_square_boundary);
    printf("  %-7s N=%2u  clamped w_c / w_ref = %.5f\n", c.name, c.N, res.w_center / wref);
    EXPECT_NEAR(res.w_center / wref, 1.0, c.tol) << c.name;
  }
}

// a three-dimensional material used in the shell: the neo-Hookean model's tangent at zero strain,
// condensed to plane stress by the adapter, gives the same plate as the closed-form section law
TEST(MITC, three_dimensional_material_through_plane_stress_adapter) {
  double E = 1.0, nu = 0.3, t = 1e-2, L = 1.0;
  double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), mu = E / (2.0 * (1.0 + nu));
  mitc::PlaneStress<NeoHookeanModel> law{NeoHookeanModel{lambda, mu}};
  mat3 zero{};
  mitc::tensor4 C3 = law.tangent(zero), Cps = mitc::PlaneStressIsotropic{E, nu}.tangent();
  double err = 0.0;
  for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 3; c++) for (int d = 0; d < 3; d++)
    err = std::max(err, std::fabs(C3[a][b][c][d] - Cps[a][b][c][d]));
  EXPECT_LT(err, 1e-12);
  // and the condensed stress of a small strain state has no normal component and the right in-plane one
  mat3 eps{}; eps(0, 0) = 1e-4; eps(1, 1) = -2e-4; eps(0, 1) = eps(1, 0) = 5e-5; eps(0, 2) = eps(2, 0) = 3e-5;
  mat3 s3 = law(eps), sps = mitc::PlaneStressIsotropic{E, nu}(eps);
  EXPECT_LT(std::fabs(s3(2, 2)), 1e-15);
  for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) EXPECT_NEAR(s3(a, b), sps(a, b), 1e-7);   // equal to O(eps^2)

  Mesh<> mesh = square_mesh(8, L, false);
  Field<Family::MITC> u = create_field<Family::MITC>(mesh, 1, mitc::ncomp);
  mitc::ShellDomain shell(mesh, MeshQuadratureRule(3), t);
  sparse_matrix<> K3 = mitc::stiffness(shell, u, C3), Kps = mitc::stiffness(shell, u, Cps);
  double kerr = 0.0, kmax = 0.0;
  for (size_t i = 0; i < K3.values.size(); i++) { kerr = std::max(kerr, std::fabs(K3.values[i] - Kps.values[i])); kmax = std::max(kmax, std::fabs(Kps.values[i])); }
  EXPECT_LT(kerr, 1e-12 * kmax);
}

////////////////////////////////////////////////////////////////////////////////
// curved shells: the benchmark problems of the MITC papers (MacNeal, Harder 1985; Belytschko et al. 1985)

struct ShellResult { double value; };

// a generic linear shell solve: nodal frames from `frame` (director, V1, V2 at a vertex),
// dofs fixed by `fixed_dofs(vertex coordinates, dof) -> bool`, point loads and a distributed load
static double solve_shell(const Mesh<> & mesh, double E, double nu, double t,
                          std::function<void(vec3, vec3 &, vec3 &, vec3 &)> frame,
                          std::function<bool(vec3, int)> fixed_dofs,
                          const std::vector<std::pair<vec3, vec3>> & point_loads, vec3 body_load,
                          vec3 probe, int probe_dof) {
  Field<Family::MITC> u = create_field<Family::MITC>(mesh, 1, mitc::ncomp);
  mitc::ShellDomain shell(mesh, MeshQuadratureRule(3), t);
  nd::cpu_array<double, 2> X = mitc::node_coordinates(mesh);
  for (uint32_t v = 0; v < mesh.X.data.shape[0]; v++) {
    vec3 x{X(v, 0), X(v, 1), X(v, 2)}, V, V1, V2;
    frame(x, V, V1, V2);
    shell.set_director(v, V, V1, V2);
  }
  sparse_matrix<> K = mitc::stiffness(shell, u, mitc::PlaneStressIsotropic{E, nu}.tangent());
  vector f = mitc::distributed_load(shell, FunctionSpace{Family::MITC, 1, mitc::ncomp}, body_load);
  auto nearest = [&](vec3 p) {
    uint32_t best = 0; double dmin = 1e300;
    for (uint32_t v = 0; v < u.offsets.tri; v++) {
      double d = norm(vec3{X(v, 0), X(v, 1), X(v, 2)} - p);
      if (d < dmin) { dmin = d; best = v; }
    }
    return best;
  };
  for (const auto & [p, F] : point_loads) { uint32_t v = nearest(p); for (int i = 0; i < 3; i++) f[mitc::ncomp * v + i] += F[i]; }
  std::vector<int> fixed = mitc::unused_dofs(u);
  for (uint32_t v = 0; v < u.offsets.tri; v++) {
    vec3 x{X(v, 0), X(v, 1), X(v, 2)};
    for (int c = 0; c < 5; c++) if (fixed_dofs(x, c)) fixed.push_back(int(mitc::ncomp * v + c));
  }
  std::vector<double> values(fixed.size(), 0.0);
  vector sol = mitc::solve(K, f, fixed, values);
  return sol[mitc::ncomp * nearest(probe) + probe_dof];
}

static bool near(double a, double b) { return std::fabs(a - b) < 1e-7 * (1.0 + std::fabs(b)); }

// Scordelis-Lo roof: a cylindrical panel (R = 25, L = 50, t = 0.25, E = 4.32e8, nu = 0, half-angle
// 40 degrees, axis z, gravity -y) on rigid end diaphragms with free straight edges, under its own
// weight 90 per unit area. Reference vertical deflection at the midpoint of a free edge: 0.3024.
// A quarter is modelled with symmetry planes z = L/2 and x = 0.
static double scordelis_lo(uint32_t N, bool triangles) {
  double R = 25.0, L = 50.0, t = 0.25, E = 4.32e8, nu = 0.0, phi = 40.0 * M_PI / 180.0;
  Mesh<> mesh = surface_mesh(N, N, [&](double a, double b) { double th = phi * a; return vec3{R * std::sin(th), R * std::cos(th), 0.5 * L * b}; }, triangles);
  auto frame = [&](vec3 x, vec3 & V, vec3 & V1, vec3 & V2) { V = normalize(vec3{x[0], x[1], 0.0}); V1 = vec3{0.0, 0.0, 1.0}; V2 = cross(V, V1); };
  auto fixed = [&](vec3 x, int c) {
    if (near(x[2], 0.0) && (c == 0 || c == 1)) return true;                 // diaphragm: u_x = u_y = 0
    if (near(x[2], 0.5 * L) && (c == 2 || c == 3)) return true;             // symmetry z = L/2: u_z = 0, no rotation about the in-plane axis (alpha, arm V1 = e_z)
    if (near(x[0], 0.0) && (c == 0 || c == 4)) return true;                 // symmetry x = 0: u_x = 0, beta = 0 (arm V2 = e_x there)
    return false;
  };
  vec3 probe{R * std::sin(phi), R * std::cos(phi), 0.5 * L};
  return -solve_shell(mesh, E, nu, t, frame, fixed, {}, vec3{0.0, -90.0, 0.0}, probe, 1) / 0.3024;
}

// Pinched cylinder with end diaphragms (R = 300, L = 600, t = 3, E = 3e6, nu = 0.3), two opposite
// radial point loads P = 1 at the center. Reference radial deflection under the load: 1.8248e-5.
// An octant is modelled with the load P/4 at its corner.
static double pinched_cylinder(uint32_t N, bool triangles) {
  double R = 300.0, L = 600.0, t = 3.0, E = 3e6, nu = 0.3;
  Mesh<> mesh = surface_mesh(N, N, [&](double a, double b) { double th = 0.5 * M_PI * a; return vec3{R * std::sin(th), R * std::cos(th), 0.5 * L * b}; }, triangles);
  auto frame = [&](vec3 x, vec3 & V, vec3 & V1, vec3 & V2) { V = normalize(vec3{x[0], x[1], 0.0}); V1 = vec3{0.0, 0.0, 1.0}; V2 = cross(V, V1); };
  auto fixed = [&](vec3 x, int c) {
    if (near(x[2], 0.0) && (c == 0 || c == 1)) return true;                 // diaphragm
    if (near(x[2], 0.5 * L) && (c == 2 || c == 3)) return true;             // symmetry z = L/2
    if (near(x[0], 0.0) && (c == 0 || c == 4)) return true;                 // symmetry x = 0 (V2 = e_x there)
    if (near(x[1], 0.0) && (c == 1 || c == 4)) return true;                 // symmetry y = 0 (V2 = -e_y there)
    return false;
  };
  vec3 probe{R, 0.0, 0.5 * L};
  return -solve_shell(mesh, E, nu, t, frame, fixed, {{probe, vec3{-0.25, 0.0, 0.0}}}, vec3{}, probe, 0) / 1.8248e-5;
}

// Hemispherical shell with an 18 degree hole (R = 10, t = 0.04, E = 6.825e7, nu = 0.3), four
// alternating radial point loads F = 2 at the equator. Reference radial deflection under a load:
// 0.0924. A quarter is modelled (symmetry planes x = 0 and y = 0) with the half loads.
static double hemisphere(uint32_t N, bool triangles) {
  double R = 10.0, t = 0.04, E = 6.825e7, nu = 0.3, top = 72.0 * M_PI / 180.0;
  Mesh<> mesh = surface_mesh(N, N, [&](double a, double b) {
    double th = 0.5 * M_PI * a, ph = top * b;
    return vec3{R * std::cos(ph) * std::cos(th), R * std::cos(ph) * std::sin(th), R * std::sin(ph)};
  }, triangles);
  auto frame = [&](vec3 x, vec3 & V, vec3 & V1, vec3 & V2) { V = normalize(x); V1 = normalize(cross(vec3{0.0, 0.0, 1.0}, V)); V2 = cross(V, V1); };
  auto fixed = [&](vec3 x, int c) {
    if (near(x[1], 0.0) && (c == 1 || c == 3)) return true;                 // symmetry y = 0: u_y = 0, alpha = 0 (V1 = e_y there)
    if (near(x[0], 0.0) && (c == 0 || c == 3)) return true;                 // symmetry x = 0: u_x = 0, alpha = 0 (V1 = -e_x there)
    if (near(x[0], R) && near(x[2], 0.0) && c == 2) return true;            // one u_z against the free translation
    return false;
  };
  vec3 probe{R, 0.0, 0.0};
  return solve_shell(mesh, E, nu, t, frame, fixed, {{probe, vec3{1.0, 0.0, 0.0}}, {vec3{0.0, R, 0.0}, vec3{0.0, -1.0, 0.0}}}, vec3{}, probe, 0) / 0.0924;
}

// the three benchmarks, normalized by their reference values; convergence with refinement and the
// values at the finest mesh are checked against what these elements are known to give
TEST(MITC, shell_obstacle_course) {
  struct Bench { const char * name; std::function<double(uint32_t, bool)> run; double tol_quad, tol_tri; };
  std::vector<Bench> benches = {
    {"Scordelis-Lo roof", scordelis_lo, 0.05, 0.10},
    {"pinched cylinder", pinched_cylinder, 0.10, 0.20},
    {"hemisphere (18 deg hole)", hemisphere, 0.05, 0.10},
  };
  for (const Bench & b : benches) {
    for (bool tri : {false, true}) {
      std::vector<double> ratios;
      for (uint32_t N : {4u, 8u, 16u}) {
        ratios.push_back(b.run(N, tri));
        printf("  %-7s %-26s N=%2u  normalized deflection %.4f\n", tri ? "MITC3+" : "MITC4+", b.name, N, ratios.back());
      }
      EXPECT_LT(std::fabs(ratios[2] - 1.0), std::fabs(ratios[0] - 1.0)) << b.name;   // converging
      EXPECT_NEAR(ratios[2], 1.0, tri ? b.tol_tri : b.tol_quad) << b.name << (tri ? " MITC3+" : " MITC4+");
    }
  }
}
