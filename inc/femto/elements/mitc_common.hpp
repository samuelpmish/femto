#pragma once

// Shared machinery for the MITC (mixed interpolation of tensorial components) shell
// elements MITC3+ and MITC4+. The displacement-based covariant strains of the degenerated
// solid are sampled at tying points and re-interpolated by an assumed strain field; the
// result is expressed in a local Cartesian frame at the quadrature point. See docs/mitc_shells.
//
// Kinematics (Ahmad / Bathe degenerated solid, five dofs per node). With midsurface nodes
// X_k, unit directors V_k, thickness a_k, and the nodal frame (V1_k, V2_k) orthogonal to V_k,
//   x(r, s, z) = sum_k h_k (X_k + z a_k/2 V_k),                    z in [-1, 1]
//   u(r, s, z) = sum_k h_k U_k + z sum_k f_k a_k/2 (alpha_k V1_k + beta_k V2_k),
// h the Lagrange functions of the vertices, f the rotation shape functions (h, plus the
// cubic bubble of MITC3+). The node dofs are (U_x, U_y, U_z, alpha, beta). The linear
// covariant strains e_ij = (g_i . u_,j + g_j . u_,i) / 2 in the natural basis
// g_i = x_,i, i in {r, s, z}, are formed with e_zz = 0 (inextensible director), the shear
// (and for MITC4+ the membrane) components are replaced by their tied fields, and the
// tensor is expressed in the local frame e_3 = g_z / |g_z|, e_1 along g_r, e_2 = e_3 x e_1.
// evaluate() therefore returns full 3 x 3 strain tensors, one per (in-plane quadrature
// point, thickness point), that a three-dimensional material law can consume directly,
// with the understanding that its normal stress along e_3 should be condensed out.

#include <cmath>
#include <vector>
#include <functional>

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"
#include "containers/ndarray.hpp"

namespace femto {
namespace mitc {

constexpr uint32_t ncomp = 5;    // dofs per node
constexpr uint32_t nstrain = 6;  // covariant components tracked: rr ss zz rs rz sz
constexpr uint32_t ntensor = 9;  // rows of the local strain operator: 3 x 3, row 3a + b

// covariant component rows
constexpr uint32_t RR = 0, SS = 1, ZZ = 2, RS = 3, RZ = 4, SZ = 5;

// a small row-major dense matrix; the tying tables are at most 12 x 12
struct dense {
  uint32_t rows = 0, cols = 0;
  std::vector<double> a;
  dense() = default;
  dense(uint32_t r, uint32_t c) : rows(r), cols(c), a(size_t(r) * c, 0.0) {}
  double & operator()(uint32_t i, uint32_t j) { return a[size_t(i) * cols + j]; }
  double operator()(uint32_t i, uint32_t j) const { return a[size_t(i) * cols + j]; }
};

inline dense matmul(const dense & A, const dense & B) {
  dense C(A.rows, B.cols);
  for (uint32_t i = 0; i < A.rows; i++)
    for (uint32_t k = 0; k < A.cols; k++) {
      double aik = A(i, k);
      if (aik == 0.0) continue;
      for (uint32_t j = 0; j < B.cols; j++) C(i, j) += aik * B(k, j);
    }
  return C;
}

// Gauss-Jordan inverse with partial pivoting (tiny systems only)
inline dense inverse(dense M) {
  uint32_t n = M.rows;
  dense I(n, n);
  for (uint32_t i = 0; i < n; i++) I(i, i) = 1.0;
  for (uint32_t c = 0; c < n; c++) {
    uint32_t piv = c;
    for (uint32_t r = c + 1; r < n; r++) if (std::fabs(M(r, c)) > std::fabs(M(piv, c))) piv = r;
    if (piv != c) {
      for (uint32_t j = 0; j < n; j++) { std::swap(M(c, j), M(piv, j)); std::swap(I(c, j), I(piv, j)); }
    }
    double d = M(c, c);
    for (uint32_t j = 0; j < n; j++) { M(c, j) /= d; I(c, j) /= d; }
    for (uint32_t r = 0; r < n; r++) {
      if (r == c) continue;
      double f = M(r, c);
      if (f == 0.0) continue;
      for (uint32_t j = 0; j < n; j++) { M(r, j) -= f * M(c, j); I(r, j) -= f * I(c, j); }
    }
  }
  return I;
}

// one tying sample: a linear combination (weights w) of the components of one strain group,
// at the in-plane point xi (and at the thickness coordinate of the evaluation point)
struct Sample {
  vec2 xi;
  double w[3];
};

// an assumed strain field for one group of covariant components: components(xi) = P(xi) a,
// with the m coefficients a = W * (sampled displacement-based strains). For interpolatory
// schemes (as many samples as coefficients) W is the inverse of the tying-condition matrix;
// MITC3+ supplies W explicitly.
struct Scheme {
  std::vector<uint32_t> rows;                     // the covariant rows of the group (e.g. {RZ, SZ})
  uint32_t m = 0;                                 // number of coefficients
  std::vector<Sample> samples;
  std::function<void(vec2, dense &)> basis;       // fills P(xi), rows.size() x m
  dense W;                                        // m x samples.size()

  // H(xi) = P(xi) W: tied components at xi = H * sampled values
  dense weights(vec2 xi) const {
    dense P(rows.size(), m);
    basis(xi, P);
    return matmul(P, W);
  }

  void solve_interpolatory() {
    uint32_t ncg = rows.size();
    dense M(m, m), P(ncg, m);
    for (uint32_t i = 0; i < m; i++) {
      basis(samples[i].xi, P);
      for (uint32_t j = 0; j < m; j++) {
        double v = 0.0;
        for (uint32_t c = 0; c < ncg; c++) v += samples[i].w[c] * P(c, j);
        M(i, j) = v;
      }
    }
    W = inverse(M);
  }
};

// shape functions of the displacement-based element at one in-plane point: h for the
// translations (and the geometry), f for the rotations (equal to h except for the MITC3+
// bubble), and their reference-coordinate gradients
struct ShapeData {
  std::vector<double> h, f;
  std::vector<vec2> dh, df;
  void resize(uint32_t n) { h.assign(n, 0.0); f.assign(n, 0.0); dh.assign(n, vec2{}); df.assign(n, vec2{}); }
};

// the reference geometry of one element, as the element kernels consume it
struct ElementGeometry {
  uint32_t nv = 0;      // vertices (3 or 4), the first nv nodes
  uint32_t n = 0;       // rotation nodes (nv, or nv + 1 for the MITC3+ bubble)
  vec3 X[4];            // midsurface vertex coordinates
  vec3 Vh[4];           // a_k / 2 V_k: half-thickness directors at the vertices
  vec3 d1[5], d2[5];    // a_k / 2 V1_k, a_k / 2 V2_k at the rotation nodes
};

// Bathe's convention for the nodal frame: V1 = e_y x V / |e_y x V| (V1 = e_z if V || e_y), V2 = V x V1
inline void default_frame(const vec3 & V, vec3 & V1, vec3 & V2) {
  vec3 c = cross(vec3{0.0, 1.0, 0.0}, V);
  V1 = (norm(c) > 1e-8) ? normalize(c) : vec3{0.0, 0.0, 1.0};
  V2 = normalize(cross(V, V1));
}

// J = [g_r, g_s, g_z] (columns) at (xi, z), from the shape data at xi
inline mat3 jacobian(const ElementGeometry & g, const ShapeData & s, double z) {
  mat3 J{};
  for (uint32_t k = 0; k < g.nv; k++) {
    for (int i = 0; i < 3; i++) {
      double xk = g.X[k][i] + z * g.Vh[k][i];
      J(i, 0) += s.dh[k][0] * xk;
      J(i, 1) += s.dh[k][1] * xk;
      J(i, 2) += s.h[k] * g.Vh[k][i];
    }
  }
  return J;
}

// displacement-based covariant strain rows (6 x 5n) at (xi, z): e_ij = (g_i . u_,j + g_j . u_,i) / 2
// for ij in {rr, ss, rs, rz, sz}; the zz row is left zero
inline void covariant_rows(const ElementGeometry & g, const ShapeData & s, double z, dense & B) {
  uint32_t n = g.n;
  B = dense(nstrain, ncomp * n);
  mat3 J = jacobian(g, s, z);
  vec3 gv[3];
  for (int a = 0; a < 3; a++) for (int i = 0; i < 3; i++) gv[a][i] = J(i, a);
  for (uint32_t k = 0; k < n; k++) {
    uint32_t c = ncomp * k;
    vec2 dh = (k < g.nv) ? s.dh[k] : vec2{0.0, 0.0};
    vec2 df = s.df[k];
    double f = s.f[k];
    // translations: u_,a = dh_a U (a = r, s), u_,z = 0
    for (int i = 0; i < 3; i++) {
      B(RR, c + i) = gv[0][i] * dh[0];
      B(SS, c + i) = gv[1][i] * dh[1];
      B(RS, c + i) = 0.5 * (gv[0][i] * dh[1] + gv[1][i] * dh[0]);
      B(RZ, c + i) = 0.5 * gv[2][i] * dh[0];
      B(SZ, c + i) = 0.5 * gv[2][i] * dh[1];
    }
    // rotations: u_,a = z df_a d, u_,z = f d, with d = d1 (alpha) or d2 (beta)
    const vec3 * d[2] = {&g.d1[k], &g.d2[k]};
    for (int m = 0; m < 2; m++) {
      double gr = dot(gv[0], *d[m]), gs = dot(gv[1], *d[m]), gz = dot(gv[2], *d[m]);
      B(RR, c + 3 + m) = z * gr * df[0];
      B(SS, c + 3 + m) = z * gs * df[1];
      B(RS, c + 3 + m) = 0.5 * z * (gr * df[1] + gs * df[0]);
      B(RZ, c + 3 + m) = 0.5 * (gr * f + z * gz * df[0]);
      B(SZ, c + 3 + m) = 0.5 * (gs * f + z * gz * df[1]);
    }
  }
}

// replace the rows of one group by the tied field: rows = H(xi) * S, where S_i are the sampled
// combinations of the displacement-based rows at the tying points (at the same thickness
// coordinate). `rows_at(xi, B)` supplies the displacement-based covariant rows at a point.
template < typename RowsAt >
inline void apply_scheme(const Scheme & sch, vec2 xi, const RowsAt & rows_at, dense & B) {
  uint32_t ns = sch.samples.size(), ncg = sch.rows.size();
  dense S(ns, B.cols), Bk;
  for (uint32_t i = 0; i < ns; i++) {
    rows_at(sch.samples[i].xi, Bk);
    for (uint32_t c = 0; c < ncg; c++) {
      double w = sch.samples[i].w[c];
      if (w == 0.0) continue;
      for (uint32_t j = 0; j < B.cols; j++) S(i, j) += w * Bk(sch.rows[c], j);
    }
  }
  dense H = sch.weights(xi);
  dense T = matmul(H, S);
  for (uint32_t c = 0; c < ncg; c++)
    for (uint32_t j = 0; j < B.cols; j++) B(sch.rows[c], j) = T(c, j);
}

// the local Cartesian frame at a point: Q = [e_1, e_2, e_3] with e_3 along g_z and e_1 along g_r
inline mat3 local_frame(const mat3 & J) {
  vec3 gr{J(0, 0), J(1, 0), J(2, 0)}, gz{J(0, 2), J(1, 2), J(2, 2)};
  vec3 e3 = normalize(gz);
  vec3 e1 = normalize(gr - dot(gr, e3) * e3);
  vec3 e2 = cross(e3, e1);
  mat3 Q;
  for (int i = 0; i < 3; i++) { Q(i, 0) = e1[i]; Q(i, 1) = e2[i]; Q(i, 2) = e3[i]; }
  return Q;
}

// covariant -> local Cartesian tensor components: eps = A^T E A with A = J^{-1} Q, where
// E is the symmetric covariant tensor. B gets the 9 rows (3a + b) of the 3 x 3 tensor.
inline void to_local(const dense & Bcov, const mat3 & J, nd::view<double, 2> B) {
  mat3 A = dot(inv(J), local_frame(J));
  const uint32_t idx[3][3] = {{RR, RS, RZ}, {RS, SS, SZ}, {RZ, SZ, ZZ}};
  for (uint32_t j = 0; j < Bcov.cols; j++) {
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        double v = 0.0;
        for (int i = 0; i < 3; i++)
          for (int k = 0; k < 3; k++) v += A(i, a) * Bcov(idx[i][k], j) * A(k, b);
        B(3 * a + b, j) = v;
      }
  }
}

} // namespace mitc
} // namespace femto
