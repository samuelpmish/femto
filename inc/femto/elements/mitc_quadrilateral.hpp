#pragma once

#include "femto/connection.hpp"
#include "femto/elements/mitc_common.hpp"

namespace femto {

// MITC4+ quadrilateral shell element (Ko, Lee, Bathe 2017), see mitc_common.hpp for the
// kinematics. Node layout and bilinear shape functions are those of the H1 quadrilateral.
// The transverse shear strains are tied as in MITC4 (Dvorkin, Bathe 1984) and the membrane
// strains follow the assumed field built from the characteristic geometry vectors; the
// bending strains are displacement-based. Only p = 1 is defined.
template <>
struct FiniteElement<Geometry::Quadrilateral, Family::MITC> {

  static constexpr int dim = 2;
  static constexpr uint32_t components = mitc::ncomp;
  static constexpr uint32_t num_vertices = 4;

  using h1 = FiniteElement<Geometry::Quadrilateral, Family::H1>;

  __host__ __device__ uint32_t num_nodes() const { return 4; }
  __host__ __device__ uint32_t num_interior_nodes() const { return 0; }
  void nodes(nd::view<double, 2> xi) const { h1{1}.nodes(xi); }
  void interior_nodes(nd::view<double, 2>) const {}
  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * quad, uint32_t * indices) const {
    h1{1}.indices(offsets, quad, indices);
  }

  void shape(vec2 xi, mitc::ShapeData & s) const {
    h1 el{1};
    s.resize(4);
    for (uint32_t k = 0; k < 4; k++) {
      s.h[k] = s.f[k] = el.shape_function(xi, k);
      s.dh[k] = s.df[k] = el.shape_function_gradient(xi, k);
    }
  }

  // the strain operator B (9 x 5 num_nodes()) at (xi, z): local Cartesian tensor components,
  // row 3a + b, of the tied strain
  void strain_matrix(vec2 xi, double z, const mitc::ElementGeometry & g, nd::view<double, 2> B) const {
    mitc::ShapeData s;
    auto rows_at_z = [&](vec2 pt, double zz, mitc::dense & Bk) { shape(pt, s); mitc::covariant_rows(g, s, zz, Bk); };
    auto rows_at = [&](vec2 pt, mitc::dense & Bk) { rows_at_z(pt, z, Bk); };
    mitc::dense Bcov;
    rows_at(xi, Bcov);
    mitc::apply_scheme(mitc4_shear(), xi, rows_at, Bcov);
    mitc4plus_membrane(xi, g, rows_at_z, Bcov);
    shape(xi, s);
    mitc::to_local(Bcov, mitc::jacobian(g, s, z), B);
  }

  // MITC4 transverse shear: e_rz constant in r and linear in s from the midpoints of the edges
  // s = 0, 1; e_sz likewise from the edges r = 0, 1. (Reference square [0, 1]^2.)
  static const mitc::Scheme & mitc4_shear() {
    static const mitc::Scheme sch = [] {
      mitc::Scheme s;
      s.rows = {mitc::RZ, mitc::SZ}; s.m = 4;
      s.samples = {{{0.5, 0.0}, {1, 0, 0}}, {{0.5, 1.0}, {1, 0, 0}}, {{0.0, 0.5}, {0, 1, 0}}, {{1.0, 0.5}, {0, 1, 0}}};
      s.basis = [](vec2 xi, mitc::dense & P) {
        P(0, 0) = 1; P(0, 1) = xi[1]; P(0, 2) = 0; P(0, 3) = 0;
        P(1, 0) = 0; P(1, 1) = 0;     P(1, 2) = 1; P(1, 3) = xi[0];
      };
      s.solve_interpolatory();
      return s;
    }();
    return sch;
  }

  // MITC4+ assumed membrane strain (Ko, Lee, Bathe 2017, eqs. 9-26). In the paper's natural
  // coordinates r, s in [-1, 1] the displacement-based covariant membrane strains (the in-plane
  // strains at z = 0) are
  //   e_rr = e_rr,con + e_rr,lin s + e_rs,bil s^2,  e_ss = e_ss,con + e_ss,lin r + e_rs,bil r^2,
  //   e_rs = e_rs,con + (e_rr,lin r + e_ss,lin s)/2 + e_rs,bil r s,
  // with e_rs,bil = x_d . u_d the term that locks. The five samples e_rr(A = (0,1)), e_rr(B = (0,-1)),
  // e_ss(C = (1,0)), e_ss(D = (-1,0)), e_rs(E = (0,0)) give every coefficient except e_rs,bil, which
  // is replaced by the combination (eq. 25) fixed by the in-plane distortions c_r, c_s of the
  // characteristic vector x_d (its components in the dual basis of x_r, x_s within the tangent
  // plane). On a flat element the result equals the displacement-based strain. The bending
  // part e_ij(z) - e_ij(0) is left displacement-based. All strains here are covariant components
  // on [0, 1]^2, a uniform factor 4 that the linear, homogeneous construction does not see.
  template < typename RowsAtZ >
  static void mitc4plus_membrane(vec2 xi, const mitc::ElementGeometry & g, const RowsAtZ & rows_at_z, mitc::dense & B) {
    // characteristic geometry vectors, nodes at (r_k, s_k) = 2 xi_k - 1 in the lexicographic order
    vec3 xr{}, xs{}, xd{};
    for (uint32_t k = 0; k < 4; k++) {
      double rk = (k % 2 == 0) ? -1.0 : 1.0, sk = (k / 2 == 0) ? -1.0 : 1.0;
      xr += 0.25 * rk * g.X[k]; xs += 0.25 * sk * g.X[k]; xd += 0.25 * rk * sk * g.X[k];
    }
    // c_r = x_d . m^r, c_s = x_d . m^s with (m^r, m^s) the dual basis of (x_r, x_s) in their plane
    mat2 G{{{dot(xr, xr), dot(xr, xs)}, {dot(xs, xr), dot(xs, xs)}}};
    vec2 c = dot(inv(G), vec2{dot(xd, xr), dot(xd, xs)});
    double cr = c[0], cs = c[1];
    double d = cr * cr + cs * cs - 1.0;

    // sampled displacement-based membrane rows (z = 0), and the membrane rows at xi
    mitc::dense BA, BB, BC, BD, BE, B0;
    rows_at_z(vec2{0.5, 1.0}, 0.0, BA); rows_at_z(vec2{0.5, 0.0}, 0.0, BB);
    rows_at_z(vec2{1.0, 0.5}, 0.0, BC); rows_at_z(vec2{0.0, 0.5}, 0.0, BD);
    rows_at_z(vec2{0.5, 0.5}, 0.0, BE); rows_at_z(xi, 0.0, B0);
    double r = 2.0 * xi[0] - 1.0, s = 2.0 * xi[1] - 1.0;
    for (uint32_t j = 0; j < B.cols; j++) {
      double eA = BA(mitc::RR, j), eB = BB(mitc::RR, j);
      double eC = BC(mitc::SS, j), eD = BD(mitc::SS, j);
      double eE = BE(mitc::RS, j);
      double mrr = 0.5 * (eA + eB), lrr = 0.5 * (eA - eB);   // e_rr,con + e_rs,bil ;  e_rr,lin
      double mss = 0.5 * (eC + eD), lss = 0.5 * (eC - eD);
      double bil = (cr * (cr * mrr - lrr) + cs * (cs * mss - lss) + 2.0 * cr * cs * eE) / d;
      // tied membrane strain replaces the displacement-based one; the bending part stays
      B(mitc::RR, j) += (mrr + lrr * s - bil * (1.0 - s * s)) - B0(mitc::RR, j);
      B(mitc::SS, j) += (mss + lss * r - bil * (1.0 - r * r)) - B0(mitc::SS, j);
      B(mitc::RS, j) += (eE + 0.5 * lrr * r + 0.5 * lss * s + bil * r * s) - B0(mitc::RS, j);
    }
  }

  uint32_t p;
};

} // namespace femto
