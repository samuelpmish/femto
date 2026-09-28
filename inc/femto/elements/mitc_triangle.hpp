#pragma once

#include "femto/connection.hpp"
#include "femto/elements/mitc_common.hpp"

namespace femto {

// MITC3+ triangular shell element (Lee, Lee, Bathe 2014), see mitc_common.hpp for the
// kinematics. Linear translations; rotations enriched with the cubic bubble
// f4 = 27 r s (1 - r - s) carried by an element-interior node (whose translation dofs are
// unused). The transverse shear strains are tied at six interior points; the in-plane
// strains are displacement-based. Only p = 1 is defined.
template <>
struct FiniteElement<Geometry::Triangle, Family::MITC> {

  static constexpr int dim = 2;
  static constexpr uint32_t components = mitc::ncomp;
  static constexpr uint32_t num_vertices = 3;

  __host__ __device__ uint32_t num_nodes() const { return 4; }
  __host__ __device__ uint32_t num_interior_nodes() const { return 1; }

  void nodes(nd::view<double, 2> xi) const {
    xi(0, 0) = 0.0; xi(0, 1) = 0.0;
    xi(1, 0) = 1.0; xi(1, 1) = 0.0;
    xi(2, 0) = 0.0; xi(2, 1) = 1.0;
    xi(3, 0) = 1.0 / 3.0; xi(3, 1) = 1.0 / 3.0;
  }

  void interior_nodes(nd::view<double, 2> xi) const { xi(0, 0) = 1.0 / 3.0; xi(0, 1) = 1.0 / 3.0; }

  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * tri, uint32_t * indices) const {
    const Connection cell = *(tri + Triangle::cell_offset);
    indices[0] = offsets.vert + tri[0].index;
    indices[1] = offsets.vert + tri[1].index;
    indices[2] = offsets.vert + tri[2].index;
    indices[3] = offsets.tri + cell.index;  // the rotation bubble
  }

  // shape functions of the displacement-based element
  void shape(vec2 xi, mitc::ShapeData & s) const {
    s.resize(4);
    double r = xi[0], t = xi[1];
    double h[3] = {1.0 - r - t, r, t};
    vec2 dh[3] = {{-1.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}};
    double f4 = 27.0 * r * t * (1.0 - r - t);
    vec2 df4 = {27.0 * (t - 2.0 * r * t - t * t), 27.0 * (r - r * r - 2.0 * r * t)};
    for (int k = 0; k < 3; k++) {
      s.h[k] = h[k]; s.dh[k] = dh[k];
      s.f[k] = h[k] - f4 / 3.0; s.df[k] = dh[k] - df4 / 3.0;
    }
    s.h[3] = 0.0; s.dh[3] = vec2{0.0, 0.0};   // the bubble node carries no translation
    s.f[3] = f4;  s.df[3] = df4;
  }

  // the strain operator B (9 x 5 num_nodes()) at (xi, z): local Cartesian tensor components,
  // row 3a + b, of the tied strain
  void strain_matrix(vec2 xi, double z, const mitc::ElementGeometry & g, nd::view<double, 2> B) const {
    mitc::ShapeData s;
    auto rows_at = [&](vec2 pt, mitc::dense & Bk) { shape(pt, s); mitc::covariant_rows(g, s, z, Bk); };
    mitc::dense Bcov;
    rows_at(xi, Bcov);
    mitc::apply_scheme(mitc3plus_shear(), xi, rows_at, Bcov);
    shape(xi, s);   // the shape data now refers to the last tying point
    mitc::to_local(Bcov, mitc::jacobian(g, s, z), B);
  }

  // MITC3+ assumed transverse shear (Lee, Lee, Bathe 2014, eqs. 13-17): an N0 field
  //   e_rz = a1 + c s,  e_sz = a2 - c r
  // whose constant part comes from the interior points A, B, C and whose linear part from
  // D, E, F at a distance d = 1/10000 from the barycenter.
  static const mitc::Scheme & mitc3plus_shear() {
    static const mitc::Scheme sch = [] {
      mitc::Scheme s;
      s.rows = {mitc::RZ, mitc::SZ}; s.m = 3;
      const double d = 1.0e-4, third = 1.0 / 3.0;
      vec2 pts[6] = {{1.0 / 6.0, 2.0 / 3.0}, {2.0 / 3.0, 1.0 / 6.0}, {1.0 / 6.0, 1.0 / 6.0},
                     {third + d, third - 2 * d}, {third - 2 * d, third + d}, {third + d, third + d}};
      for (int k = 0; k < 6; k++) {          // samples 2k: e_rz, 2k+1: e_sz at point k
        s.samples.push_back({pts[k], {1.0, 0.0, 0.0}});
        s.samples.push_back({pts[k], {0.0, 1.0, 0.0}});
      }
      s.basis = [](vec2 xi, mitc::dense & P) {
        P(0, 0) = 1.0; P(0, 1) = 0.0; P(0, 2) = xi[1];
        P(1, 0) = 0.0; P(1, 1) = 1.0; P(1, 2) = -xi[0];
      };
      // A = 0, B = 1, C = 2, D = 3, E = 4, F = 5; column 2k is e_rz, 2k+1 is e_sz
      s.W = mitc::dense(3, 12);
      auto rt = [](int k) { return 2 * k; };
      auto st = [](int k) { return 2 * k + 1; };
      // c^ = e_rz(F) - e_rz(D) - e_sz(F) + e_sz(E)
      double chat[12] = {};
      chat[rt(5)] = 1.0; chat[rt(3)] = -1.0; chat[st(5)] = -1.0; chat[st(4)] = 1.0;
      // constant parts (eq. 15)
      double Cr[12] = {}, Cs[12] = {};
      Cr[rt(1)] = 2.0 / 3.0; Cr[st(1)] = -1.0 / 3.0; Cr[rt(2)] = 1.0 / 3.0; Cr[st(2)] = 1.0 / 3.0;
      Cs[st(0)] = 2.0 / 3.0; Cs[rt(0)] = -1.0 / 3.0; Cs[rt(2)] = 1.0 / 3.0; Cs[st(2)] = 1.0 / 3.0;
      // e_rz = Cr + (1/3) c^ (3s - 1) = (Cr - c^/3) + c^ s ;  e_sz = (Cs + c^/3) - c^ r
      for (int j = 0; j < 12; j++) {
        s.W(0, j) = Cr[j] - chat[j] / 3.0;
        s.W(1, j) = Cs[j] + chat[j] / 3.0;
        s.W(2, j) = chat[j];
      }
      return s;
    }();
    return sch;
  }

  uint32_t p;
};

} // namespace femto
