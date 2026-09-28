#pragma once

#include "misc/macros.hpp"
#include "containers/ndarray.hpp"

#include "femto/geometry.hpp"

namespace femto {

struct ElementQuadratureRule {
  bool compact;
  nd::array<double, 2, memory::space::cpu> points;
  nd::array<double, 1, memory::space::cpu> weights;
};

enum class QuadratureRuleType { UniformStructured, UniformUnstructured, Nonuniform };

struct MeshQuadratureRule : public GeometryData< ElementQuadratureRule >{
  MeshQuadratureRule(uint32_t q, bool compact = true);
  QuadratureRuleType type;
};

void gauss_legendre_segment_rule(uint32_t q, double * qpts, double * qwts);
void gauss_legendre_triangle_rule(uint32_t q, double * qpts, double * qwts);
void gauss_legendre_tetrahedron_rule(uint32_t q, double * qpts, double * qwts);

GeometryInfo qpts_per_geom(const MeshQuadratureRule & qrule);
GeometryInfo qpts_per_geom(const MeshQuadratureRule & qrule, int dim);

namespace impl {

template < Geometry geom >
uint32_t qpe(int q) {
  if (geom == Geometry::Quadrilateral) { return q * q; }
  if (geom == Geometry::Hexahedron) { return q * q * q; }
  return q;
}

template < Geometry geom, typename view_t = nd::view<const double, 1> >
__host__ __device__ double integration_weight(int i, const view_t & w) {
  int Q = w.shape[0];
  if constexpr (geom == Geometry::Quadrilateral) { 
    int ix = i % Q;
    int iy = i / Q;
    return w(ix) * w(iy);
  }
  if constexpr (geom == Geometry::Hexahedron) { 
    int ix = i % Q;
    int iy = (i % (Q * Q)) / Q;
    int iz = i / (Q * Q);
    return w(ix) * w(iy) * w(iz);
  }
  return w(i);
}

}

}
