#pragma once

// Evaluation of *test* basis functions for sparse matrix assembly, shared by
// the CPU kernel (integrate_spmat.hpp) and the CUDA kernel (integrate_spmat.cuh).
//
// Rather than tabulating every shape function at every quadrature point, quads
// and hexes tabulate only the 1D factors of their tensor-product basis and
// reconstruct each shape function on demand. That turns an O(qpts x nodes x
// components) table into an O(qpts x p) one -- the difference between 221 KB
// and 448 bytes for a cubic Hcurl hexahedron.

#include "common.hpp"

#include <type_traits>

namespace femto {

namespace impl {

// quads and hexes have tensor-product shape functions, so only the 1D factors
// are tabulated and the kernel reconstructs each shape function on the fly.
// Everything else gets a dense {qpts, nodes, components} table.
template < Geometry geom, Family family >
inline constexpr bool tensor_product_shape_functions =
  (geom == Geometry::Quadrilateral || geom == Geometry::Hexahedron) &&
  (is_scalar_valued(family) || family == Family::Hcurl);

template < Geometry geom, Family family, DerivedQuantity op >
auto evaluate_test_shape_functions(FiniteElement<geom, family> element, nd::view<const double, 2> xi) {
  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t shape = qshape(family, op, gdim);

  if constexpr (tensor_product_shape_functions<geom, family> && is_scalar_valued(family)) {
    if constexpr (op == DerivedQuantity::VALUE) {
      return element.evaluate_shape_functions(xi);
    } else if constexpr (op == DerivedQuantity::GRAD) {
      return element.evaluate_shape_function_gradients(xi);
    }
  } else if constexpr (tensor_product_shape_functions<geom, family> && family == Family::Hcurl) {
    if constexpr (op == DerivedQuantity::VALUE) {
      return element.evaluate_shape_functions(xi);
    } else {
      return element.evaluate_shape_function_curls(xi);
    }
  } else {
    uint32_t qpts_per_element = qpe<geom>(xi.shape[0]);
    uint32_t nodes_per_element = element.num_nodes();

    if constexpr (shape == 1) {
      nd::array<double, 2, memory::space::cpu> shape_fns({qpts_per_element, nodes_per_element});
      for (uint32_t q = 0; q < qpts_per_element; q++) {
        auto xi_q = quadrature_point<geom>(q, xi);
        for (uint32_t i = 0; i < nodes_per_element; i++) {
          auto phi_i = shape_function<op>(element, xi_q, i, 0);
          if constexpr (std::is_arithmetic_v<std::decay_t<decltype(phi_i)>>) {
            shape_fns(q, i) = phi_i;
          } else {
            shape_fns(q, i) = phi_i[0];
          }
        }
      }
      return shape_fns;
    } else {
      nd::array<double, 3, memory::space::cpu> shape_fns({qpts_per_element, nodes_per_element, shape});
      for (uint32_t q = 0; q < qpts_per_element; q++) {
        auto xi_q = quadrature_point<geom>(q, xi);
        for (uint32_t i = 0; i < nodes_per_element; i++) {
          auto phi_i = shape_function<op>(element, xi_q, i, 0);
          for (uint32_t k = 0; k < shape; k++) {
            shape_fns(q, i, k) = phi_i[k];
          }
        }
      }
      return shape_fns;
    }
  }
}

template < Geometry geom, Family family, DerivedQuantity op >
auto evaluate_weighted_trial_shape_functions(
  FiniteElement<geom, family> element,
  nd::view<const double, 2> xi,
  nd::view<const double, 1> weights) {
  if constexpr (op == DerivedQuantity::VALUE) {
    return element.evaluate_weighted_shape_functions(xi, weights);
  }

  if constexpr (op == DerivedQuantity::CURL && family == Family::Hcurl) {
    return element.evaluate_weighted_shape_function_curls(xi, weights);
  }

  if constexpr (op == DerivedQuantity::GRAD && is_scalar_valued(family)) {
    return element.evaluate_weighted_shape_function_gradients(xi, weights);
  }
}

// shape_function() returns a scalar for scalar-valued VALUE operators and a
// vec for everything else; normalize to vec<qshape>
template < uint32_t qshape, typename T >
__host__ __device__ vec<qshape> as_qtype(const T & raw) {
  vec<qshape> output{};
  if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
    output[0] = raw;
  } else {
    for (uint32_t k = 0; k < qshape; k++) { output[k] = raw[k]; }
  }
  return output;
}

template < uint32_t qshape, typename view_t >
__host__ __device__ vec<qshape> load_test_shape(const view_t & shape_fns, uint32_t q, uint32_t i) {
  constexpr uint32_t shape_rank = array_rank<view_t>::value;
  vec<qshape> output{};
  if constexpr (shape_rank == 2) {
    output[0] = shape_fns(q, i);
  } else {
    for (uint32_t k = 0; k < qshape; k++) {
      output[k] = shape_fns(q, i, k);
    }
  }
  return output;
}

template < typename view_t >
__host__ __device__ double load_tensor_product_shape_1D(const view_t & shape_fns, uint32_t component, uint32_t q, uint32_t i) {
  constexpr uint32_t shape_rank = array_rank<view_t>::value;
  if constexpr (shape_rank == 2) {
    return shape_fns(q, i);
  } else {
    return shape_fns(component, q, i);
  }
}

template < Geometry geom, DerivedQuantity op, uint32_t qshape, typename view_t >
__host__ __device__ vec<qshape> load_tensor_product_scalar_test_shape(const view_t & shape_fns, uint32_t q, uint32_t i) {
  constexpr uint32_t shape_rank = array_rank<view_t>::value;
  vec<qshape> output{};

  uint32_t q1D = (shape_rank == 2) ? shape_fns.shape[0] : shape_fns.shape[1];
  uint32_t nodes_1D = (shape_rank == 2) ? shape_fns.shape[1] : shape_fns.shape[2];

  uint32_t qx = q % q1D;
  uint32_t qy = (q % (q1D * q1D)) / q1D;
  uint32_t ix = i % nodes_1D;
  uint32_t iy = (i / nodes_1D) % nodes_1D;

  double Bx = load_tensor_product_shape_1D(shape_fns, 0, qx, ix);
  double By = load_tensor_product_shape_1D(shape_fns, 0, qy, iy);

  if constexpr (geom == Geometry::Quadrilateral) {
    if constexpr (op == DerivedQuantity::VALUE) {
      output[0] = Bx * By;
    }

    if constexpr (op == DerivedQuantity::GRAD) {
      double Gx = load_tensor_product_shape_1D(shape_fns, 1, qx, ix);
      double Gy = load_tensor_product_shape_1D(shape_fns, 1, qy, iy);
      output[0] = Gx * By;
      output[1] = Bx * Gy;
    }
  }

  if constexpr (geom == Geometry::Hexahedron) {
    uint32_t qz = q / (q1D * q1D);
    uint32_t iz = i / (nodes_1D * nodes_1D);

    double Bz = load_tensor_product_shape_1D(shape_fns, 0, qz, iz);

    if constexpr (op == DerivedQuantity::VALUE) {
      output[0] = Bx * By * Bz;
    }

    if constexpr (op == DerivedQuantity::GRAD) {
      double Gx = load_tensor_product_shape_1D(shape_fns, 1, qx, ix);
      double Gy = load_tensor_product_shape_1D(shape_fns, 1, qy, iy);
      double Gz = load_tensor_product_shape_1D(shape_fns, 1, qz, iz);
      output[0] = Gx * By * Bz;
      output[1] = Bx * Gy * Bz;
      output[2] = Bx * By * Gz;
    }
  }

  return output;
}

// reconstruct one Hcurl shape function (or its curl) on a quad/hex from the 1D
// tables laid out by evaluate_shape_functions / evaluate_shape_function_curls:
//
//   B1 : [0,       q1D*p)           open   (gauss-legendre), p per point
//   B2 : [q1D*p,   q1D*(p+n))       closed (gauss-lobatto),  n per point
//   G2 : [q1D*(p+n), q1D*(p+2n))    closed derivatives,      n per point
//
// the dof blocks follow FiniteElement<geom,Hcurl>::nodes(); note the quad and
// the hex order their y-directed dofs differently
template < Geometry geom, DerivedQuantity op, uint32_t qshape, typename view_t >
__host__ __device__ vec<qshape> load_tensor_product_hcurl_test_shape(
  const view_t & shape_fns,
  uint32_t p,
  uint32_t q1D,
  uint32_t q,
  uint32_t i) {

  uint32_t n = p + 1;
  auto B1 = [&](uint32_t qi, uint32_t j) { return shape_fns[qi * p + j]; };
  auto B2 = [&](uint32_t qi, uint32_t j) { return shape_fns[q1D * p + qi * n + j]; };
  auto G2 = [&](uint32_t qi, uint32_t j) { return shape_fns[q1D * (p + n) + qi * n + j]; };

  vec<qshape> output{};

  uint32_t qx = q % q1D;
  uint32_t qy = (q / q1D) % q1D;

  if constexpr (geom == Geometry::Quadrilateral) {
    uint32_t S = p * n;
    if (i < S) {                                    // x-directed: m = ix + p*iy
      uint32_t ix = i % p, iy = i / p;
      if constexpr (op == DerivedQuantity::VALUE) {
        output[0] = B1(qx, ix) * B2(qy, iy);
      } else {                                      // curl = -dy(phi_x)
        output[0] = -B1(qx, ix) * G2(qy, iy);
      }
    } else {                                        // y-directed: m = iy + p*ix
      uint32_t m = i - S, iy = m % p, ix = m / p;
      if constexpr (op == DerivedQuantity::VALUE) {
        output[1] = B2(qx, ix) * B1(qy, iy);
      } else {                                      // curl = dx(phi_y)
        output[0] = G2(qx, ix) * B1(qy, iy);
      }
    }
  }

  if constexpr (geom == Geometry::Hexahedron) {
    uint32_t qz = q / (q1D * q1D);
    uint32_t S = p * n * n;
    if (i < S) {                                    // x-directed: m = ix + p*(iy + n*iz)
      uint32_t m = i, ix = m % p, iy = (m / p) % n, iz = m / (p * n);
      if constexpr (op == DerivedQuantity::VALUE) {
        output[0] = B1(qx, ix) * B2(qy, iy) * B2(qz, iz);
      } else {                                      // curl = (0, dz, -dy)
        output[1] =  B1(qx, ix) * B2(qy, iy) * G2(qz, iz);
        output[2] = -B1(qx, ix) * G2(qy, iy) * B2(qz, iz);
      }
    } else if (i < 2 * S) {                         // y-directed: m = ix + n*(iy + p*iz)
      uint32_t m = i - S, ix = m % n, iy = (m / n) % p, iz = m / (n * p);
      if constexpr (op == DerivedQuantity::VALUE) {
        output[1] = B2(qx, ix) * B1(qy, iy) * B2(qz, iz);
      } else {                                      // curl = (-dz, 0, dx)
        output[0] = -B2(qx, ix) * B1(qy, iy) * G2(qz, iz);
        output[2] =  G2(qx, ix) * B1(qy, iy) * B2(qz, iz);
      }
    } else {                                        // z-directed: m = ix + n*(iy + n*iz)
      uint32_t m = i - 2 * S, ix = m % n, iy = (m / n) % n, iz = m / (n * n);
      if constexpr (op == DerivedQuantity::VALUE) {
        output[2] = B2(qx, ix) * B2(qy, iy) * B1(qz, iz);
      } else {                                      // curl = (dy, -dx, 0)
        output[0] =  B2(qx, ix) * G2(qy, iy) * B1(qz, iz);
        output[1] = -G2(qx, ix) * B2(qy, iy) * B1(qz, iz);
      }
    }
  }

  return output;
}

// one shape function, before any reorientation
template < Geometry geom, Family family, DerivedQuantity op, uint32_t qshape, typename view_t >
__host__ __device__ vec<qshape> load_raw_test_shape(
  const view_t & shape_fns,
  uint32_t p,
  uint32_t q1D,
  uint32_t q,
  uint32_t i) {
  if constexpr (tensor_product_shape_functions<geom, family> && is_scalar_valued(family)) {
    return load_tensor_product_scalar_test_shape<geom, op, qshape>(shape_fns, q, i);
  } else if constexpr (tensor_product_shape_functions<geom, family> && family == Family::Hcurl) {
    return load_tensor_product_hcurl_test_shape<geom, op, qshape>(shape_fns, p, q1D, q, i);
  } else {
    return load_test_shape<qshape>(shape_fns, q, i);
  }
}

template < Geometry geom, Family family, DerivedQuantity op, uint32_t qshape, typename view_t >
__host__ __device__ vec<qshape> load_reoriented_test_shape(
  const view_t & shape_fns,
  uint32_t p,
  uint32_t q1D,
  uint32_t q,
  uint32_t i,
  int8_t transformation) {
  if constexpr (is_vector_valued(family)) {
    if (transformation == -1) {
      return -load_raw_test_shape<geom, family, op, qshape>(shape_fns, p, q1D, q, i);
    }
    if (transformation == 0) {
      return load_raw_test_shape<geom, family, op, qshape>(shape_fns, p, q1D, q, i);
    }

    uint32_t ix = (i & 0xFFFFFFFEu) + 0;
    uint32_t iy = (i & 0xFFFFFFFEu) + 1;
    vec2 weights = face_transformation(transformation)[i % 2];
    return load_raw_test_shape<geom, family, op, qshape>(shape_fns, p, q1D, q, ix) * weights[0] +
           load_raw_test_shape<geom, family, op, qshape>(shape_fns, p, q1D, q, iy) * weights[1];
  } else {
    return load_raw_test_shape<geom, family, op, qshape>(shape_fns, p, q1D, q, i);
  }
}
} // namespace impl

} // namespace femto
