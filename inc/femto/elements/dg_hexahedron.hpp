#pragma once

#include "femto/connection.hpp" 

namespace femto {

template <>
struct FiniteElement<Geometry::Hexahedron, Family::DG> : private FiniteElement<Geometry::Hexahedron, Family::H1> {

  using value_type = vec1;
  using derivative_type = vec3;

  using source_type = vec1;
  using flux_type = vec3;

  using FiniteElement<Geometry::Hexahedron, Family::H1>::nodes;
  using FiniteElement<Geometry::Hexahedron, Family::H1>::num_nodes;

  using FiniteElement<Geometry::Hexahedron, Family::H1>::shape_function;
  using FiniteElement<Geometry::Hexahedron, Family::H1>::shape_function_gradient;
  using FiniteElement<Geometry::Hexahedron, Family::H1>::shape_function_derivative;

  using FiniteElement<Geometry::Hexahedron, Family::H1>::evaluate_shape_functions;
  using FiniteElement<Geometry::Hexahedron, Family::H1>::evaluate_shape_function_gradients;
  using FiniteElement<Geometry::Hexahedron, Family::H1>::evaluate_weighted_shape_functions;
  using FiniteElement<Geometry::Hexahedron, Family::H1>::evaluate_weighted_shape_function_gradients;

  FiniteElement(uint32_t p_) : FiniteElement<Geometry::Hexahedron, Family::H1>{p_} {}

  // all DG nodes are interior
  __host__ __device__ uint32_t num_interior_nodes() const { return num_nodes(); }

  // all DG nodes are interior
  void interior_nodes(nd::view< double, 2 > xi) const { nodes(xi); }

  __host__ __device__ inline void reorient(Connection c, uint32_t & i0, uint32_t & i1, uint32_t & i2, uint32_t & i3) const {
    uint8_t o = c.orientation();

    uint32_t copy[4] = {i0, i1, i2, i3};

    if (c.sign() == Sign::Negative) {
      if (o == 0) { i0 = copy[0]; i1 = copy[2]; i2 = copy[1]; i3 = copy[3]; } // ok
      if (o == 1) { i0 = copy[1]; i1 = copy[0]; i2 = copy[3]; i3 = copy[2]; } // ok
      if (o == 2) { i0 = copy[3]; i1 = copy[1]; i2 = copy[2]; i3 = copy[0]; } // ok
      if (o == 3) { i0 = copy[2]; i1 = copy[3]; i2 = copy[0]; i3 = copy[1]; } // ok
    } else {
      if (o == 0) { /* nothing to do */ }
      if (o == 1) { /* should never occur */ }
      if (o == 2) { /* should never occur */ }
      if (o == 3) { /* should never occur */ }
    }
  }

  __host__ __device__ void indices(const GeometryInfo & offsets, 
                                   const Connection * hex, 
                                   uint32_t * ids) const {
    uint32_t nodes_per_hex = num_nodes();
    uint32_t hex_id = hex[Hexahedron::cell_offset].index;
    uint32_t base_offset = offsets.hex + hex_id * nodes_per_hex;
    for (uint32_t i = 0; i < nodes_per_hex; i++) {
      ids[i] = base_offset + i;
    } 
  }

  double interpolate(vec3 xi, double * values) const {
    uint32_t count = 0;
    double output = 0.0;
    for (int iz = 0; iz < (p+1); iz++) {
      double phi_z = phi_1D(xi[2], iz);
      for (int iy = 0; iy < (p+1); iy++) {
        double phi_y = phi_1D(xi[1], iy);
        for (int ix = 0; ix < (p+1); ix++) {
          double phi_x = phi_1D(xi[0], ix);
          output += phi_x * phi_y * phi_z * values[count];
          count++;
        }
      }
    }
    return output;
  }

  vec3 gradient(vec3 xi, double * values) const {
    vec3 output{};
    uint32_t count = 0;
    for (int iz = 0; iz < (p+1); iz++) {
      double phi_z  = phi_1D(xi[2], iz);
      double dphi_z = dphi_1D(xi[2], iz);
      for (int iy = 0; iy < (p+1); iy++) {
        double phi_y  = phi_1D(xi[1], iy);
        double dphi_y = dphi_1D(xi[1], iy);
        for (int ix = 0; ix < (p+1); ix++) {
          double phi_x  = phi_1D(xi[0], ix);
          double dphi_x = dphi_1D(xi[0], ix);
          output[0] += dphi_x *  phi_y *  phi_z * values[count];
          output[1] +=  phi_x * dphi_y *  phi_z * values[count];
          output[2] +=  phi_x *  phi_y * dphi_z * values[count];
          count++;
        }
      }
    }
    return output;
  }

  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    uint32_t n = p + 1;
    uint32_t q = xi.shape[0];
    return q * n * (n + q);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 3, memory::space::cpu > evaluate_shape_functions(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3, memory::space::cpu> buffer({1, q, p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &buffer(0, i, 0));
    }
    return buffer;
  }

  __host__ __device__ void interpolate(nd::view<value_type> values_q, nd::view<const double> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> phi(&shape_fns(0,0,0), {q, n});
    nd::view<double, 3> A1(buffer, {q, n, n});
    nd::view<double, 3> A2(buffer + q * n * n, {q, q, n});

    nd::view<const double, 3> values_e_3D = nd::reshape<3>(values_e, {n, n, n});
    nd::view<double, 3> values_q_3D((double*)&values_q[0], {q, q, q});
    contract(values_e_3D, phi, A1);
    contract(         A1, phi, A2);
    contract(         A2, phi, values_q_3D);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 3, memory::space::cpu > evaluate_shape_function_gradients(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3, memory::space::cpu> shape_fn_grads({2, q, p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &shape_fn_grads(0, i, 0));
      GaussLobattoInterpolationDerivative(xi(i, 0), p+1, &shape_fn_grads(1, i, 0));
    }
    return shape_fn_grads;
  }

  __host__ __device__ void gradient(nd::view<derivative_type> gradients_q, nd::view<const double, 1> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> B(&shape_fns(0,0,0), {q, n});
    nd::view<double, 2> G(&shape_fns(1,0,0), {q, n});
    nd::view<double, 3> A1(buffer, {q, n, n});
    nd::view<double, 3> A2(buffer + q * n * n, {q, q, n});

    nd::range All{0u, gradients_q.shape[0]};

    nd::view<const double, 3> values_e_3D = nd::reshape<3>(values_e, {n, n, n});
    nd::view<double, 3> gradients_q_3D((double*)&gradients_q[0], {q, q, q}, {3*q*q, 3*q, 3});

    // du_dxi
    contract(values_e_3D, G, A1);
    contract(         A1, B, A2);
    contract(         A2, B, gradients_q_3D);
    gradients_q_3D.values++;

    // du_deta
    contract(values_e_3D, B, A1);
    contract(         A1, G, A2);
    contract(         A2, B, gradients_q_3D);
    gradients_q_3D.values++;

    // du_dzeta
    contract(A1, B, A2);
    contract(A2, G, gradients_q_3D);
    gradients_q_3D.values++;

  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 2, memory::space::cpu > evaluate_weighted_shape_functions(nd::view<const double,2> xi,
                                                                        nd::view<const double,1> weights) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2, memory::space::cpu> buffer({p+1, q});
    nd::array<double, 1, memory::space::cpu> tmp({p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(j, i) = tmp(j) * weights(i);
      }
    }
    return buffer;
  }

  __host__ __device__ void integrate_source(nd::view<double, 1> residual_e, nd::view<const source_type, 1> source_q, nd::view<double, 2> shape_fn, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fn.shape[1];

    nd::view<double, 2> BT = shape_fn;
    nd::view<double, 3> A1(buffer, {n, q, q});
    nd::view<double, 3> A2(buffer + n*q*q, {n, n, q});

    nd::view<const double, 3> s3D((double*)&source_q[0], {q, q, q});
    nd::view<double, 3> r3D = nd::reshape<3>(residual_e, {n, n, n});

    //   r(iz, iy, ix) = s(qz, qy, qx) * BT(ix, qx) * BT(iy, qy) * BT(iz, qz)
    // 
    //  A1(ix, qz, qy) = BT(ix, qx) *  s(qz, qy, qx)
    //  A2(iy, ix, qz) = BT(iy, qy) * A1(ix, qz, qy)
    //   r(iz, iy, ix) = BT(iz, qz) * A2(iy, ix, qz)
    contract(s3D, BT, A1);
    contract( A1, BT, A2);
    contract( A2, BT, r3D);
  }

  // note: the return value here includes the shape function evaluations as well as
  // a buffer space to hold intermediate values in the `interpolation` function
  nd::array< double, 3, memory::space::cpu > evaluate_weighted_shape_function_gradients(nd::view<const double,2> xi,
                                                                                 nd::view<const double,1> weights) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3, memory::space::cpu> buffer({2, p+1, q});
    nd::array<double, 1, memory::space::cpu> tmp({p+1});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolation(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(0, j, i) = tmp(j) * weights(i);
      }

      GaussLobattoInterpolationDerivative(xi(i, 0), p+1, &tmp(0));
      for (int j = 0; j < p+1; j++) {
        buffer(1, j, i) = tmp(j) * weights(i);
      }
    }
    return buffer;
  }

  void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[2];

    nd::view<double, 2> BT = shape_fns(0);
    nd::view<double, 2> GT = shape_fns(1);
    nd::view<double, 3> A1(buffer, {n, q, q});
    nd::view<double, 3> A2(buffer+n*q*q, {n, n, q});

    nd::range All{0u, flux_q.shape[0]};

    uint32_t s = flux_q.stride[0] * 3;
    nd::view<const double, 3> f3D(nullptr, {q, q, q}, {s*q*q, s*q, s});
    nd::view<double, 3> r3D = nd::reshape<3>(residual_e, {n, n, n});

    f3D.values = &flux_q(0)[0];
    contract(f3D, GT, A1);
    contract( A1, BT, A2);
    contract( A2, BT, r3D, /* accumulate = */ false);

    f3D.values = &flux_q(0)[1];
    contract(f3D, BT, A1);
    contract( A1, GT, A2);
    contract( A2, BT, r3D, /* accumulate = */ true);

    f3D.values = &flux_q(0)[2];
    contract(f3D, BT, A1);
    contract( A1, BT, A2);
    contract( A2, GT, r3D, /* accumulate = */ true);
  }

  #ifdef __CUDACC__ 
  __device__ void cuda_interpolate(nd::view<value_type> values_q, nd::view<double, 1> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> phi(&shape_fns(0,0,0), {q, n});
    nd::view<double, 3> A1(buffer, {q, n, n});
    nd::view<double, 3> A2(buffer + q * n * n, {q, q, n});

    nd::view<double, 3> values_e_3D = nd::reshape<3>(values_e, {n, n, n});
    nd::view<double, 3> values_q_3D((double*)&values_q[0], {q, q, q});

    threadid tid;
    tid.x = threadIdx.x % q;
    tid.y = (threadIdx.x % (q * q)) / q;
    tid.z = threadIdx.x / (q * q);
    tid.stride = q;

    cuda_contract(tid, values_e_3D, phi, A1);
    cuda_contract(tid,          A1, phi, A2);
    cuda_contract(tid,          A2, phi, values_q_3D);
  }

  __device__ void cuda_gradient(nd::view<vec3> gradients_q, nd::view<double, 1> values_e, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[1];

    nd::view<double, 2> B(&shape_fns(0,0,0), {q, n});
    nd::view<double, 2> G(&shape_fns(1,0,0), {q, n});
    nd::view<double, 3> A1(buffer, {q, n, n});
    nd::view<double, 3> A2(buffer + q * n * n, {q, q, n});

    nd::range All{0u, gradients_q.shape[0]};

    nd::view<double, 3> values_e_3D = nd::reshape<3>(values_e, {n, n, n});
    nd::view<double, 3> gradients_q_3D((double*)&gradients_q[0], {q, q, q}, {3*q*q, 3*q, 3});

    threadid tid; 
    tid.x = threadIdx.x % q;
    tid.y = (threadIdx.x % (q * q)) / q;
    tid.z = threadIdx.x / (q * q);
    tid.stride = q;

    // du_dxi
    cuda_contract(tid, values_e_3D, G, A1);
    cuda_contract(tid,          A1, B, A2);
    cuda_contract(tid,          A2, B, gradients_q_3D);
    gradients_q_3D.values++;

    // du_deta
    cuda_contract(tid, values_e_3D, B, A1);
    cuda_contract(tid,          A1, G, A2);
    cuda_contract(tid,          A2, B, gradients_q_3D);
    gradients_q_3D.values++;

    // du_dzeta
    cuda_contract(tid, A1, B, A2);
    cuda_contract(tid, A2, G, gradients_q_3D);
    gradients_q_3D.values++;

  }

  __device__ void cuda_integrate_source(nd::view<double, 1> residual_e, nd::view<const source_type, 1> source_q, nd::view<double, 2> shape_fn, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fn.shape[1];

    nd::view<double, 2> BT = shape_fn;
    nd::view<double, 3> A1(buffer, {n, q, q});
    nd::view<double, 3> A2(buffer + n*q*q, {n, n, q});

    nd::view<double, 3> s3D((double*)&source_q[0], {q, q, q});
    nd::view<double, 3> r3D = nd::reshape<3>(residual_e, {n, n, n});

    threadid tid;
    tid.x = threadIdx.x % n;
    tid.y = (threadIdx.x % (n * n)) / n;
    tid.z = threadIdx.x / (n * n);
    tid.stride = n;

    cuda_contract(tid, s3D, BT, A1);
    cuda_contract(tid,  A1, BT, A2);
    cuda_contract(tid,  A2, BT, r3D);
  }

  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<double, 3> shape_fns, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = shape_fns.shape[2];

    nd::view<double, 2> BT = shape_fns(0);
    nd::view<double, 2> GT = shape_fns(1);
    nd::view<double, 3> A1(buffer, {n, q, q});
    nd::view<double, 3> A2(buffer+n*q*q, {n, n, q});

    uint32_t s = flux_q.stride[0] * 3;
    nd::view<double, 3> f3D(nullptr, {q, q, q}, {s*q*q, s*q, s});
    nd::view<double, 3> r3D = nd::reshape<3>(residual_e, {n, n, n});

    threadid tid; 
    tid.x = threadIdx.x % n;
    tid.y = (threadIdx.x % (n * n)) / n;
    tid.z = threadIdx.x / (n * n);
    tid.stride = n;

    f3D.values = &flux_q(0)[0];
    cuda_contract(tid, f3D, GT, A1);
    cuda_contract(tid,  A1, BT, A2);
    cuda_contract(tid,  A2, BT, r3D, /* accumulate = */ false);

    f3D.values = &flux_q(0)[1];
    cuda_contract(tid, f3D, BT, A1);
    cuda_contract(tid,  A1, GT, A2);
    cuda_contract(tid,  A2, BT, r3D, /* accumulate = */ true);

    f3D.values = &flux_q(0)[2];
    cuda_contract(tid, f3D, BT, A1);
    cuda_contract(tid,  A1, BT, A2);
    cuda_contract(tid,  A2, GT, r3D, /* accumulate = */ true);
  }
  #endif

};

} // namespace femto
