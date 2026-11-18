// SPDX-FileCopyrightText: Copyright (c) 2023-2026 Sam Mish
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

// the device half of forall_void_tests.cpp: forall() over a void q-function
// has to agree with the host result, size an unallocated output, and leave an
// output that is already the right shape in place.
#include "common.hpp"

#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

using namespace femto;

struct Flux {
  HOSTDEV vec<3> operator()(vec<3> du_dxi, mat<3,3> dX_dxi) const {
    mat<3,3> dxi_dX = inv(dX_dxi);
    return dot(dot(du_dxi, dxi_dX), transpose(dxi_dX)) * det(dX_dxi);
  }
};
struct FluxVoid {
  HOSTDEV void operator()(vec<3> du_dxi, mat<3,3> dX_dxi, vec<3> & f) const {
    f = Flux{}(du_dxi, dX_dxi);
  }
};

TEST(CudaForallVoid, matches_the_host_result_and_reuses_its_output) {

  auto h_mesh = Mesh< memory::space::cpu >::load(FEMTO_MESH_DIR + std::string("patch_test_tets.json"));

  Field u = create_field<Family::H1>(h_mesh, 2, 1);
  u.data = femto::random(u.data.shape);
  Domain h_domain(h_mesh, MeshQuadratureRule(2));

  nd::cpu_array<double, 3> h_dX_dxi_q = evaluate(grad(h_mesh.X), isoparametric(h_domain));
  nd::cpu_array<double, 3> h_du_dxi_q = evaluate(grad(u), isoparametric(h_domain));
  nd::cpu_array<double, 2> expected = forall(Flux{}, h_du_dxi_q, h_dX_dxi_q);

  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Field<Family::H1, memory::space::gpu> d_u = u;
  Domain< memory::space::gpu > d_domain(d_mesh, MeshQuadratureRule(2));

  nd::gpu_array<double, 3> d_dX_dxi_q = evaluate(grad(d_mesh.X), isoparametric(d_domain));
  nd::gpu_array<double, 3> d_du_dxi_q = evaluate(grad(d_u), isoparametric(d_domain));

  nd::gpu_array<double, 2> d_f_q;
  EXPECT_EQ(d_f_q.sz, 0u);
  forall(FluxVoid{}, d_du_dxi_q, d_dX_dxi_q, d_f_q);
  ASSERT_EQ(d_f_q.sz, expected.sz);

  const double * values_before = d_f_q.data();
  forall(FluxVoid{}, d_du_dxi_q, d_dX_dxi_q, d_f_q);
  EXPECT_EQ(d_f_q.data(), values_before) << "forall() reallocated an output that was already the right size";

  nd::cpu_array<double, 2> got = d_f_q;
  ASSERT_EQ(got.sz, expected.sz);
  for (uint32_t i = 0; i < expected.sz; i++) {
    EXPECT_NEAR(got.data()[i], expected.data()[i], 1.0e-14) << "differs at " << i;
  }
}

// forall_staged() writes what forall() does: two outputs of different sizes
// at once, one of them a 648 B record, an input taken by value (which arrives
// as a non-const pointer, like an output), and a length that leaves the last
// block part full
struct TwoOutputs {
  HOSTDEV void operator()(const mat<3,3> & A, vec<3> x, mat<3,3> & B, mat<3,3,mat<3,3>> & C) const {
    B = dot(A, transpose(A)) + outer(x, x);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) { C[i][j] = A(i, j) * B + x[i] * transpose(A); }
    }
  }
};

TEST(CudaForallStaged, matches_forall) {
  uint32_t n = 1000 + 37;
  nd::cpu_array<double, 3> h_A = femto::random(stack::array< uint32_t, 3 >{n, 3u, 3u});
  nd::cpu_array<double, 2> h_x = femto::random(stack::array< uint32_t, 2 >{n, 3u});
  nd::gpu_array<double, 3> A = h_A, B, B_staged;
  nd::gpu_array<double, 2> x = h_x;
  nd::gpu_array<double, 5> C, C_staged;

  forall(TwoOutputs{}, A, x, B, C);
  forall_staged(TwoOutputs{}, A, x, B_staged, C_staged);

  nd::cpu_array<double, 3> b = B, b_staged = B_staged;
  nd::cpu_array<double, 5> c = C, c_staged = C_staged;
  ASSERT_EQ(b_staged.sz, b.sz);
  ASSERT_EQ(c_staged.sz, c.sz);
  for (uint32_t i = 0; i < b.sz; i++) { ASSERT_EQ(b_staged.data()[i], b.data()[i]) << "B differs at " << i; }
  for (uint32_t i = 0; i < c.sz; i++) { ASSERT_EQ(c_staged.data()[i], c.data()[i]) << "C differs at " << i; }
}
