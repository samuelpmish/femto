// End-to-end agreement between the CPU and CUDA matrix-free kernels
// (evaluate / integrate_residual) across every geometry and both families,
// on isoparametric and spatial domains.
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "containers/ndarray_conversions.hpp"

using namespace femto;

namespace {

bool cuda_device_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

double rel_err(const nd::array<double, 3, memory::space::cpu> & a,
               const nd::array<double, 3, memory::space::cpu> & b) {
  EXPECT_EQ(a.size(), b.size());
  double num = 0.0, den = 0.0;
  for (uint32_t i = 0; i < a.size(); i++) {
    double d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

template < Family family >
double rel_err(const Residual<family> & a, const Residual<family> & b) {
  EXPECT_EQ(a.data.size(), b.data.size());
  double num = 0.0, den = 0.0;
  for (uint32_t i = 0; i < a.data.size(); i++) {
    double d = a.data[i] - b.data[i];
    num += d * d;
    den += b.data[i] * b.data[i];
  }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// the two paths sum in different orders, so they agree to roundoff
constexpr double tolerance = 1.0e-12;

template < Family family, DerivedQuantity op >
void check_against_cpu(const Mesh<> & mesh, uint32_t p, uint32_t q, const std::string & label) {

  SCOPED_TRACE(label + " p=" + std::to_string(p) + " rule=" + std::to_string(q));

  auto differentiate = [](const auto & f) {
    if constexpr (op == DerivedQuantity::GRAD) { return grad(f); } else { return curl(f); }
  };

  Field<family> u_cpu = create_field<family>(mesh, p, 1);
  for (uint32_t i = 0; i < u_cpu.data.shape[0]; i++) {
    u_cpu.data(i, 0) = std::sin(0.7 * i + 0.3);
  }
  Domain<> cpu_domain(mesh, MeshQuadratureRule(q));
  BasisFunction phi_cpu(u_cpu);

  nd::array<double, 3, memory::space::cpu> cpu_iso = evaluate(differentiate(u_cpu), isoparametric(cpu_domain));
  nd::array<double, 3, memory::space::cpu> cpu_spa = evaluate(differentiate(u_cpu), cpu_domain);
  Residual<family> cpu_res_iso = integrate(dot(cpu_iso, differentiate(phi_cpu)), isoparametric(cpu_domain));
  Residual<family> cpu_res_spa = integrate(dot(cpu_iso, differentiate(phi_cpu)), cpu_domain);

  Mesh<memory::space::gpu> d_mesh = mesh;
  Field<family, memory::space::gpu> u = create_field<family>(d_mesh, p, 1);
  u.data = u_cpu.data;
  Domain<memory::space::gpu> domain(d_mesh, MeshQuadratureRule(q));
  BasisFunction phi(u);

  nd::gpu_array<double, 3> iso_gpu = evaluate(differentiate(u), isoparametric(domain));
  nd::gpu_array<double, 3> spa_gpu = evaluate(differentiate(u), domain);
  nd::array<double, 3, memory::space::cpu> iso = iso_gpu;
  nd::array<double, 3, memory::space::cpu> spa = spa_gpu;

  nd::gpu_array<double, 3> f_q = evaluate(differentiate(u), isoparametric(domain));
  Residual<family> res_iso = Residual<family, memory::space::gpu>(integrate(dot(f_q, differentiate(phi)), isoparametric(domain)));
  Residual<family> res_spa = Residual<family, memory::space::gpu>(integrate(dot(f_q, differentiate(phi)), domain));

  EXPECT_LT(rel_err(iso, cpu_iso), tolerance) << "evaluate, isoparametric";
  EXPECT_LT(rel_err(spa, cpu_spa), tolerance) << "evaluate, spatial";
  EXPECT_LT(rel_err(res_iso, cpu_res_iso), tolerance) << "residual, isoparametric";
  EXPECT_LT(rel_err(res_spa, cpu_res_spa), tolerance) << "residual, spatial";
}

Mesh<> quad_mesh() { return Mesh<>::cuboid({7u, 5u}, vec2{1.0, 1.0}); }
Mesh<> hex_mesh()  { return Mesh<>::cuboid({4u, 3u, 3u}, vec3{1.0, 1.0, 1.0}); }
Mesh<> tri_mesh()  { return Mesh<>::disk(vec2{0.0, 0.0}, 1.0, 0.3, 1, Geometry::Triangle); }
Mesh<> tet_mesh()  { return Mesh<>::ball(vec3{0.0, 0.0, 0.0}, 1.0, 0.5, 1, Geometry::Tetrahedron); }

} // namespace

#define MATRIX_FREE_TEST(name, mesh_fn, family, op)                         \
  TEST(matrix_free_cuda, name) {                                            \
    if (!cuda_device_available()) {                                         \
      GTEST_SKIP() << "No CUDA-capable device is available";                \
    }                                                                       \
    auto mesh = mesh_fn();                                                  \
    for (uint32_t p = 1; p <= 2; p++) {                                     \
      check_against_cpu<family, op>(mesh, p, p + 1, #name);                 \
    }                                                                       \
    /* over-integration: a rich rule pushes qpts (and so the block size and  \
       shared memory footprint) well past what the default rule asks for */  \
    check_against_cpu<family, op>(mesh, 1, 6, #name);                       \
  }

MATRIX_FREE_TEST(h1_grad_triangles,    tri_mesh,  Family::H1,    DerivedQuantity::GRAD)
MATRIX_FREE_TEST(h1_grad_quads,        quad_mesh, Family::H1,    DerivedQuantity::GRAD)
MATRIX_FREE_TEST(h1_grad_tets,         tet_mesh,  Family::H1,    DerivedQuantity::GRAD)
MATRIX_FREE_TEST(h1_grad_hexes,        hex_mesh,  Family::H1,    DerivedQuantity::GRAD)

MATRIX_FREE_TEST(hcurl_curl_triangles, tri_mesh,  Family::Hcurl, DerivedQuantity::CURL)
MATRIX_FREE_TEST(hcurl_curl_quads,     quad_mesh, Family::Hcurl, DerivedQuantity::CURL)
MATRIX_FREE_TEST(hcurl_curl_tets,      tet_mesh,  Family::Hcurl, DerivedQuantity::CURL)
MATRIX_FREE_TEST(hcurl_curl_hexes,     hex_mesh,  Family::Hcurl, DerivedQuantity::CURL)
