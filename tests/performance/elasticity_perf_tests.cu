#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "linear_algebra/krylov.hpp"

#include "forall.hpp"

#include "misc/binary_io.hpp"
#include "misc/enzyme_wrapper.hpp"

#include "femto/elasticity_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <string>

using namespace femto;

static constexpr int dim = 3;

struct NeoHookeanModel {

  __host__ __device__ mat3 operator()(mat3 du_dX) const {
    mat3 F = Identity<3>() + du_dX;
    double J = det(F);
    mat3 invFT = transpose(inv(F));
    return (lambda * log(J) - mu) * invFT + mu * F;
  }

  double lambda, mu;
};

struct NeoHookeanJacobianModel {

  // returns dP/dF
  __host__ __device__ auto operator()(mat3 du_dX) const {
    mat3 F = Identity<3>() + du_dX;
    double logJ = log(det(F));
    mat3 invFT = transpose(inv(F));

    mat< 3,3, mat<3,3> > dP_dF;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            dP_dF[i][j][k][l] = lambda * invFT[i][j] * invFT[k][l] + (mu - lambda * logJ) * invFT[k][j] * invFT[i][l] + mu * (i == k) * (j == l);
          }
        }
      }
    }
    return dP_dF;
  }

  double lambda, mu;
};

struct NeoHookeanModelIsoparametric {
  __host__ __device__ mat3 operator()(mat3 du_dxi, mat3 dX_dxi) const {
    mat3 dxi_dX = inv(dX_dxi);
    mat3 du_dX = dot(du_dxi, dxi_dX);
    mat3 F = Identity<3>() + du_dX;
    double J = det(F);
    mat3 invFT = transpose(inv(F));
    mat3 P = (lambda * log(J) - mu) * invFT + mu * F;
    return dot(P, transpose(dxi_dX)) * det(dX_dxi);
  }

  double lambda, mu;
};

int p = 2; // quadratic
double lambda = 10.0;
double mu = 10.0;
uint64_t num_qpts;
NeoHookeanModel material{lambda, mu};
NeoHookeanJacobianModel material_jac{lambda, mu};
NeoHookeanModelIsoparametric isoparametric_material{lambda, mu};

BasisFunction<Family::H1> phi{uint32_t(p), uint32_t(dim)};

////////////////////////////////////////////////////////////////////////////////

template < uint32_t rank >
double normalized_l2_error(
  const std::string & label,
  nd::view<const double, rank> expected,
  nd::view<const double, rank> actual) {
  FEMTO_ASSERT(expected.shape == actual.shape, label + " shape mismatch");

  double error = 0.0;
  double expected_norm = 0.0;
  double actual_norm = 0.0;
  for (uint32_t i = 0; i < expected.size(); i++) {
    double diff = expected[i] - actual[i];
    error += diff * diff;
    expected_norm += expected[i] * expected[i];
    actual_norm += actual[i] * actual[i];
  }

  double scale = std::max(expected_norm, actual_norm);
  return (scale > 0.0) ? std::sqrt(error / scale) : std::sqrt(error);
}

template < uint32_t rank >
double max_abs_difference(
  const std::string & label,
  nd::view<const double, rank> expected,
  nd::view<const double, rank> actual) {
  FEMTO_ASSERT(expected.shape == actual.shape, label + " shape mismatch");

  double max_diff = 0.0;
  for (uint32_t i = 0; i < expected.size(); i++) {
    max_diff = std::max(max_diff, std::abs(expected[i] - actual[i]));
  }
  return max_diff;
}

void print_comparison(
  const std::string & label,
  double l2_error,
  double max_diff,
  double tolerance) {
  std::cout << std::setprecision(16)
            << label << ": normalized_l2_error = " << l2_error
            << ", max_abs_difference = " << max_diff
            << ", tolerance = " << tolerance
            << std::endl;
}

template < memory::space actual_space >
void check_residual_agreement(
  const std::string & label,
  const Residual<Family::H1, memory::space::cpu> & expected,
  const Residual<Family::H1, actual_space> & actual,
  double tolerance) {
  Residual<Family::H1, memory::space::cpu> actual_cpu = actual;
  double l2_error = normalized_l2_error<2>(label, expected.data, actual_cpu.data);
  double max_diff = max_abs_difference<2>(label, expected.data, actual_cpu.data);

  print_comparison(label, l2_error, max_diff, tolerance);
  FEMTO_ASSERT(l2_error <= tolerance, label + " residual mismatch");
}

void check_same_sparsity(
  const std::string & label,
  const sparse_matrix<memory::space::cpu> & expected,
  const sparse_matrix<memory::space::cpu> & actual) {
  FEMTO_ASSERT(expected.nrows == actual.nrows, label + " nrows mismatch");
  FEMTO_ASSERT(expected.ncols == actual.ncols, label + " ncols mismatch");
  FEMTO_ASSERT(expected.nnz == actual.nnz, label + " nnz mismatch");
  FEMTO_ASSERT(expected.row_ptr.size() == actual.row_ptr.size(), label + " row_ptr size mismatch");
  FEMTO_ASSERT(expected.col_ind.size() == actual.col_ind.size(), label + " col_ind size mismatch");

  for (uint32_t i = 0; i < expected.row_ptr.size(); i++) {
    FEMTO_ASSERT(expected.row_ptr[i] == actual.row_ptr[i], label + " row_ptr mismatch at " + std::to_string(i));
  }

  for (uint32_t i = 0; i < expected.col_ind.size(); i++) {
    FEMTO_ASSERT(expected.col_ind[i] == actual.col_ind[i], label + " col_ind mismatch at " + std::to_string(i));
  }
}

template < memory::space actual_space >
void check_sparse_matrix_entries(
  const std::string & label,
  const sparse_matrix<memory::space::cpu> & expected,
  const sparse_matrix<actual_space> & actual,
  double tolerance) {
  sparse_matrix<memory::space::cpu> actual_cpu = actual;
  check_same_sparsity(label, expected, actual_cpu);

  double l2_error = normalized_l2_error<1>(label, expected.values, actual_cpu.values);
  double max_diff = max_abs_difference<1>(label, expected.values, actual_cpu.values);

  print_comparison(label, l2_error, max_diff, tolerance);
  FEMTO_ASSERT(l2_error <= tolerance, label + " sparse matrix entry mismatch");
}

////////////////////////////////////////////////////////////////////////////////

Residual<Family::H1, memory::space::cpu> calculate_residual_cpu(const Mesh< memory::space::cpu > & mesh) {
  Domain< memory::space::cpu > domain(mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::cpu> u = create_field<Family::H1>(mesh, p, dim);

  num_qpts = total(domain.num_qpts);

  nd::array<double, 3, memory::space::cpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 3, memory::space::cpu > P_q = forall(material, du_dX_q);
  return integrate(dot(P_q, grad(phi)), domain);
}

sparse_matrix< memory::space::cpu > calculate_stiffness_cpu(const Mesh< memory::space::cpu > & mesh) {
  Domain< memory::space::cpu > domain(mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::cpu> u = create_field<Family::H1>(mesh, p, dim);

  nd::array<double, 3, memory::space::cpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 5, memory::space::cpu > dP_dF_q = forall(material_jac, du_dX_q);
  return integrate(dot(grad(phi), dP_dF_q, grad(phi)), domain);
}

////////////////////////////////////////////////////////////////////////////////

Residual<Family::H1, memory::space::cpu> calculate_residual_gpu_1(const Mesh< memory::space::cpu > & h_mesh) {
  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Domain< memory::space::gpu > domain(d_mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p, dim);

  nd::array<double, 3, memory::space::gpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 3, memory::space::gpu > P_q = forall(material, du_dX_q);
  Residual<Family::H1, memory::space::gpu> r = integrate(dot(P_q, grad(phi)), domain);
  return r;  // device to host
}

Residual<Family::H1, memory::space::cpu> calculate_residual_gpu_2(const Mesh< memory::space::cpu > & h_mesh) {
  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Domain< memory::space::gpu > domain(d_mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p, dim);

  nd::array<double, 3, memory::space::gpu> d_dX_dxi_q = evaluate(grad(d_mesh.X), isoparametric(domain));
  nd::array<double, 3, memory::space::gpu> d_du_dxi_q = evaluate(grad(u), isoparametric(domain));
  nd::array<double, 3, memory::space::gpu> d_P_xi_q = forall(isoparametric_material, d_du_dxi_q, d_dX_dxi_q);
  Residual<Family::H1, memory::space::gpu> r = integrate(dot(d_P_xi_q, grad(phi)), isoparametric(domain));
  return r;  // device to host
}

////////////////////////////////////////////////////////////////////////////////

sparse_matrix< memory::space::gpu > calculate_stiffness_gpu_1(const Mesh< memory::space::cpu > & h_mesh) {
  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Domain< memory::space::gpu > domain(d_mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p, dim);

  nd::array<double, 3, memory::space::gpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 5, memory::space::gpu > dP_dF_q = forall(material_jac, du_dX_q);
  return integrate(dot(grad(phi), dP_dF_q, grad(phi)), domain);
}

sparse_matrix< memory::space::gpu > calculate_stiffness_gpu_2(const Mesh< memory::space::cpu > & h_mesh) {
  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Domain< memory::space::gpu > domain(d_mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p, dim);

  nd::array<double, 3, memory::space::gpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 5, memory::space::gpu > dP_dF_q = forall(material_jac, du_dX_q);
  return integrate(dot(grad(phi), dP_dF_q, grad(phi)), isoparametric(domain));
}

sparse_matrix< memory::space::gpu > calculate_stiffness_gpu_3(const Mesh< memory::space::cpu > & h_mesh) {
  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Domain< memory::space::gpu > domain(d_mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p, dim);

  nd::array<double, 3, memory::space::gpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 5, memory::space::gpu > dP_dF_q = forall(material_jac, du_dX_q);

  sparse_matrix< memory::space::gpu > K;
  femto::impl::stiffness_cuda::integrate_h1_tet_spmat(
    K,
    grad(phi),
    dP_dF_q,
    grad(phi),
    domain,
    DomainType::SPATIAL);
  return K;
}

sparse_matrix< memory::space::gpu > calculate_stiffness_gpu_4(const Mesh< memory::space::cpu > & h_mesh) {
  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Domain< memory::space::gpu > domain(d_mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p, dim);

  nd::array<double, 3, memory::space::gpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 5, memory::space::gpu > dP_dF_q = forall(material_jac, du_dX_q);

  sparse_matrix< memory::space::gpu > K;
  femto::impl::stiffness_cuda::integrate_h1_tet_spmat(
    K,
    grad(phi),
    dP_dF_q,
    grad(phi),
    domain,
    DomainType::ISOPARAMETRIC);
  return K;
}

nd::array<double, 5, memory::space::gpu > calculate_stiffness_gpu_5(const Mesh< memory::space::cpu > & h_mesh) {
  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Domain< memory::space::gpu > domain(d_mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p, dim);

  nd::array<double, 3, memory::space::gpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 5, memory::space::gpu > dP_dF_q = forall(material_jac, du_dX_q);

  nd::array<double, 5, memory::space::gpu > K;
  femto::impl::stiffness_cuda::integrate_h1_tet_emat(
    K,
    grad(phi),
    dP_dF_q,
    grad(phi),
    domain,
    DomainType::SPATIAL);
  return K;
}



nd::array<double, 5, memory::space::gpu > calculate_stiffness_gpu_6(const Mesh< memory::space::cpu > & h_mesh) {
  Mesh< memory::space::gpu > d_mesh = h_mesh;
  Domain< memory::space::gpu > domain(d_mesh, MeshQuadratureRule(p));
  Field<Family::H1, memory::space::gpu> u = create_field<Family::H1>(d_mesh, p, dim);

  nd::array<double, 3, memory::space::gpu > du_dX_q = evaluate(grad(u), domain);
  nd::array<double, 5, memory::space::gpu > dP_dF_q = forall(material_jac, du_dX_q);

  nd::array<double, 5, memory::space::gpu > K;
  femto::impl::stiffness_cuda::integrate_h1_tet_emat(
    K,
    grad(phi),
    dP_dF_q,
    grad(phi),
    domain,
    DomainType::ISOPARAMETRIC);
  return K;
}

////////////////////////////////////////////////////////////////////////////////

int main() {

  std::cout << "loading mesh ... " << std::flush;
  ///Mesh< memory::space::cpu > h_mesh = Mesh< memory::space::cpu >::load("/home/sam/code/femto/linear_tets_only.msh");
  Mesh< memory::space::cpu > h_mesh = Mesh< memory::space::cpu >::load(FEMTO_DATA_DIR "/meshes/ball1000000.msh");
  std::cout << "finished" << std::endl;

  std::cout << "mesh has:" << std::endl;
  std::cout << "  " << h_mesh.vert.shape[0] << " vertices" << std::endl;
  std::cout << "  " << h_mesh.edge.shape[0] << " edges" << std::endl;
  std::cout << "  " << h_mesh.tri.shape[0] << " triangles" << std::endl;
  std::cout << "  " << h_mesh.quad.shape[0] << " quadrilaterals" << std::endl;
  std::cout << "  " << h_mesh.tet.shape[0] << " tetrahedra" << std::endl;
  std::cout << "  " << h_mesh.hex.shape[0] << " hexahedra" << std::endl;

  constexpr double comparison_tolerance = 1.0e-8;

  Residual<Family::H1, memory::space::cpu> h_r = calculate_residual_cpu(h_mesh);
  {
    Residual<Family::H1, memory::space::gpu> d_r1 = calculate_residual_gpu_1(h_mesh);
    check_residual_agreement("h_r vs d_r1", h_r, d_r1, comparison_tolerance);
  }

  {
    Residual<Family::H1, memory::space::gpu> d_r2 = calculate_residual_gpu_2(h_mesh);
    check_residual_agreement("h_r vs d_r2", h_r, d_r2, comparison_tolerance);
  }

  sparse_matrix< memory::space::cpu > h_K = calculate_stiffness_cpu(h_mesh);

  {
    sparse_matrix< memory::space::gpu > d_K1 = calculate_stiffness_gpu_1(h_mesh);
    check_sparse_matrix_entries("h_K vs d_K1", h_K, d_K1, comparison_tolerance);
  }

  {
    sparse_matrix< memory::space::gpu > d_K2 = calculate_stiffness_gpu_2(h_mesh);
  }

  {
    sparse_matrix< memory::space::gpu > d_K3 = calculate_stiffness_gpu_3(h_mesh);
    check_sparse_matrix_entries("h_K vs d_K3", h_K, d_K3, comparison_tolerance);
  }

  {
    sparse_matrix< memory::space::gpu > d_K4 = calculate_stiffness_gpu_4(h_mesh);
  }

  {
    nd::array<double, 5, memory::space::gpu > d_K5 = calculate_stiffness_gpu_5(h_mesh);
  }

  {
    nd::array<double, 5, memory::space::gpu > d_K6 = calculate_stiffness_gpu_6(h_mesh);
  }

  std::cout << num_qpts << " quadrature points" << std::endl;

  std::cout << num_qpts * 9 * 8 << " bytes to store dxi/dX" << std::endl;
  std::cout << num_qpts * 81 * 8 << " bytes to store dP/dF" << std::endl;

  FiniteElement<Geometry::Tetrahedron, Family::H1> elem{(uint32_t)p};

  std::cout << h_mesh.tet.shape[0] * elem.num_nodes() * elem.num_nodes() * 9 * 8 << " bytes for element stiffness matrices" << std::endl;;

  std::cout << h_K.nnz * 8 << " bytes for sparse matrix (" << h_K.nnz << " nonzeros)" << std::endl;

}
