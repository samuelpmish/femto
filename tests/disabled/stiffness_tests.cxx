#include <gtest/gtest.h>

#include "femto/mesh.hpp"

#include "forall.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

#include "linear_algebra/vector.hpp"

#include "containers/dual.hpp"
#include "containers/ndarray_conversions.hpp"

#include "femto/element_index.hpp"

#include <iostream>
#include <functional>

void batch_integrate_jacobian_tri(nd::view<double, 5> element_jacobians,
                                  nd::view<const double, 6> qdata, 
                                  Family test_family,
                                  uint64_t test_p,
                                  Family trial_family,
                                  uint64_t trial_p,
                                  nd::view<const uint64_t, 2> tris,
                                  nd::view<const double, 2> qpts,
                                  nd::view<const double> qwts);

int main() {

  static constexpr int dim = 2;
  static constexpr int k = 2.0;

  uint64_t p = 2;
  uint64_t q = 3;

  Mesh<> mesh = Mesh<>::load(FEMTO_DATA_DIR"meshes/patch_test_tris.json");

  QuadratureRule rule = gauss_legendre_rule(q);

  Field u = create_field(mesh, Family::H1, p);
  Field v = create_field(mesh, Family::H1, p);
  u = forall(std::function<double(vec2)>([](auto x){ return x[0]; }), nodes_for(u, mesh));

  auto dx_dxi_q = gradient_wrt_xi(mesh.X, mesh, rule);

  auto r = [&](Field u){
    auto du_dxi_q = gradient_wrt_xi(u, mesh, rule);

    auto heat_flux_q = forall(std::function< mat<1,2>(vec2, mat2) >([](vec2 du_dxi, mat2 dx_dxi) {
      mat2 dxi_dx = inv(dx_dxi);
      vec2 du_dx = dot(du_dxi, dxi_dx);
      vec2 heat_flux = -k * du_dx * norm(du_dx);
      return mat<1,dim>{dot(heat_flux, transpose(dxi_dx)) * det(dx_dxi)};
    }), du_dxi_q, dx_dxi_q);

    return integrate_flux(heat_flux_q, mesh, rule, u.family, u.degree);
  };

  auto drdu = [&](Field u){
    auto du_dxi_q = gradient_wrt_xi(u, mesh, rule);

    auto dheat_flux_dgradu_q = forall(std::function< mat2(vec2, mat2) >([](vec2 du_dxi, mat2 dx_dxi) {
      mat2 dxi_dx = inv(dx_dxi);
      auto du_dx = dot(gradient_wrt(du_dxi), dxi_dx);
      auto heat_flux = -k * du_dx * norm(du_dx);
      auto output = dot(heat_flux, transpose(dxi_dx)) * det(dx_dxi); 
      return get_gradient(output);
    }), du_dxi_q, dx_dxi_q);

    FunctionSpace test{Family::H1, p};
    FunctionSpace trial{Family::H1, p};

    GeometryInfo counts = mesh.geometry_counts();
    GeometryInfo offsets_test = scan(counts * dofs_per_geom(test.family, test.degree));
    GeometryInfo offsets_trial = scan(counts * dofs_per_geom(trial.family, trial.degree));

    uint64_t n_test = (test.degree + 1) * (test.degree + 2) / 2;
    uint64_t n_trial = (trial.degree + 1) * (trial.degree + 2) / 2;

    uint64_t num_tris = mesh.tri.shape[0];
    uint64_t Q_tri = rule.qpts_tri.shape[0];
    nd::cpu_array<double, 5> element_jacobians({num_tris, n_trial, 1, n_test, 1});

    batch_integrate_jacobian_tri(element_jacobians,
                                 ndview(&dheat_flux_dgradu_q(0, 0, 0), {num_tris, Q_tri, 1, 2, 1, 2}), 
                                 test.family,
                                 test.degree,
                                 trial.family,
                                 trial.degree,
                                 mesh.tri,
                                 rule.qpts_tri,
                                 rule.qwts_tri);

    std::vector< femto::triplet > triplets(element_jacobians.size());

    std::vector < uint64_t > test_ids(n_test);
    std::vector < uint64_t > trial_ids(n_trial);
    for (int e = 0; e < num_tris; e++) {
      element_indices_H1_tri(test.degree, offsets_trial, &mesh.tri(e, 0), &test_ids[0]);
      element_indices_H1_tri(trial.degree, offsets_trial, &mesh.tri(e, 0), &trial_ids[0]);
      for (int i = 0; i < n_trial; i++) {
        for (int j = 0; j < n_test; j++) {
          triplets[(e * n_trial + i) * n_test + j] = {int(test_ids[i]), int(trial_ids[j]), element_jacobians(e, i, 0, j, 0)};
        }
      }
    }
    
    uint64_t nrows = v.size();
    uint64_t ncols = u.size();

    return femto::sparse_matrix<>::from_triplets(triplets, nrows, ncols);

  };

  femto::vector dx(u.data.size());

  double epsilon = 1.0e-8;

  Field u2 = create_field(mesh, Family::H1, p);

  for (int i = 0; i < u.data.size(); i++) {
    dx[i] = sin(i);
    u2.data(i,0) = u.data(i, 0) + dx[i] * epsilon;
  }

  auto K = drdu(u);
  auto difference = r(u2) - r(u);

  nd::cpu_array<double, 2> dr1 = difference * (1.0 / epsilon);
  femto::vector dr2 = dot(K, dx);

  double error = 0.0;
  double norm = 0.0;
  for (int i = 0; i < u.data.size(); i++) {
    error += (dr1(i, 0) - dr2[i]) * (dr1(i, 0) - dr2[i]);
    norm += (dr1(i, 0) * dr1(i, 0));
  }
  std::cout << sqrt(error / norm) << std::endl;

  femto::export_matrix_market(K, "K.mtx");

}