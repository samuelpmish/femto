#if 0
// 2D compressible hydro example (triple-point-like shock problem) that
// publishes its solution fields to a browser-based viewer via femto::server
// (see data/viewer.html for the client)
//
// usage: hydro_2D [port]   (default: 8080)

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "femto/geometry.hpp"
#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"
#include "server.hpp"

#include "fm/types/vec.hpp"
#include "fm/types/matrix.hpp"

#include "linear_algebra/sparse_direct.hpp"

using namespace fm;
using namespace femto;

constexpr uint32_t dim = 2;

Mesh<> rectangle_mesh(float h, int p) {

  double LX = 7.0;
  double LY = 3.0;

  stack::array< uint32_t, 2 > divisions = {
    uint32_t(7 * ceil(LX / (7 * h))),
    uint32_t(2 * ceil(LY / (2 * h)))
  };

  auto mesh = Mesh<>::cuboid(divisions, vec2{LX, LY});

  if (p > 1) {
    Field X_p = create_field<Family::H1>(mesh, p, dim);
    X_p.data = nodes_for(X_p, mesh);
    mesh.X = X_p;
  }

  return mesh;

}

struct Hydro2D {

  Hydro2D(float h, int p);

  int p;
  float h;

  Mesh<> mesh;

  Field<Family::H1> displacement;
  Field<Family::H1> velocity;
  Field<Family::DG> density;
  Field<Family::DG> energy;

  nd::cpu_array<double, 1> length_q;
  nd::cpu_array<double, 1> gamma_q;
  std::vector<int> homogeneous_bcs;

  sparse_factorization invM_v;
  sparse_factorization invM_e;

  double t;

  void initial_conditions();
  void simulate(double dt, int num_steps);

};

Hydro2D::Hydro2D(float h, int p) : p(p), h(h) {
  mesh = rectangle_mesh(h, p);
  initial_conditions();
}

void Hydro2D::initial_conditions() {

  vec2 * X = reinterpret_cast< vec2 * >(&mesh.X.data[0]);

  displacement = create_field<Family::H1>(mesh, p, dim);
  velocity = create_field<Family::H1>(mesh, p, dim);
  density = create_field<Family::DG>(mesh, p-1, 1);
  energy = create_field<Family::DG>(mesh, p-1, 1);

  MeshQuadratureRule qrule(p+1);
  Domain domain = Domain(mesh, qrule);

  uint32_t num_quadrature_points = total(domain.num_qpts);

  length_q.resize(num_quadrature_points);
  for (uint32_t i = 0; i < num_quadrature_points; i++) {
    length_q[i] = h;
  }

  nd::cpu_array<double, 3> X_q = evaluate(mesh.X, domain);
  gamma_q.resize(num_quadrature_points);

  for (uint32_t i = 0; i < num_quadrature_points; i++) {
    if (X_q(i, 0, 0) < 1.0) {
      gamma_q[i] = 1.5;
    } else {
      if (X_q(i, 1, 0) < 1.5) {
        gamma_q[i] = 1.4;
      } else {
        gamma_q[i] = 1.5;
      }
    }
  }

  velocity.data = forall(+[](const vec2 & X){ return vec2{0.0, 0.0}; }, mesh.X.data);
  displacement.data = forall(+[](const vec2 & X){ return vec2{0.0, 0.0}; }, mesh.X.data);

  FiniteElement<Geometry::Quadrilateral, Family::DG> dg_element{uint32_t(p-1)};

  std::vector< uint32_t > ids(dg_element.num_nodes());
  uint32_t num_quads = mesh.quad.shape[0];
  for (uint32_t i = 0; i < num_quads; i++) {
    vec2 center = 0.25 * (X[mesh.quad(i, 0).index] + X[mesh.quad(i, 1).index] + X[mesh.quad(i, 2).index] + X[mesh.quad(i, 3).index]);

    dg_element.indices(energy.offsets, &mesh.quad(i, 0), &ids[0]);

    if (center[0] < 1.0) {
      for (auto id : ids) {
        density.data[id] = 1.0;
        energy.data[id] = 2.0; // pressure = 1
      }
    } else {
      if (center[1] < 1.5) {
        for (auto id : ids) {
          density.data[id] = 1.0;
          energy.data[id] = 0.25; // pressure = 0.1
        }
      } else {
        for (auto id : ids) {
          density.data[id] = 0.125;
          energy.data[id] = 1.6; // pressure = 0.1
        }
      }
    }
  }

  homogeneous_bcs.clear();
  uint32_t nnodes = mesh.X.data.shape[0];
  for (uint32_t i = 0; i < nnodes; i++) {
    if (mesh.X.data(i, 0) < 0.01 || mesh.X.data(i, 0) > 6.99) {
      homogeneous_bcs.push_back(2*i+0);
    }

    if (mesh.X.data(i, 1) < 0.01 || mesh.X.data(i, 1) > 2.99) {
      homogeneous_bcs.push_back(2*i+1);
    }
  }

  velocity.v()[homogeneous_bcs] = 0.0;

  BasisFunction psi(energy);
  BasisFunction phi(velocity);

  // precompute and factorize mass matrices,
  // since they don't change during simulation
  nd::cpu_array<double, 3> rho_q = evaluate(density, domain);
  sparse_matrix M_e = integrate(dot(psi, rho_q, psi), domain);
  M_e.symmetry = Symmetry::Symmetric;
  M_e.definiteness = Definiteness::PositiveDefinite;
  invM_e = inv(M_e);

  auto rho_I_q = forall(+[](const double & rho){
    return mat2(rho * Identity<dim>());
  }, rho_q);

  sparse_matrix M_v = integrate(dot(phi, rho_I_q, phi), domain);
  M_v({}, homogeneous_bcs) = [](int i, int j){ return (i == j); };
  M_v(homogeneous_bcs, {}) = [](int i, int j){ return (i == j); };
  M_v.symmetry = Symmetry::Symmetric;
  M_v.definiteness = Definiteness::PositiveDefinite;
  invM_v = inv(M_v);

  t = 0.0;

}

void Hydro2D::simulate(double dt, int num_steps) {

  MeshQuadratureRule qrule(p+1);
  Domain domain = Domain(mesh, qrule);

  BasisFunction psi(energy);
  BasisFunction phi(velocity);

  auto calculate_stress = +[](const double & rho0,
                              const double & length0,
                              const double & gamma,
                              const double & energy,
                              const mat2 & du_dX,
                              const mat2 & dv_dX) {
    mat2 F = Identity<dim>() + du_dX;
    double J = det(F);
    double rho = rho0 / J;

    mat2 L = dot(dv_dX, inv(F));
    mat2 D = 0.5 * (L + transpose(L));

    double q[2] = {0.25, 0.66};
    double c = sqrt(energy * gamma * (gamma - 1)); // speed of sound
    double length = length0 * sqrt(J);
    double v_jump = tr(L) * length;

    double linear_term = q[0] * length * c;
    double quadratic_term = q[1] * length * std::abs(v_jump);
    double compression_switch = 1.0 / (1.0 + exp(-v_jump / (0.2 * c)));

    double mu = 0.75 * rho * (linear_term + quadratic_term) * compression_switch;

    double p = rho * energy * (gamma - 1); // pressure

    return -p * Identity<dim>() + mu * D;
  };

  auto cauchy_to_PK1 = +[](const mat2 & sigma, const mat2 & du_dX){
    mat2 F = Identity<dim>() + du_dX;
    double J = det(F);
    return J * dot(sigma, transpose(inv(F)));
  };

  auto energy_source_term = +[](const mat2 & sigma, const mat2 & du_dX, const mat2 & dv_dX){
    mat2 F = Identity<dim>() + du_dX;
    mat2 L = dot(dv_dX, inv(F));
    return ddot(sigma, L);
  };

  nd::cpu_array<double, 3> rho_q = evaluate(density, domain);

  for (int k = 0; k < num_steps; k++) {

    femto::vector displacement_n = displacement.v();
    femto::vector velocity_n = velocity.v();
    femto::vector energy_n = energy.v();

    nd::cpu_array<double, 3> du_dX_q = evaluate(grad(displacement), domain);
    nd::cpu_array<double, 3> dv_dX_q = evaluate(grad(velocity), domain);
    nd::cpu_array<double, 3> e_q = evaluate(energy, domain);

    nd::cpu_array<double, 3> sigma_q = forall(calculate_stress, rho_q, length_q, gamma_q, e_q, du_dX_q, dv_dX_q);
    nd::cpu_array<double, 3> P_q = forall(cauchy_to_PK1, sigma_q, du_dX_q);

    // v^{n+½}
    Residual<Family::H1> r_v = integrate(dot(P_q, grad(phi)), domain);
    r_v.v()[homogeneous_bcs] = 0;
    velocity.v() = vector(velocity.v() - (0.5 * dt) * dot(invM_v, r_v.v()));

    // e^{n+½}
    dv_dX_q = evaluate(grad(velocity), domain);
    nd::cpu_array<double, 2> e_source_q = forall(energy_source_term, sigma_q, du_dX_q, dv_dX_q);
    Residual<Family::DG> r_e = integrate(e_source_q * psi, domain);
    energy.v() = vector(energy.v() + (0.5 * dt) * dot(invM_e, r_e.v()));

    // u^{n+½}
    displacement.v() = vector(displacement.v() + (0.5 * dt) * velocity.v());

////////////////////////////////////////////////////////////////////////////////

    // evaluate quadrature point quantities at the half step
    du_dX_q = evaluate(grad(displacement), domain);
    dv_dX_q = evaluate(grad(velocity), domain);
    e_q = evaluate(energy, domain);

    // v^{n+1}
    sigma_q = forall(calculate_stress, rho_q, length_q, gamma_q, e_q, du_dX_q, dv_dX_q);
    P_q = forall(cauchy_to_PK1, sigma_q, du_dX_q);
    r_v = integrate(dot(P_q, grad(phi)), domain);
    r_v.v()[homogeneous_bcs] = 0;
    femto::vector velocity_n_plus_1 = velocity_n - dt * dot(invM_v, r_v.v());

    // \bar{v}^{n+½}
    velocity.v() = vector(0.5 * (velocity_n + velocity_n_plus_1));
    dv_dX_q = evaluate(grad(velocity), domain);

    // e^{n+1}
    e_source_q = forall(energy_source_term, sigma_q, du_dX_q, dv_dX_q);
    r_e = integrate(e_source_q * psi, domain);
    energy.v() = vector(energy_n + dt * dot(invM_e, r_e.v()));

    // u^{n+1}
    displacement.v() = vector(displacement_n + dt * velocity.v());

    velocity.v() = velocity_n_plus_1;

    t += dt;

  }

}

int main(int argc, char* argv[]) {

  int port = (argc > 1) ? std::atoi(argv[1]) : 8080;
  if (port <= 0 || port > 65535) {
    std::cout << "usage: " << argv[0] << " [port]" << std::endl;
    return 1;
  }

  server srv({.port = port});

  Hydro2D sim(/* h = */ 0.125f, /* p = */ 2);
  srv.set_mesh(sim.mesh);

  std::cout << sim.mesh.tri.shape[0] + sim.mesh.quad.shape[0] << " elements" << std::endl;
  std::cout << sim.velocity.size() + sim.energy.size() << " degrees of freedom ("
            << sim.velocity.size() << " velocity, "
            << sim.energy.size() << " energy)" << std::endl;

  double dt = 0.0025 * sim.h;
  int num_steps = 2500;

  srv.push_field("displacement", sim.displacement);
  srv.push_field("energy", sim.energy);

  for (int k = 0; k < num_steps; k++) {
    sim.simulate(dt, 1);

    if ((k + 1) % 25 == 0) {
      std::cout << "step " << (k + 1) << " / " << num_steps << ", t = " << sim.t << std::endl;
      srv.push_field("displacement", sim.displacement);
      srv.push_field("energy", sim.energy);
    }
  }

  std::cout << "simulation finished, view results at " << srv.url() << std::endl;
  srv.wait();

  return 0;

}

#else

int main() { return 0; }

#endif