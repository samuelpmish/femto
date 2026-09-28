#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"
#include "materials/neohookean.hpp"
#include "materials/plane_strain.hpp"

#include "linear_algebra/krylov.hpp"
#include "linear_algebra/sparse_direct.hpp"

#include "forall.hpp"

#include "misc/binary_io.hpp"
#include "misc/enzyme_wrapper.hpp"

#include <functional>

using namespace femto;

int main() {

  femto::timer stopwatch;

  constexpr int dim = 2;

  //Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR"I.msh");
  Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR"soft_structure.msh");
  //Mesh<> mesh = Mesh<>::load("/Users/sam/Dropbox/meshes/damper2D.msh");
  //Mesh<> mesh = Mesh<>::load("/home/sam/Dropbox/meshes/damper2D.msh");

  std::cout << mesh.vert.shape[0] << " vertices" << std::endl;
  std::cout << mesh.edge.shape[0] << " edges" << std::endl;
  std::cout << mesh.tri.shape[0] << " triangles" << std::endl;

  double lambda = 10.0;
  double mu = 10.0;
  PlaneStrain<NeoHookeanModel> material{{lambda, mu}};

  int polynomial_order = 2;
  int num_components = dim;
  Field u = create_field<Family::H1>(mesh, polynomial_order, num_components);

  BasisFunction phi(u);

  Domain domain(mesh, MeshQuadratureRule(polynomial_order + 1));

  // the work arrays live outside the lambdas, so every evaluation refills them in place
  nd::cpu_array<double, 3> du_dX_q, P_q;
  nd::cpu_array<double, 5> dP_dF_q;
  Residual<Family::H1> r_u;

  auto residual = [&](const Field<Family::H1> & u_) -> const Residual<Family::H1> & {
    du_dX_q = evaluate(grad(u_), domain);
    P_q = forall(std::function<mat2(const mat2 &)>([&](const mat2 & du_dX){
      return material(du_dX);
    }), du_dX_q);
    r_u = integrate(dot(P_q, grad(phi)), domain);
    return r_u;
  };

  auto stiffness_matrix = [&](const Field<Family::H1> & u){
    du_dX_q = evaluate(grad(u), domain);
    dP_dF_q = forall(std::function< mat<2,2, mat2>(const mat2 &) >([&](const mat2 & du_dX){
      return material.jac(du_dX);
    }), du_dX_q);
    return integrate(dot(grad(phi), dP_dF_q, grad(phi)), domain);
  };

  double epsilon = 0.001;
  std::vector<int> bc_dofs;
  std::vector<int> nonhomogeneous_bc_dofs;
  nd::cpu_array<double, 2> nodes = nodes_for(u, mesh); 
  for (int i = 0; i < nodes.shape[0]; i++) {
    vec2 X = load<vec2>(nodes, i);
    bool bottom = (X[1] < (-1.0 + epsilon));
    bool top = (X[1] > (+1.0 - epsilon));
    bool top_left = (X[1] > 0.9) && (X[0] < (-1.0 + epsilon));
    bool bottom_right = (X[1] < -0.9) && (X[0] > (1.0 - epsilon));

    if (bottom || top) {
      bc_dofs.push_back(2 * i + 1); // y component of node i
    }

    if (bottom_right || top_left) {
      bc_dofs.push_back(2 * i + 0); // x component of node i
    }

    if (bottom_right) {
      nonhomogeneous_bc_dofs.push_back(2 * i + 0); // x component of node i
    }
  }

  int ndof = u.size();
  double umax = 0.50;
  int num_load_steps = 64;
  double step_limit = sqrt(ndof) * umax / num_load_steps;

  /*
    ⎛ X X           ⎞
    ⎜ X X X         ⎜
    ⎜   X X X       ⎜
    ⎜     X X X     ⎜
    ⎜       X X X   ⎜
    ⎜         X X X ⎜
    ⎝           X X ⎠
           K0        
  */
  sparse_matrix K0 = stiffness_matrix(u);

  /*
         bc_dofs
          V V V
    ⎛ X X⎛     ⎞    ⎞
      X X⎜X    ⎜    
    ⎜   X⎜X X  ⎜    ⎜
         ⎜X X X⎜    
    ⎜    ⎜  X X⎜X   ⎜
         ⎜    X⎜X X 
    ⎝    ⎝     ⎠X X ⎠
        Kessential    
  */
  sparse_matrix Kessential = K0({}, bc_dofs);

  /*
         bc_dofs
          V V V      
    ⎛ X X           ⎞
    ⎜ X X 0         ⎜
  > ⎜   0 1 0       ⎜
  > ⎜     0 1 0     ⎜
  > ⎜       0 1 0   ⎜
    ⎜         0 X X ⎜
    ⎝           X X ⎠
        Kmodified    
  */
  sparse_matrix Kmodified = K0;
  Kmodified({}, bc_dofs) = [](int i, int j){ return (i == j); };
  Kmodified(bc_dofs, {}) = [](int i, int j){ return (i == j); };

  auto invK = inv(Kmodified);

  // load stepping
  std::vector< std::array<double,2> > force_displacement_curve;
  force_displacement_curve.push_back({0, 0});
  for (int k = 0; k < num_load_steps; k++) {

    float t = float(k + 1) / num_load_steps;

    vector u_target = zeros(K0.nrows);
    u_target[nonhomogeneous_bc_dofs] = umax * t;

    // "newton" loop
    for (int i = 0; i < 10; i++) {
      vector u_error = u.v() - u_target;
      vector f = residual(u).v() - dot(Kessential, u_error[bc_dofs]);
      f[bc_dofs] = u_error[bc_dofs];
      std::cout << "  prescribed displacement error: " << norm(u_error[bc_dofs]) << std::endl;

      vector delta_u = dot(invK, f);
      float norm_delta_u = norm(delta_u);
      if (norm_delta_u > step_limit) {
        delta_u *= step_limit / norm_delta_u;
      }
      u.v() -= delta_u;

      vector r = residual(u).v();
      r[bc_dofs] = 0.0;
      std::cout << "  residual norm: " << norm(r) << std::endl;
    }

    vector r = residual(u).v();
    force_displacement_curve.push_back({umax * t, total(r[nonhomogeneous_bc_dofs])});

    K0 = stiffness_matrix(u);

    Kessential = K0({}, bc_dofs);

    Kmodified = K0;
    Kmodified({}, bc_dofs) = [](int i, int j){ return (i == j); };
    Kmodified(bc_dofs, {}) = [](int i, int j){ return (i == j); };

    invK.update(Kmodified);

  }

  for (auto [u, f] : force_displacement_curve) {
    std::cout << u << ", " << f << std::endl;
  }

}