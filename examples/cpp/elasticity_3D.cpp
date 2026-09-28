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

  constexpr int dim = 3;

  //Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR"I.msh");
  //Mesh<> mesh = Mesh<>::load(FEMTO_MESH_DIR"soft_structure.msh");
  //Mesh<> mesh = Mesh<>::load("/Users/sam/Dropbox/meshes/damper2D.msh");
  Mesh<> mesh = Mesh<>::load("/Users/sam/Dropbox/meshes/logic_gate_hex.msh");

  std::cout << mesh.vert.shape[0] << " vertices" << std::endl;
  std::cout << mesh.edge.shape[0] << " edges" << std::endl;
  std::cout << mesh.tri.shape[0] << " triangles" << std::endl;
  std::cout << mesh.quad.shape[0] << " quads" << std::endl;
  std::cout << mesh.tet.shape[0] << " tets" << std::endl;
  std::cout << mesh.hex.shape[0] << " hexes" << std::endl;

  double lambda = 10.0;
  double mu = 10.0;
  NeoHookeanModel material{lambda, mu};

  int polynomial_order = 2;
  int num_components = dim;
  Field u = create_field<Family::H1>(mesh, polynomial_order, num_components);

  std::cout << u.data.size() << " dofs" << std::endl;

  BasisFunction phi(u);

  Domain domain(mesh, MeshQuadratureRule(polynomial_order + 1));

  // the work arrays live outside the lambdas, so every evaluation refills them in place
  nd::cpu_array<double, 3> du_dX_q, P_q;
  nd::cpu_array<double, 5> dP_dF_q;
  Residual<Family::H1> r_u;

  auto residual = [&](const Field<Family::H1> & u_) -> const Residual<Family::H1> & {
    du_dX_q = evaluate(grad(u_), domain);
    P_q = forall(std::function<mat3(const mat3 &)>([&](const mat3 & du_dX){
      return material(du_dX);
    }), du_dX_q);
    r_u = integrate(dot(P_q, grad(phi)), domain);
    return r_u;
  };

  auto stiffness_matrix = [&](const Field<Family::H1> & u){
    du_dX_q = evaluate(grad(u), domain);
    dP_dF_q = forall(std::function< mat<3,3, mat3>(const mat3 &) >([&](const mat3 & du_dX){
      return material.jac(du_dX);
    }), du_dX_q);
    return integrate(dot(grad(phi), dP_dF_q, grad(phi)), domain);
  };

  double epsilon = 0.001;
  std::vector<int> bc_dofs;
  std::vector<int> nonhomogeneous_bc_dofs;
  nd::cpu_array<double, 2> nodes = nodes_for(u, mesh); 
  for (int i = 0; i < nodes.shape[0]; i++) {
    vec3 X = load<vec3>(nodes, i);
    bool left = (X[0] < 0.01);
    bool right = (X[0] > 8.79);
    bool top_right = (X[1] > 7.19) && (X[0] > 7.5);

    if (left || right) {
      bc_dofs.push_back(3 * i + 0); // x component of node i
      bc_dofs.push_back(3 * i + 1); // y component of node i
      bc_dofs.push_back(3 * i + 2); // z component of node i
    }

    if (top_right) {
      bc_dofs.push_back(3 * i + 1); // z component of node i
      nonhomogeneous_bc_dofs.push_back(3 * i + 1); // z component of node i
    }

  }

  int ndof = u.size();
  double umax = 2.00;
  int num_load_steps = 256;
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

  std::cout << bc_dofs.size() << " homogeneous bcs" << std::endl;
  std::cout << nonhomogeneous_bc_dofs.size() << " nonhomogeneous bcs" << std::endl;

  Kmodified.symmetry = Symmetry::Symmetric;
  Kmodified.definiteness = Definiteness::PositiveDefinite;
  auto invK = inv(Kmodified); // Cholesky, since Kmodified is SPD

  // load stepping
  std::vector< std::array<double,2> > force_displacement_curve;
  force_displacement_curve.push_back({0, 0});
  for (int k = 0; k < num_load_steps; k++) {

    std::cout << "k = " << k << std::endl;
    float t = float(k + 1) / num_load_steps;

    vector u_target = zeros(K0.nrows);
    u_target[nonhomogeneous_bc_dofs] = umax * t;

    // "newton" loop
    for (int i = 0; i < 10; i++) {
      vector u_error = u.v() - u_target;
      vector f = residual(u).v() - dot(Kessential, u_error[bc_dofs]);
      f[bc_dofs] = u_error[bc_dofs];
      std::cout << " |u_p|: " << norm(u_error[bc_dofs]);

      vector delta_u = dot(invK, f);
      float norm_delta_u = norm(delta_u);
      if (norm_delta_u > step_limit) {
        delta_u *= step_limit / norm_delta_u;
      }
      u.v() -= delta_u;

      vector r = residual(u).v();
      r[bc_dofs] = 0.0;

      double norm_r = norm(r);
      std::cout << ", |r|: " << norm_r << std::endl;
      if (norm_r < 1.0e-10) {
        break;
      }
    }

    vector r = residual(u).v();
    force_displacement_curve.push_back({umax * t, total(r[nonhomogeneous_bc_dofs])});

    K0 = stiffness_matrix(u);

    Kessential = K0({}, bc_dofs);

    Kmodified = K0;
    Kmodified({}, bc_dofs) = [](int i, int j){ return (i == j); };
    Kmodified(bc_dofs, {}) = [](int i, int j){ return (i == j); };

    invK.update(Kmodified);

    if (k == 7) break;

  }

  for (auto [u, f] : force_displacement_curve) {
    std::cout << u << ", " << f << std::endl;
  }

}