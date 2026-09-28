#include "femto/mesh.hpp"
#include "femto/field.hpp"
#include "femto/domain.hpp"

#include "forall.hpp"

#include "misc/parse.hpp"
#include "misc/timer.hpp"

#include <functional>

namespace compiler {
static void please_do_not_optimize_away([[maybe_unused]] void* p) { asm volatile("" : : "g"(p) : "memory"); }
}

using namespace femto;

timer stopwatch;

template < uint32_t dim >
vec<dim> qfunction(const vec<dim> & du_dxi, const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  vec<dim> du_dX = dot(du_dxi, dxi_dX);
  vec<dim> heat_flux = 3.0 * du_dX;
  for (int i = 0; i < 100; i++) {
    heat_flux = heat_flux * 3.0;
  }
  return dot(heat_flux, transpose(dxi_dX)) * det(dX_dxi);
}

template < uint32_t dim >
mat<dim,dim> qfunction_derivative(const mat<dim,dim> & dX_dxi) {
  mat<dim,dim> dxi_dX = inv(dX_dxi);
  return dot(dxi_dX, transpose(dxi_dX)) * (3.0 * det(dX_dxi));
}

namespace impl {

template < Geometry geom >
void fused_kernel(Residual<Family::H1> & r,
                  const Field<Family::H1> & values,
                  nd::view<const double, 3 > dX_dxi,
                  nd::view<const Connection, 2> connectivity,
                  const nd::view<const int> elements,
                  const nd::view<const double, 2> xi) {

  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  FiniteElement< geom, Family::H1 > el{values.degree};

  using flux_type = typename FiniteElement< geom, Family::H1 >::flux_type;

  // allocate storage for an element's nodal values
  constexpr uint32_t gdim = dimension(geom);
  uint32_t components = values.data.shape[1];
  uint32_t nodes_per_element = el.num_nodes();
  uint32_t qpts_per_element = qpe<geom>(xi.shape[0]);
  nd::cpu_array<uint32_t, 1> ids({nodes_per_element});
  nd::cpu_array<double, 1> u_e({nodes_per_element});
  nd::cpu_array<double, 1> r_e({nodes_per_element});

  nd::cpu_array<double, 3> du_dxi_q({qpts_per_element, 1, gdim});
  nd::cpu_array<flux_type, 2> flux_xi_q({qpts_per_element, 1});

  using vec_t = vec<gdim>;
  using mat_t = mat<gdim, gdim>;

  const mat_t * dX_dxi_ptr = reinterpret_cast<const mat_t *>(dX_dxi.data());

  // precalculate shape functions for the provided quadrature rule
  auto shape_fn_grads = el.evaluate_shape_function_gradients(xi);

  // for each element with this geometry
  for (uint32_t i = 0; i < num_elements; i++) {

    // figure out which nodal values belong to this element
    el.indices(values.offsets, connectivity(elements(i)).data(), ids.data());

    for (int c = 0; c < components; c++) {
      // load the nodal values for this element
      for (int j = 0; j < nodes_per_element; j++) {
        u_e(j) = values.data(ids(j), c);
      }

      // interpolate the quadrature point values for this element
      nd::range qpts{0u, qpts_per_element};
      el.gradient(du_dxi_q(qpts, c), u_e, shape_fn_grads);
    }

    const vec_t * du_dxi_ptr = reinterpret_cast< const vec_t * >(du_dxi_q.data());
    vec_t * flux_xi_ptr = reinterpret_cast< vec_t * >(flux_xi_q.data());
    for (int q = 0; q < qpts_per_element; q++) {
      flux_xi_ptr[q] = qfunction<gdim>(du_dxi_ptr[q], dX_dxi_ptr[q]);
    }
    dX_dxi_ptr += qpts_per_element;

    for (int c = 0; c < components; c++) {
      nd::range qpts{0u, qpts_per_element};
      el.integrate_flux(r_e, flux_xi_q(qpts, c), shape_fn_grads);

      // scatter-add the nodal forces for this element
      for (int j = 0; j < nodes_per_element; j++) {
        r.data(ids(j), c) += r_e(j);
      }
    }

  }

}

#if 0
template < Geometry geom, int p >
void fused_kernel_constexpr(Residual<Family::H1> & r,
                  const Field<Family::H1> & values,
                  nd::view<const double, 3 > dX_dxi,
                  nd::view<const Connection, 2> connectivity,
                  const nd::view<const int> elements,
                  const nd::view<const double, 2> xi) {

  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  constexpr FiniteElement< geom, Family::H1 > el{p};

  using flux_type = typename FiniteElement< geom, Family::H1 >::flux_type;

  // allocate storage for an element's nodal values
  constexpr uint32_t gdim = dimension(geom);
  uint32_t components = values.data.shape[1];
  uint32_t nodes_per_element = el.num_nodes();
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);
  nd::cpu_array<uint32_t, 1> ids({nodes_per_element});
  nd::cpu_array<double, 1> u_e({nodes_per_element});
  nd::cpu_array<double, 1> r_e({nodes_per_element});

  nd::cpu_array<double, 3> du_dxi_q({qpts_per_element, 1, gdim});
  nd::cpu_array<flux_type, 2> flux_xi_q({qpts_per_element, 1});

  using vec_t = vec<gdim>;
  using mat_t = mat<gdim, gdim>;

  const mat_t * dX_dxi_ptr = reinterpret_cast<const mat_t *>(dX_dxi.data());

  // precalculate shape functions for the provided quadrature rule
  auto shape_fn_grads = el.evaluate_shape_function_gradients(xi);

  // for each element with this geometry
  for (uint32_t i = 0; i < num_elements; i++) {

    // figure out which nodal values belong to this element
    el.indices(values.offsets, connectivity(elements(i)).data(), ids.data());

    for (int c = 0; c < components; c++) {
      // load the nodal values for this element
      for (int j = 0; j < nodes_per_element; j++) {
        u_e(j) = values.data(ids(j), c);
      }

      // interpolate the quadrature point values for this element
      nd::range qpts{0u, qpts_per_element};
      el.gradient(du_dxi_q(qpts, c), u_e, shape_fn_grads);
    }

    const vec_t * du_dxi_ptr = reinterpret_cast< const vec_t * >(du_dxi_q.data());
    vec_t * flux_xi_ptr = reinterpret_cast< vec_t * >(flux_xi_q.data());
    for (int q = 0; q < qpts_per_element; q++) {
      flux_xi_ptr[q] = qfunction<gdim>(du_dxi_ptr[q], dX_dxi_ptr[q]);
    }
    dX_dxi_ptr += qpts_per_element;

    for (int c = 0; c < components; c++) {
      nd::range qpts{0u, qpts_per_element};
      el.integrate_flux(r_e, flux_xi_q(qpts, c), shape_fn_grads);

      // scatter-add the nodal forces for this element
      for (int j = 0; j < nodes_per_element; j++) {
        r.data(ids(j), c) += r_e(j);
      }
    }

  }

}
#endif

}

Residual<Family::H1> fused_kernel(const Field<Family::H1> & u,
                      nd::view<const double, 3 > dX_dxi,
                      const Domain<> & domain,
                      bool use_templated_kernel) {

  auto phi = BasisFunction{u};

  Residual<Family::H1> output(phi.space, domain.mesh);

  uint32_t gdim = domain.mesh.geometry_dimension;
  uint32_t num_components = phi.space.components;

  auto qranges = ranges(domain.num_qpts);

#if 0
  femto::timer stopwatch;
  stopwatch.start();
  foreach_geometry([&](auto geom){
    if constexpr (geom != Geometry::Vertex) {
      nd::view<const int> elements = domain.active_elements[geom];
      if (gdim == dimension(geom) && elements.size() > 0) {
        nd::view<const double, 2> xi = domain.rule[geom].points;
        nd::view<const Connection, 2> connectivity = domain.mesh[geom];

        nd::range all{0u, num_components};
        if (use_templated_kernel) {
          if (u.degree == 1) {
            impl::fused_kernel_constexpr<geom, 1>(output, u, dX_dxi, connectivity, elements, xi);
          }

          if (u.degree == 2) {
            impl::fused_kernel_constexpr<geom, 2>(output, u, dX_dxi, connectivity, elements, xi);
          }

          if (u.degree == 3) {
            impl::fused_kernel_constexpr<geom, 3>(output, u, dX_dxi, connectivity, elements, xi);
          }
        } else {
          impl::fused_kernel<geom>(output, u, dX_dxi, connectivity, elements, xi);
        }
      }
    }
  });
  stopwatch.stop();

  if (print_timings) {
    std::cout << "fused implementation calculation time: " << stopwatch.elapsed() * 1000.0 << "ms" << std::endl;
  }
#endif

  return output;

}

template < int dim >
void run_test(const Mesh<> & mesh, int p, int q) {

  femto::timer stopwatch;

  Field u = create_field<Family::H1>(mesh, p);

  BasisFunction phi(u);

  auto X = nodes_for(u, mesh);
  u = forall(std::function< double(vec<dim>)> ([](vec<dim> X){ return X[0]; }), X);

  Domain domain(mesh, MeshQuadratureRule(p + 1));

  nd::cpu_array<double, 3> dX_dxi_q = evaluate(grad(mesh.X), isoparametric(domain));

  for (int i = 0; i < 3; i++) {

    std::cout << "interpolate gradient ";
    nd::cpu_array<double, 3> du_dxi_q = evaluate(grad(u), isoparametric(domain));
    std::cout << std::flush;

    std::cout << "qfunction ";
    nd::cpu_array<double, 2> f_q = forall(qfunction<dim>, du_dxi_q, dX_dxi_q);
    std::cout << std::flush;

    std::cout << "interpolate flux ";
    Residual<Family::H1> r = integrate(dot(f_q, grad(phi)), isoparametric(domain));
    std::cout << std::flush;

    //std::cout << "qfunction derivative ";
    //nd::cpu_array<double, 3> k_q = forall(qfunction_derivative<dim>, dX_dxi_q);
    //std::cout << std::flush;

    //std::cout << "stiffness ";
    //sparse_matrix K = integrate(dot(grad_wrt_xi(phi), k_q, grad_wrt_xi(phi)), domain);
    //std::cout << std::flush;

    std::cout << "poisson kernel (runtime degree) ";
    Residual<Family::H1> r2 = fused_kernel(u, dX_dxi_q, domain, false);
    std::cout << std::flush;

    std::cout << "poisson kernel (constexpr degree) ";
    Residual<Family::H1> r3 = fused_kernel(u, dX_dxi_q, domain, false);
    std::cout << std::flush;

    nd::cpu_array<double, 2> tmp1 = f_q;
    nd::cpu_array<double, 2> tmp2 = f_q;
    stopwatch.start();
    tmp1 = f_q;
    tmp2 = f_q;
    stopwatch.stop();
    std::cout << "additional memory read/write time " << stopwatch.elapsed() * 1000.0 << "ms " << std::endl;

    std::cout << std::endl;
  }

}

int main(int argc, char **argv) {

  #if 1
  std::string meshfile;
  #else
  std::string meshfile = FEMTO_MESH_DIR"icosahedron.msh";
  #endif

  int order = 1;

  int q = 2;

  parse(argc, argv,
    std::make_tuple(std::string("-m"), &meshfile),
    std::make_tuple(std::string("-o"), &order),
    std::make_tuple(std::string("-q"), &q)
  );

  if (meshfile.empty()) {
    std::cout << "error: expected mesh file name after -m" << std::endl;
    return 1;
  }

  print_timings = true;

  std::cout << "loading mesh ... ";
  Mesh<> mesh = Mesh<>::load(meshfile);
  std::cout << "finished" << std::endl;

  std::cout << "mesh has:" << std::endl;
  std::cout << "  " << mesh.vert.shape[0] << " vertices" << std::endl;
  std::cout << "  " << mesh.edge.shape[0] << " edges" << std::endl;
  std::cout << "  " << mesh.tri.shape[0] << " triangles" << std::endl;
  std::cout << "  " << mesh.quad.shape[0] << " quadrilaterals" << std::endl;
  std::cout << "  " << mesh.tet.shape[0] << " tetrahedra" << std::endl;
  std::cout << "  " << mesh.hex.shape[0] << " hexahedra" << std::endl;

  if (mesh.spatial_dimension == 2) { run_test<2>(mesh, order, q); }
  if (mesh.spatial_dimension == 3) { run_test<3>(mesh, order, q); }

}
