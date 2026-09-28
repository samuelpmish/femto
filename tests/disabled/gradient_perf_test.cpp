#include "femto/finite_element.hpp"

template < Geometry geom >
void gradient_stress_test(nd::view< double, 2 > input, 
                          nd::view< double, 3 > output, 
                          nd::view< double, 2 > xi,
                          const uint64_t p,
                          const uint64_t num_repetitions) {

  constexpr int dim = dimension(geom);

  FiniteElement<geom, Family::H1> element{p};

  std::vector< double > u_e(element.num_nodes());
  std::vector< vec<dim> > du_dxi_q(xi.shape[0], vec<dim>{});

  int num_elements = input.shape[0];
  for (int i = 0; i < num_elements; i++) {
    for (int j = 0; j < u_e.size(); j++) {
      u_e[j] = input(i, j);
    }

    for (int j = 0; j < num_repetitions; j++) {
      for (int k = 0; k < du_dxi_q.size(); k++) {
        du_dxi_q[k] += element.gradient(&xi(k, 0), &u_e[0]);
      }
    }

    for (int j = 0; j < du_dxi_q.size(); j++) {
      for (int k = 0; k < dim; k++) {
        output(i, j, k) = du_dxi_q[j][k];
      }
    }
  }

}

int main() {

  const uint64_t p = 2;
  const uint64_t num_repetitions = 20;
  constexpr Geometry geom = Geometry::Hexahedron;

  FiniteElement< Geometry::Hexahedron, Family::H1 > element{p};
  uint64_t num_elements = 50;
  uint64_t qpts_per_element = 20;

  nd::cpu_array<double, 2> u_e({num_elements, element.num_nodes()});
  nd::cpu_array<double, 3> du_dxi_q({num_elements, qpts_per_element, dimension(geom)});
  nd::cpu_array<double, 2> xi({qpts_per_element, dimension(geom)});

  gradient_stress_test< Geometry::Hexahedron >(u_e, du_dxi_q, xi, p, num_repetitions);

}