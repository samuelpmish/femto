#include "femto/finite_element.hpp"

#include "misc/timer.hpp"
#include "misc/nanobench.h"

#include "fm/types/vec.hpp"

#include "common.hpp"

nd::cpu_array<double, 2> default_quadrature_rule(Geometry geom, int p) {

  uint32_t q = p + 1;

  if (geom == Geometry::Edge) {
    nd::cpu_array<double, 2> xi({q, 1});
    nd::cpu_array<double, 1> wts({q});
    gauss_legendre_segment_rule(q, xi.data(), wts.data());
    return xi;
  }

  if (geom == Geometry::Triangle) {
    nd::cpu_array<double, 2> xi({(q * (q + 1)) / 2, 2});
    nd::cpu_array<double, 1> wts({(q * (q + 1)) / 2});
    gauss_legendre_triangle_rule(q, xi.data(), wts.data());
    return xi;
  }

  if (geom == Geometry::Quadrilateral) {
    nd::cpu_array<double, 1> xi_1D({q});
    nd::cpu_array<double, 1> wts_1D({q});
    gauss_legendre_segment_rule(q, xi_1D.data(), wts_1D.data());

    nd::cpu_array<double, 2> xi({q * q, 2});
    uint32_t count = 0;
    for (int j = 0; j < q; j++) {
      for (int i = 0; i < q; i++) {
        xi(count, 0) = xi_1D(i);
        xi(count, 1) = xi_1D(j);
        count++;
      }
    }
    return xi;
  }

  if (geom == Geometry::Tetrahedron) {
    nd::cpu_array<double, 2> xi({(q * (q + 1) * (q + 2)) / 6, 3});
    nd::cpu_array<double, 1> wts({(q * (q + 1) * (q + 2)) / 6});
    gauss_legendre_triangle_rule(q, xi.data(), wts.data());
    return xi;
  }

  if (geom == Geometry::Hexahedron) {
    nd::cpu_array<double, 1> xi_1D({q});
    nd::cpu_array<double, 1> wts_1D({q});
    gauss_legendre_segment_rule(q, xi_1D.data(), wts_1D.data());

    nd::cpu_array<double, 2> xi({q * q * q, 3});
    uint32_t count = 0;
    for (int k = 0; k < q; k++) {
      for (int j = 0; j < q; j++) {
        for (int i = 0; i < q; i++) {
          xi(count, 0) = xi_1D(i);
          xi(count, 1) = xi_1D(j);
          xi(count, 2) = xi_1D(k);
          count++;
        }
      }
    }
    return xi;
  }

  return {};

}

template < Geometry g, int q, int i >
constexpr auto constexpr_qpt() {
  vec< dimension(g) > v{};
  v[0] = q;
  v[1] = i;
  return v;
}

template < Geometry geom, int p, int q1D >
void interpolation_stress_test_constexpr(nd::view< double, 2 > input, 
                               nd::view< double, 2 > output, 
                               const uint32_t num_repetitions) {

  FiniteElement<geom, Family::H1> element{p};

  constexpr int q = (dimension(geom) == 2) ? q1D * q1D : q1D * q1D * q1D;

  std::vector< double > u_e(element.num_nodes());
  std::vector< double > u_q(q, 0.0);

  int num_nodes = element.num_nodes();
  int num_elements = input.shape[0];
  for (int i = 0; i < num_elements; i++) {
    for (int j = 0; j < u_e.size(); j++) {
      u_e[j] = input(i, j);
    }

    for (int j = 0; j < num_repetitions; j++) {
      for_constexpr<q>([&](auto k){
        double sum = 0.0;
        for_constexpr<q>([&](auto l){
          constexpr auto xi = constexpr_qpt<geom, q, l>();
          constexpr uint32_t ix = l % (p + 1);
          constexpr uint32_t iy = (l % ((p + 1) * (p + 1))) / (p + 1) ;
          constexpr uint32_t iz = l / ((p + 1) * (p + 1));
          constexpr auto phix = GaussLobattoInterpolation(xi[0], p + 1, ix); 
          constexpr auto phiy = GaussLobattoInterpolation(xi[1], p + 1, iy); 
          constexpr auto phiz = GaussLobattoInterpolation(xi[2], p + 1, iz); 
          constexpr auto phi = phix * phiy * phiz;
          sum += phi * u_e[l];
        });
        u_q[k] += sum;
      });
    }

    for (int j = 0; j < u_q.size(); j++) {
      output(i, j) = u_q[j];
    }
  }

}

template < Geometry geom >
void interpolation_stress_test_sf(nd::view< double, 2 > input, 
                                  nd::view< double, 2 > output, 
                                  nd::view< double, 2 > xi,
                                  const uint32_t p,
                                  const uint32_t num_repetitions) {

  FiniteElement<geom, Family::H1> element{p};

  int q1D = xi.shape[0];
  int qpts_per_element = pow(q1D, dimension(geom));

  std::vector< double > u_e(element.num_nodes());
  std::vector< double > u_q(qpts_per_element, 0.0);
  
  nd::view< const double, 2 > u_e_view{&u_e[0], {element.num_nodes(), 1}};
  nd::view< double, 2 > u_q_view{&u_q[0], {xi.shape[0], 1}};

  auto precomputed_shape_fns = element.evaluate_shape_functions(xi);

  int num_elements = input.shape[0];
  for (int i = 0; i < num_elements; i++) {
    for (int j = 0; j < u_e.size(); j++) {
      u_e[j] = input(i, j);
    }

    for (int j = 0; j < num_repetitions; j++) {
      element.interpolate(u_q_view, u_e_view, precomputed_shape_fns);
    }

    for (int j = 0; j < u_q.size(); j++) {
      output(i, j) = u_q[j];
    }
  }

}

template < Geometry geom >
void interpolation_stress_test_precalculated_shape_fn(nd::view< double, 2 > input, 
                                                      nd::view< double, 2 > output, 
                                                      nd::view< double, 2 > xi,
                                                      const uint32_t p,
                                                      const uint32_t num_repetitions) {

  using vecd = vec< dimension(geom) >;

  FiniteElement<geom, Family::H1> element{p};

  auto nodes_per_element = element.num_nodes();
  auto qpts_per_element = xi.shape[0];

  std::vector< double > shape_fns(qpts_per_element * nodes_per_element);
  auto phi = [&](int q, int i) -> double & { 
    return shape_fns[q * nodes_per_element + i]; 
  };

  for (int q = 0; q < qpts_per_element; q++) {
    for (int i = 0; i < nodes_per_element; i++) {
      vecd xi_q; 
      for (int d = 0; d < dimension(geom); d++) {
        xi_q[d] = xi(i, d);
      }
      phi(q,i) = element.shape_function(xi_q, i);
    }
  }

  std::vector< double > u_e(nodes_per_element);
  std::vector< double > u_q(qpts_per_element, 0.0);

  int num_elements = input.shape[0];
  for (int i = 0; i < num_elements; i++) {
    for (int j = 0; j < u_e.size(); j++) {
      u_e[j] = input(i, j);
    }

    for (int j = 0; j < num_repetitions; j++) {
      for (int k = 0; k < qpts_per_element; k++) {
        double sum{};
        for (int l = 0; l < nodes_per_element; l++) {
          sum += phi(k, l) * u_e[l];
        }
        u_q[k] += sum;
      }
    }

    for (int j = 0; j < u_q.size(); j++) {
      output(i, j) = u_q[j];
    }
  }

}

template < Geometry geom >
void interpolation_stress_test(nd::view< double, 2 > input, 
                               nd::view< double, 2 > output, 
                               nd::view< double, 2 > xi,
                               const uint32_t p,
                               const uint32_t num_repetitions) {

  using vecd = vec< dimension(geom) >;

  FiniteElement<geom, Family::H1> element{p};

  std::vector< double > u_e(element.num_nodes());
  std::vector< double > u_q(xi.shape[0], 0.0);

  int num_elements = input.shape[0];
  for (int i = 0; i < num_elements; i++) {
    for (int j = 0; j < u_e.size(); j++) {
      u_e[j] = input(i, j);
    }

    for (int j = 0; j < num_repetitions; j++) {
      for (int k = 0; k < u_q.size(); k++) {
        vecd xi_q; 
        for (int d = 0; d < dimension(geom); d++) {
          xi_q[d] = xi(k, d);
        }

        u_q[k] += element.interpolate(xi_q, &u_e[0]);
      }
    }

    for (int j = 0; j < u_q.size(); j++) {
      output(i, j) = u_q[j];
    }
  }

}

template < Geometry geom >
void comparison(const uint32_t p, const uint32_t num_repetitions) {

  FiniteElement< geom, Family::H1 > element{p};
  uint32_t num_elements = 100000;

  nd::cpu_array<double, 2> u_e = random(stack::array<uint32_t,2>{num_elements, element.num_nodes()});
  nd::cpu_array<double, 2> xi = default_quadrature_rule(geom, p);
  uint32_t qpts_per_element = xi.shape[0];

  nd::cpu_array<double, 2> u_q({num_elements, qpts_per_element});

  femto::timer stopwatch;

  stopwatch.start();
  interpolation_stress_test< geom >(u_e, u_q, xi, p, num_repetitions);
  ankerl::nanobench::doNotOptimizeAway(u_q);
  stopwatch.stop();
  std::cout << stopwatch.elapsed() * 1000 << " ";

  stopwatch.start();
  interpolation_stress_test_precalculated_shape_fn< geom >(u_e, u_q, xi, p, num_repetitions);
  ankerl::nanobench::doNotOptimizeAway(u_q);
  stopwatch.stop();
  std::cout << stopwatch.elapsed() * 1000 << " ";

  if constexpr (geom == Geometry::Hexahedron) {
    stopwatch.start();
    switch (p) {
      case 1: 
        interpolation_stress_test_constexpr< geom, 1, 2 >(u_e, u_q, num_repetitions);
        break;
      case 2: 
        interpolation_stress_test_constexpr< geom, 2, 3 >(u_e, u_q, num_repetitions);
        break;
      case 3: 
        interpolation_stress_test_constexpr< geom, 3, 4 >(u_e, u_q, num_repetitions);
        break;
    }
    ankerl::nanobench::doNotOptimizeAway(u_q);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000 << " ";
  }

  if (geom == Geometry::Hexahedron || geom == Geometry::Quadrilateral) {
    nd::cpu_array<double, 2> xi_1D = default_quadrature_rule(Geometry::Edge, p);
    stopwatch.start();
    interpolation_stress_test_sf< geom >(u_e, u_q, xi_1D, p, num_repetitions);
    ankerl::nanobench::doNotOptimizeAway(u_q);
    stopwatch.stop();
    std::cout << stopwatch.elapsed() * 1000 << " ";
  }

  std::cout << std::endl;

}

int main() {

  //comparison< Geometry::Quadrilateral >(1, 10);
  //comparison< Geometry::Quadrilateral >(2, 10);
  //comparison< Geometry::Quadrilateral >(3, 10);

  comparison< Geometry::Tetrahedron >(1, 3);
  comparison< Geometry::Tetrahedron >(2, 3);
  comparison< Geometry::Tetrahedron >(3, 3);

  //comparison< Geometry::Hexahedron >(1, 10);
  //comparison< Geometry::Hexahedron >(2, 10);
  //comparison< Geometry::Hexahedron >(3, 10);
}