#include <gtest/gtest.h>

#include <random>
#include <iostream>
#include <functional>

#include "common.hpp"
#include "forall.hpp"
#include "femto/finite_element.hpp"

using namespace femto;

#define close_enough(a, b, eps) static_assert(((b - a) * (b - a)) < (eps * eps))

template < int dim >
std::vector< vec<dim, double> > random_points(int n, uint32_t seed = 0) {
  std::default_random_engine generator;
  std::uniform_real_distribution< double > dist(0.0, 1.0);
  
  std::vector< vec<dim> > points(n);
  for (int i = 0; i < n; i++) {
    for (int d = 0; d < dim; d++) {
      points[i][d] = dist(generator);
    }
  }
  return points;
}

template < typename T >
std::ostream& operator<<(std::ostream & out, const std::vector<T> & v) {
  for (int i = 0; i < v.size(); i++) {
    out << i << ": " << v[i] << std::endl;
  }
  return out;
}

constexpr auto f(vec2 xi) {
  return vec2{1+2*xi[1], 3-2*xi[0]};
}

constexpr auto f(vec3 xi) {
  return vec3{1+xi[2]-2*xi[1], 2+2*xi[0]+4*xi[2], 3-xi[0]-4*xi[1]};
}

constexpr auto curl_f(vec2 xi) { return -4.0; };
constexpr auto curl_f(vec3 xi) { return vec3{-8, 2, 4}; };

template < Geometry geom >
void hcurl_element_test(uint32_t p, double tolerance) {

  constexpr uint32_t dim = dimension(geom);

  using vecd = vec<dim, double>;

  FiniteElement<geom, Family::Hcurl> element{p};

  nd::cpu_array<double, 2> nodes({element.num_nodes(), dim});
  nd::cpu_array<double, 2> directions({element.num_nodes(), dim});

  element.nodes(nodes);
  element.directions(directions);

  nd::cpu_array<double, 2> values = forall(+[](const vecd & x, const vecd & direction) {
    return dot(f(x), direction);
  }, nodes, directions);

  // kronecker delta property 
  for (int i = 0; i < nodes.shape[0]; i++) {
    vecd x{};
    vecd dir{};
    for (int j = 0; j < dim; j++) {
      x[j] = nodes(i,j);
      dir[j] = directions(i,j);
    }


    vecd value = element.interpolate(x, values.data());
    EXPECT_NEAR(dot(value, dir), values(i,0), tolerance);
  }

  std::vector< vecd > random_xi = random_points<dim>(6);

  // linear reproducibility
  for (vecd xi : random_xi) {
    vecd value = element.interpolate(xi, values.data());
    auto curl = element.curl(xi, values.data());

    for (int d = 0; d < dim; d++) {
      EXPECT_NEAR(value[d], f(xi)[d], tolerance);
    }

    if constexpr (dim == 2) {
      EXPECT_NEAR(curl, curl_f(xi), tolerance);
    }
    if constexpr (dim == 3) {
      for (int d = 0; d < 3; d++) {
        EXPECT_NEAR(curl[d], curl_f(xi)[d], tolerance);
      }
    }
  }

}

#if 0
template < Geometry geom, int p >
constexpr bool constexpr_tests() {

  constexpr int dim = dimension(geom);

  using vecd = vec<dim, double>;

  constexpr FiniteElement<geom, Family::Hcurl> element{p};

  constexpr int num_nodes = element.num_nodes();

  double nodes[dim * num_nodes];
  double directions[dim * num_nodes];
  double values[num_nodes];

  element.nodes(nodes);
  element.directions(directions);

  for (int i = 0; i < num_nodes; i++) {
    vecd x;
    vecd dir;
    for (int d = 0; d < dim; d++) {
      x[d] = nodes[i * dim + d];
      dir[d] = directions[i * dim + d];
    }
    values[i] = dot(dir, f(x));
  }

  bool pass = true;

  for (int i = 0; i < num_nodes; i++) {
    vecd value = element.interpolate(nodes + i*dim, values);
    vecd dir;
    for (int d = 0; d < dim; d++) {
      dir[d] = directions[i * dim + d];
    }
    close_enough(dot(value, dir), values[i], 3.0e-17);
  }

  // linear reproducibility
  for (int i = 0; i < 10; i++) {
    vecd xi;
    for (int d = 0; d < dim; d++) {
      xi[d] = constexpr_random_values[i * dim + d];
    }

    vecd value = element.interpolate(xi.data, values);
    auto curl = element.curl(xi.data, values);

    for (int d = 0; d < dim; d++) {
      close_enough(value[d], f(xi)[d], 5.0e-14);
    }

    if constexpr (dim == 2) {
      close_enough(curl, curl_f(xi), 3.0e-14);
    }
    if constexpr (dim == 3) {
      for (int d = 0; d < 3; d++) {
        close_enough(curl[d], curl_f(xi)[d], 8.0e-14);
      }
    }
  }

  return true;
}

TEST(shape_functions_hcurl, constexpr_tests) {
  constexpr bool z1 = constexpr_tests<Geometry::Tetrahedron, 1>();
  constexpr bool z2 = constexpr_tests<Geometry::Tetrahedron, 2>();
  constexpr bool z3 = constexpr_tests<Geometry::Tetrahedron, 3>();
  //static_assert(constexpr_tests<Geometry::Tetrahedron, 2>());
  //static_assert(constexpr_tests<Geometry::Tetrahedron, 3>());
}
#endif

TEST(shape_functions_hcurl, triangle_p1) { hcurl_element_test<Geometry::Triangle>(1, 1.0e-15); }
TEST(shape_functions_hcurl, triangle_p2) { hcurl_element_test<Geometry::Triangle>(2, 1.0e-14); }
TEST(shape_functions_hcurl, triangle_p3) { hcurl_element_test<Geometry::Triangle>(3, 1.0e-13); }

TEST(shape_functions_hcurl, quadrilateral_p1) { hcurl_element_test<Geometry::Quadrilateral>(1, 1.0e-15); }
TEST(shape_functions_hcurl, quadrilateral_p2) { hcurl_element_test<Geometry::Quadrilateral>(2, 1.0e-14); }
TEST(shape_functions_hcurl, quadrilateral_p3) { hcurl_element_test<Geometry::Quadrilateral>(3, 1.0e-13); }

TEST(shape_functions_hcurl, tetrahedron_p1) { hcurl_element_test<Geometry::Tetrahedron>(1, 1.0e-15); }
TEST(shape_functions_hcurl, tetrahedron_p2) { hcurl_element_test<Geometry::Tetrahedron>(2, 3.0e-14); }
TEST(shape_functions_hcurl, tetrahedron_p3) { hcurl_element_test<Geometry::Tetrahedron>(3, 1.0e-12); }

TEST(shape_functions_hcurl, hexahedron_p1) { hcurl_element_test<Geometry::Hexahedron>(1, 3.0e-15); }
TEST(shape_functions_hcurl, hexahedron_p2) { hcurl_element_test<Geometry::Hexahedron>(2, 1.0e-14); }
TEST(shape_functions_hcurl, hexahedron_p3) { hcurl_element_test<Geometry::Hexahedron>(3, 1.0e-12); }