#include <gtest/gtest.h>

#include <random>
#include <iostream>
#include <functional>

#include "common.hpp"
#include "forall.hpp"
#include "femto/finite_element.hpp"

using namespace femto;

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

template < Geometry geom >
void h1_element_test(uint32_t p) {

  constexpr int dim = dimension(geom);

  using vecd = vec<dim, double>;

  auto f = [](const vecd & xi) { 
    double value = 1.0;
    for (int i = 0; i < dim; i++) {
      value += (i + 1) * xi[i];  
    }
    return value;
  };

  auto grad_f = [](const vecd & xi) { 
    vecd output{};
    for (int i = 0; i < dim; i++) {
      output[i] = (i + 1);
    }
    return output;
  };

  FiniteElement<geom, Family::H1> element{p};

  nd::cpu_array<double, 2> nodes({element.num_nodes(), dim});

  element.nodes(nodes);

  nd::cpu_array<double, 2> values = forall(+f, nodes);

  // kronecker delta property 
  for (int i = 0; i < nodes.shape[0]; i++) {
    vecd x{};
    for (int j = 0; j < dim; j++) {
      x[j] = nodes(i,j);
    }

    double value = element.interpolate(x, values.data());
    EXPECT_NEAR(value, values[i], 3.0e-15);
  }

  std::vector< vecd > random_xi = random_points<dim>(6);

  // linear reproducibility
  for (vecd xi : random_xi) {
    double value = element.interpolate(xi, values.data());
    vecd grad = element.gradient(xi, values.data());

    EXPECT_NEAR(value, f(xi), 3.0e-14);
    for (int d = 0; d < dim; d++) {
      EXPECT_NEAR(grad[d], grad_f(xi)[d], 4.0e-14);
    }
  }

  // partition of unity
  for (vecd xi : random_xi) {
    double total_value{};
    vecd total_gradient{};
    for (int i = 0; i < nodes.shape[0]; i++) {
      total_value += element.shape_function(xi, i);
      total_gradient += element.shape_function_gradient(xi, i);
    }

    EXPECT_NEAR(total_value, 1.0, 2.0e-14);
    for (int d = 0; d < dim; d++) {
      EXPECT_NEAR(total_gradient[d], 0.0, 2.0e-14);
    }
  }

}

TEST(h1_shape_functions, triangle_p1) { h1_element_test<Geometry::Triangle>(1); }
TEST(h1_shape_functions, triangle_p2) { h1_element_test<Geometry::Triangle>(2); }
TEST(h1_shape_functions, triangle_p3) { h1_element_test<Geometry::Triangle>(3); }

TEST(h1_shape_functions, quadrilateral_p1) { h1_element_test<Geometry::Quadrilateral>(1); }
TEST(h1_shape_functions, quadrilateral_p2) { h1_element_test<Geometry::Quadrilateral>(2); }
TEST(h1_shape_functions, quadrilateral_p3) { h1_element_test<Geometry::Quadrilateral>(3); }

TEST(h1_shape_functions, tetrahedron_p1) { h1_element_test<Geometry::Tetrahedron>(1); }
TEST(h1_shape_functions, tetrahedron_p2) { h1_element_test<Geometry::Tetrahedron>(2); }
TEST(h1_shape_functions, tetrahedron_p3) { h1_element_test<Geometry::Tetrahedron>(3); }

TEST(h1_shape_functions, hexahedron_p1) { h1_element_test<Geometry::Hexahedron>(1); }
TEST(h1_shape_functions, hexahedron_p2) { h1_element_test<Geometry::Hexahedron>(2); }
TEST(h1_shape_functions, hexahedron_p3) { h1_element_test<Geometry::Hexahedron>(3); }