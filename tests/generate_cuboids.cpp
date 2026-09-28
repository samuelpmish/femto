#include <gtest/gtest.h>

#include "femto/mesh.hpp"
#include "misc/timer.hpp"

#include <iostream>

using namespace femto;

TEST(generate, rectangle) {
  Mesh mesh = Mesh<>::cuboid({2, 2}, vec2{1.0, 1.0});
  save(mesh, "rectangle.msh");
  save(mesh, "rectangle.vtk");
  save(mesh, "rectangle.vtu");
}

TEST(generate, cube) {
  Mesh mesh = Mesh<>::cuboid({2, 2, 2}, vec3{1.0, 1.0, 1.0});
  save(mesh, "cube.msh");
  save(mesh, "cube.vtk");
  save(mesh, "cube.vtu");
}
