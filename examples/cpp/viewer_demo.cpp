// minimal example of embedding the femto::server viewer in an application:
// pushes a few timesteps of a scalar field and a displacement field on a
// degree-2 3D hex mesh, then keeps serving them until ctrl+C
//
// usage: viewer_demo [port]   (default: 8080)
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "server.hpp"
#include "femto/mesh.hpp"

using namespace femto;

int main(int argc, char* argv[]) {

  int port = (argc > 1) ? std::atoi(argv[1]) : 8080;
  if (port <= 0 || port > 65535) {
    std::cout << "usage: " << argv[0] << " [port]" << std::endl;
    return 1;
  }

  server srv({.port = port});

  int p = 2;

  Mesh<> mesh = Mesh<>::cuboid({8, 8, 8}, fm::vec3{1.0, 1.0, 1.0});
  Field X_p = create_field<Family::H1>(mesh, p, 3);
  X_p.data = nodes_for(X_p, mesh);
  mesh.X = X_p;

  // for 3D meshes, only the boundary surface of the mesh (and the
  // degrees of freedom on it) is serialized
  srv.set_mesh(mesh);

  uint32_t dim = mesh.spatial_dimension;

  Field T = create_field<Family::H1>(mesh, p, 1);
  Field T1 = create_field<Family::H1>(mesh, 1, 1);
  Field u = create_field<Family::H1>(mesh, p, dim);
  auto nodes = nodes_for(T, mesh);
  auto verts = nodes_for(T1, mesh);

  for (int step = 0; step < 20; step++) {
    double t = 0.2 * step;
    for (uint32_t i = 0; i < T.data.shape[0]; i++) {
      double x = nodes(i, 0), y = nodes(i, 1);
      T.data(i, 0) = std::sin(4 * x + t) * std::cos(4 * y);
      for (uint32_t c = 0; c < dim; c++) {
        u.data(i, c) = 0.03 * std::sin(3 * nodes(i, c) + t + c);
      }
    }
    for (uint32_t i = 0; i < T1.data.shape[0]; i++) {
      T1.data(i, 0) = std::sin(4 * verts(i, 0) + t) * std::cos(4 * verts(i, 1));
    }
    srv.push_field("temperature", T);
    srv.push_field("temperature_p1", T1);
    srv.push_field("displacement", u);
  }

  srv.wait();

  return 0;
}
