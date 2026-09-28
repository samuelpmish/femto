#pragma once

#include <memory>
#include <string>

#include "femto/field.hpp"

namespace femto {

// embeds a small webserver in an application, so that meshes and solution
// fields can be viewed from a browser (see data/viewer.html for the client):
//
//   femto::server server({.port = 8080});
//   server.set_mesh(mesh);
//   while (simulating) {
//     ...
//     server.push_field("displacement", u);
//   }
//   server.wait(); // keep the application alive until ctrl+C, if desired
//
// the client keeps the mesh for the duration of the session, receives the
// sequence of fields pushed by the application, and lets the user step
// through them (arrow keys), pan/zoom/orbit the viewport, and visualize
//   - scalar fields, by mapping values to colors interpolated over the mesh
//   - displacement fields (components == spatial dimension), by drawing the
//     deformed configuration with an adjustable exaggeration factor
//
// high-order meshes and fields (degree <= 3) are transmitted as raw node
// values together with their element connectivity, and the client evaluates
// the femto shape functions to tessellate each element at an adjustable
// resolution -- curved elements and sub-element solution variation are
// rendered faithfully, not with linear interpolation between vertices.
//
// for 3D meshes, only the boundary surface (and the degrees of freedom on
// it) is serialized, to keep the data footprint small.
//
// notes:
//  - H1 fields are supported in 2D and 3D; DG fields are supported in 2D
//    (in 3D their nodes are interior to the elements, so they have no trace
//    on the boundary surface that could be drawn)
//  - the http types are kept out of this header (see src/server.cpp)
struct server {

  struct settings {
    int port = 8080;

    // the client webpage served at "/". If empty, defaults to the
    // viewer.html shipped in this project's data directory
    std::string html_path = "";
  };

  server();
  server(settings s);
  ~server();

  server(const server &) = delete;
  server & operator=(const server &) = delete;

  // replace the mesh (this also discards any previously-pushed fields,
  // since they are only meaningful for the mesh they were computed on)
  void set_mesh(const Mesh<memory::space::cpu> & mesh);

  // append one entry to the sequence of viewable fields.
  // scalar fields are color-mapped, fields with as many components
  // as the spatial dimension are treated as displacements
  template < Family family >
  void push_field(const std::string & name, const Field<family, memory::space::cpu> & f) {
    static_assert(family == Family::H1 || family == Family::DG, "femto::server only supports H1 and DG fields");
    push_field(name, family, f.degree, f.offsets, f.data);
  }

  void push_field(const std::string & name, Family family, uint32_t degree, GeometryInfo offsets,
                  const nd::array<double, 2, memory::space::cpu> & data);

  // discard the sequence of viewable fields (the mesh is kept)
  void clear_fields();

  std::string url() const;

  // block until the server is shut down (e.g. ctrl+C), so that a finished
  // simulation can keep serving its results
  void wait();

  struct impl;
  std::unique_ptr<impl> pimpl;

};

}  // namespace femto
