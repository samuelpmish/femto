#include "server.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "femto/assert.hpp"
#include "femto/mesh.hpp"
#include "femto/finite_element.hpp"
#include "misc/json.hpp"

#include "httplib.h"

namespace femto {

namespace {

// all binary payloads are little-endian.
//
// the mesh is transmitted as a list of high-order elements (in 2D, the
// elements themselves; in 3D, the triangular/quadrilateral faces on the
// boundary surface). fields are transmitted as raw node values together
// with the id of a "space" describing how each element indexes into those
// values. the client evaluates the actual shape functions to tessellate
// each element, so high-order geometry and solution fields are rendered
// faithfully (see data/viewer.html).
//
// GET /mesh
//   uint32   spatial dimension (2 or 3)
//   uint32   space id of the coordinate field
//   uint32   num_nodes
//   float32  coordinates[num_nodes * 3] (z == 0 for 2D meshes)
//
// GET /space/<s>  -- one per (family, degree) in use, shared by fields
//   uint32   family (0 == H1, 1 == DG)
//   uint32   degree
//   uint32   num_nodes
//   uint32   num_tris
//   uint32   num_quads
//   uint32   tri_conn[num_tris * nodes_per_tri]     (femto node ordering)
//   uint32   quad_conn[num_quads * nodes_per_quad]  (femto node ordering)
//
// GET /field/<i>
//   uint32   space id
//   uint32   components
//   uint32   num_nodes
//   float32  values[num_nodes * components]
//
// the element lists of every space refer to the same entities in the same
// order (all triangles, then all quadrilaterals), so one tessellation of
// the mesh can sample any field.

void append_u32(std::string & blob, uint32_t value) {
  blob.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

void append_f32(std::string & blob, float value) {
  blob.append(reinterpret_cast<const char *>(&value), sizeof(value));
}

uint32_t nodes_per(Geometry g, uint32_t degree) {
  if (g == Geometry::Triangle) { return ((degree + 1) * (degree + 2)) / 2; }
  else                         { return (degree + 1) * (degree + 1); }
}

// the (compile-time) element templates, dispatched at runtime. note that the
// H1 and DG variants of an element share the same reference nodes and shape
// functions -- they only differ in how nodes map to global ids
void entity_indices(Geometry g, Family family, uint32_t degree,
                    const GeometryInfo & offsets, const Connection * row, uint32_t * ids) {
  if (g == Geometry::Triangle) {
    if (family == Family::H1) { FiniteElement<Geometry::Triangle, Family::H1> el{degree}; el.indices(offsets, row, ids); }
    else                      { FiniteElement<Geometry::Triangle, Family::DG> el(degree); el.indices(offsets, row, ids); }
  } else {
    if (family == Family::H1) { FiniteElement<Geometry::Quadrilateral, Family::H1> el{degree}; el.indices(offsets, row, ids); }
    else                      { FiniteElement<Geometry::Quadrilateral, Family::DG> el(degree); el.indices(offsets, row, ids); }
  }
}

struct space_record {
  Family family;
  uint32_t degree;
  uint32_t max_node_id;             // largest global node id referenced
  std::vector<uint32_t> node_ids;   // serialized node -> row in the field data
  std::string blob;
};

struct field_record {
  std::string name;
  uint32_t space;
  uint32_t components;
  double min, max;
  std::string blob;
};

}  // namespace

struct server::impl {

  settings opts;

  httplib::Server http;
  std::thread listener;

  std::mutex mtx;
  uint64_t mesh_version = 0;
  std::string mesh_blob;
  uint32_t spatial_dimension = 0;
  uint32_t geometry_dimension = 0;

  // the connectivity rows of the drawn entities (2D: the elements
  // themselves, 3D: the faces on the mesh boundary), copied from the mesh
  uint32_t num_tris = 0, num_quads = 0;
  uint32_t tri_row_width = 0, quad_row_width = 0;
  std::vector<Connection> tri_rows;
  std::vector<Connection> quad_rows;

  std::vector<space_record> spaces;
  std::vector<field_record> fields;

  // find (or lazily build) the serialized description of a function space,
  // restricted to the drawn entities. caller must hold mtx
  uint32_t space_for(Family family, uint32_t degree, const GeometryInfo & offsets) {

    for (uint32_t s = 0; s < spaces.size(); s++) {
      if (spaces[s].family == family && spaces[s].degree == degree) { return s; }
    }

    uint32_t tri_nodes = nodes_per(Geometry::Triangle, degree);
    uint32_t quad_nodes = nodes_per(Geometry::Quadrilateral, degree);

    std::vector<uint32_t> conn(num_tris * tri_nodes + num_quads * quad_nodes);
    for (uint32_t i = 0; i < num_tris; i++) {
      entity_indices(Geometry::Triangle, family, degree, offsets,
                     &tri_rows[i * tri_row_width], &conn[i * tri_nodes]);
    }
    for (uint32_t i = 0; i < num_quads; i++) {
      entity_indices(Geometry::Quadrilateral, family, degree, offsets,
                     &quad_rows[i * quad_row_width], &conn[num_tris * tri_nodes + i * quad_nodes]);
    }

    // keep only the nodes referenced by the drawn entities, renumbered
    // (in order of first appearance) so the values can be shipped densely
    space_record space{family, degree, 0, {}, {}};
    std::unordered_map<uint32_t, uint32_t> local_id;
    local_id.reserve(conn.size());
    for (auto & c : conn) {
      auto it = local_id.find(c);
      if (it == local_id.end()) {
        uint32_t id = uint32_t(space.node_ids.size());
        local_id[c] = id;
        space.node_ids.push_back(c);
        space.max_node_id = std::max(space.max_node_id, c);
        c = id;
      } else {
        c = it->second;
      }
    }

    std::string blob;
    blob.reserve(20 + conn.size() * 4);
    append_u32(blob, (family == Family::H1) ? 0 : 1);
    append_u32(blob, degree);
    append_u32(blob, uint32_t(space.node_ids.size()));
    append_u32(blob, num_tris);
    append_u32(blob, num_quads);
    for (auto c : conn) { append_u32(blob, c); }
    space.blob = std::move(blob);

    spaces.push_back(std::move(space));
    return uint32_t(spaces.size() - 1);
  }

  std::string html_path() const {
    return opts.html_path.empty() ? (FEMTO_DATA_DIR "viewer.html") : opts.html_path;
  }

};

server::server() : server(settings{}) {}

server::server(settings s) : pimpl(std::make_unique<impl>()) {

  pimpl->opts = s;
  auto * state = pimpl.get();

  pimpl->http.Get("/", [state](const httplib::Request &, httplib::Response & res) {
    std::ifstream file(state->html_path(), std::ios::binary);
    if (!file) {
      res.status = 404;
      res.set_content("viewer page not found: " + state->html_path(), "text/plain");
      return;
    }
    std::stringstream ss;
    ss << file.rdbuf();
    res.set_content(ss.str(), "text/html");
  });

  pimpl->http.Get("/state", [state](const httplib::Request &, httplib::Response & res) {
    nlohmann::json json;
    {
      std::lock_guard<std::mutex> lock(state->mtx);
      json["mesh_version"] = state->mesh_version;
      json["elements"] = state->num_tris + state->num_quads;
      json["fields"] = nlohmann::json::array();
      for (auto & f : state->fields) {
        json["fields"].push_back({
          {"name", f.name},
          {"space", f.space},
          {"components", f.components},
          {"min", f.min},
          {"max", f.max}
        });
      }
    }
    res.set_content(json.dump(), "application/json");
  });

  pimpl->http.Get("/mesh", [state](const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(state->mtx);
    if (state->mesh_version == 0) {
      res.status = 404;
      res.set_content("no mesh has been set", "text/plain");
      return;
    }
    res.set_content(state->mesh_blob, "application/octet-stream");
  });

  pimpl->http.Get(R"(/space/(\d+))", [state](const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(state->mtx);
    size_t s = std::stoul(req.matches[1]);
    if (s >= state->spaces.size()) {
      res.status = 404;
      res.set_content("no such space", "text/plain");
      return;
    }
    res.set_content(state->spaces[s].blob, "application/octet-stream");
  });

  pimpl->http.Get(R"(/field/(\d+))", [state](const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(state->mtx);
    size_t i = std::stoul(req.matches[1]);
    if (i >= state->fields.size()) {
      res.status = 404;
      res.set_content("no such field", "text/plain");
      return;
    }
    res.set_content(state->fields[i].blob, "application/octet-stream");
  });

  if (!pimpl->http.bind_to_port("0.0.0.0", s.port)) {
    std::cout << "error: femto::server failed to bind to port " << s.port << std::endl;
    exit(1);
  }

  pimpl->listener = std::thread([state] { state->http.listen_after_bind(); });

  std::cout << "femto::server listening on " << url() << std::endl;

}

server::~server() {
  pimpl->http.stop();
  if (pimpl->listener.joinable()) {
    pimpl->listener.join();
  }
}

std::string server::url() const {
  return "http://localhost:" + std::to_string(pimpl->opts.port);
}

void server::wait() {
  if (pimpl->listener.joinable()) {
    pimpl->listener.join();
  }
}

void server::set_mesh(const Mesh<memory::space::cpu> & mesh) {

  uint32_t dim = mesh.spatial_dimension;
  uint32_t gdim = mesh.geometry_dimension;

  FEMTO_ASSERT(gdim == 2 || gdim == 3, "femto::server only supports 2D and 3D meshes");
  FEMTO_ASSERT(mesh.X.family == Family::H1, "femto::server expects H1 mesh coordinates");
  FEMTO_ASSERT(1 <= mesh.X.degree && mesh.X.degree <= 3, "femto::server only supports meshes of degree 1, 2, 3");

  auto * state = pimpl.get();
  std::lock_guard<std::mutex> lock(state->mtx);

  state->tri_row_width = mesh.tri.shape[1];
  state->quad_row_width = mesh.quad.shape[1];

  auto copy_row = [](std::vector<Connection> & dst, const auto & src, uint32_t i) {
    for (uint32_t j = 0; j < src.shape[1]; j++) { dst.push_back(src(i, j)); }
  };

  state->tri_rows.clear();
  state->quad_rows.clear();

  if (gdim == 2) {

    // draw the elements themselves
    state->num_tris = mesh.tri.shape[0];
    state->num_quads = mesh.quad.shape[0];
    for (uint32_t i = 0; i < state->num_tris; i++) { copy_row(state->tri_rows, mesh.tri, i); }
    for (uint32_t i = 0; i < state->num_quads; i++) { copy_row(state->quad_rows, mesh.quad, i); }

  } else {

    // draw the faces on the boundary surface (and serialize only the
    // nodes that appear on it, to keep the data footprint small)
    SubMesh<> bdr = boundary_of(mesh);
    state->num_tris = bdr.tri.shape[0];
    state->num_quads = bdr.quad.shape[0];
    for (uint32_t i = 0; i < state->num_tris; i++) { copy_row(state->tri_rows, mesh.tri, bdr.tri(i)); }
    for (uint32_t i = 0; i < state->num_quads; i++) { copy_row(state->quad_rows, mesh.quad, bdr.quad(i)); }

  }

  state->spaces.clear();
  state->fields.clear();
  state->spatial_dimension = dim;
  state->geometry_dimension = gdim;

  uint32_t geo_space = state->space_for(mesh.X.family, mesh.X.degree, mesh.X.offsets);
  const auto & node_ids = state->spaces[geo_space].node_ids;

  std::string blob;
  blob.reserve(12 + node_ids.size() * 12);
  append_u32(blob, dim);
  append_u32(blob, geo_space);
  append_u32(blob, uint32_t(node_ids.size()));
  for (auto v : node_ids) {
    append_f32(blob, float(mesh.X.data(v, 0)));
    append_f32(blob, float(mesh.X.data(v, 1)));
    append_f32(blob, (dim == 3) ? float(mesh.X.data(v, 2)) : 0.0f);
  }

  state->mesh_blob = std::move(blob);
  state->mesh_version++;

}

void server::push_field(const std::string & name, Family family, uint32_t degree, GeometryInfo offsets,
                        const nd::array<double, 2, memory::space::cpu> & data) {

  auto * state = pimpl.get();
  std::lock_guard<std::mutex> lock(state->mtx);

  FEMTO_ASSERT(state->mesh_version > 0, "femto::server: set_mesh() must be called before push_field()");
  FEMTO_ASSERT(degree <= 3, "femto::server only supports fields of degree <= 3");
  FEMTO_ASSERT(!(family == Family::DG && state->geometry_dimension == 3),
               "femto::server: DG fields on 3D meshes are not supported "
               "(their nodes are interior to the elements, so they have no trace on the boundary surface)");

  uint32_t space = state->space_for(family, degree, offsets);
  const auto & node_ids = state->spaces[space].node_ids;
  uint32_t components = data.shape[1];

  FEMTO_ASSERT(data.shape[0] > state->spaces[space].max_node_id,
               "femto::server: field does not match the current mesh");

  double min = +1.0e300;
  double max = -1.0e300;

  std::string blob;
  blob.reserve(12 + node_ids.size() * components * 4);
  append_u32(blob, space);
  append_u32(blob, components);
  append_u32(blob, uint32_t(node_ids.size()));
  for (auto v : node_ids) {
    double norm_sq = 0.0;
    for (uint32_t c = 0; c < components; c++) {
      double value = data(v, c);
      norm_sq += value * value;
      append_f32(blob, float(value));
    }
    // color scale: the value itself for scalar fields,
    // the magnitude for vector-valued fields
    double scalar = (components == 1) ? data(v, 0) : std::sqrt(norm_sq);
    min = std::min(min, scalar);
    max = std::max(max, scalar);
  }

  state->fields.push_back({name, space, components, min, max, std::move(blob)});

}

void server::clear_fields() {
  std::lock_guard<std::mutex> lock(pimpl->mtx);
  pimpl->fields.clear();
}

}  // namespace femto
