#include "femto/mesh/io.hpp"

#include "util.hpp"
#include "node_ordering.hpp"

#include <fstream>

using io::Mesh;
  
template <typename T>
void write_binary(std::ofstream &outfile, T value) {
  T be_value = to_big_endian(value);
  outfile.write((char*)&be_value, sizeof(T));
}

static bool export_vtk_ascii(const Mesh & mesh, std::string filename) {

  std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);

  outfile << "# vtk DataFile Version 3.0\n";
  outfile << "--------------------------\n";
  outfile << "ASCII\n";
  outfile << "DATASET UNSTRUCTURED_GRID\n";

  outfile << "POINTS " << mesh.nodes.size() << " float\n";
  for (auto node : mesh.nodes) {
    double x = node[0];
    double y = node[1];
    double z = node[2];
    outfile << x << " " << y << " " << z << '\n';
  }

  int32_t nelems = mesh.elements.size();
  int32_t size = 0;
  for (auto & elem : mesh.elements) {
    size += 1 + nodes_per_elem(elem.type);
  }
  outfile << "CELLS " << nelems << " " << size << '\n';
  for (auto & elem : mesh.elements) {
    if (elem.type == io::Element::Type::Pyr14) {
      exit_with_error("vtk does not support 14-node pyramid elements");
    }
    outfile << nodes_per_elem(elem.type);
    for (int32_t i : vtk::permutation(elem.type)) {
      int32_t id = elem.node_ids[i];
      outfile << " " << id;
    }
    outfile << '\n';
  }

  outfile << "CELL_TYPES " << nelems << '\n';
  for (auto & elem: mesh.elements) {
    outfile << vtk::element_type(elem.type) << '\n';
  }
  outfile.close();

  return false;
}

static bool export_vtk_binary(const Mesh & mesh, std::string filename) {

  std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);

  outfile << "# vtk DataFile Version 3.0\n";
  outfile << "--------------------------\n";
  outfile << "BINARY\n";
  outfile << "DATASET UNSTRUCTURED_GRID\n";

  outfile << "POINTS " << mesh.nodes.size() << " float\n";
  for (auto p : mesh.nodes) {
    write_binary(outfile, float(p[0]));
    write_binary(outfile, float(p[1]));
    write_binary(outfile, float(p[2]));
  }
  outfile << '\n';

  int32_t nelems = mesh.elements.size();
  int32_t size = 0;
  for (auto & elem : mesh.elements) {
    size += 1 + nodes_per_elem(elem.type);
  }
  outfile << "CELLS " << nelems << " " << size << '\n';
  for (auto & elem : mesh.elements) {
    int32_t npe = nodes_per_elem(elem.type);
    write_binary(outfile, npe);
    for (int32_t i : vtk::permutation(elem.type)) {
      int32_t id = elem.node_ids[i];
      write_binary(outfile, id);
    }
  }
  outfile << '\n';
  
  outfile << "CELL_TYPES " << nelems << '\n';
  for (auto & elem: mesh.elements) {
    write_binary(outfile, vtk::element_type(elem.type));
  }
  outfile << '\n';

  outfile.close();

  return false;
}

namespace io {

bool export_vtk(const Mesh & mesh, std::string filename, FileEncoding enc) {
  if (enc == FileEncoding::ASCII) {
    return export_vtk_ascii(mesh, filename);
  } else {
    return export_vtk_binary(mesh, filename);
  }
}

} // namespace io
