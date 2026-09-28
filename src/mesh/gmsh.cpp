#include "femto/mesh/io.hpp"

#include "fm/operations/print.hpp" // operator<< for the node coordinates

#include "util.hpp"
#include "node_ordering.hpp"

#include <map>
#include <tuple>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

static bool export_gmsh_v22_binary(const io::Mesh & mesh, std::string filename) {

  std::ofstream outfile(filename);

  //////////////////
  // write header //
  //////////////////
  const double version = 2.2;
  const int filetype = 1; // denotes binary encoding
  const int datasize = 8;
  const int one = 1;
  outfile << "$MeshFormat\n";
  outfile << version << " " << filetype << " " << datasize << std::endl;
  outfile.write((char*)&one, sizeof(int)); // used for checking endianness
  outfile << "\n$EndMeshFormat\n";

  /////////////////
  // write nodes //
  /////////////////
  outfile << "$Nodes\n";
  outfile << mesh.nodes.size() << std::endl;
  for (int i = 0; i < mesh.nodes.size(); i++) {
    int id = i + 1; // gmsh uses 1-based indexing
    outfile.write((char*)&id, sizeof(int));
    outfile.write((char*)&mesh.nodes[i], sizeof(double) * 3); 
  }
  outfile << "\n$EndNodes\n";

  /////////////////
  // write elems //
  /////////////////
  outfile << "$Elements\n";
  outfile << mesh.elements.size() << std::endl;

  // group elements together by type 
  std::map< io::Element::Type, std::vector< const io::Element * > > element_blocks;
  for (int i = 0; i < mesh.elements.size(); i++) {
    const io::Element * e = &mesh.elements[i];
    element_blocks[e->type].push_back(e);
  }

  // then write each block out
  for (auto & [type, block] : element_blocks) {
    int etype = gmsh::element_type(type);
    int block_size = int(block.size());

    // gmsh requires two tags: ("physical" and "elementary")
    // if not provided, these will be set to zero.
    // this assumes all elements in the block have the same
    // number of tags
    int num_tags = std::max(int(block[0]->tags.size()), 2);

    outfile.write((char*)&etype, sizeof(int));
    outfile.write((char*)&block_size, sizeof(int));
    outfile.write((char*)&num_tags, sizeof(int));
    int count = 1; // gmsh uses 1-based indexing
    for (auto * elem : block) {
      outfile.write((char*)&count, sizeof(int));
      count++;

      for (int i = 0; i < num_tags; i++) {
        int tag = (i < elem->tags.size()) ? elem->tags[i] : 0;
        outfile.write((char*)&tag, sizeof(int));
      }

      // gmsh uses 1-based indexing
      auto node_ids = elem->node_ids;
      for (auto & id : node_ids) { id++; }
      outfile.write((char*)&node_ids[0], sizeof(int) * node_ids.size());
    }
  }

  outfile << "\n$EndElements\n";

  outfile.close();

  return false;

}

static bool export_gmsh_v22_ascii(const io::Mesh & mesh, std::string filename) {

  std::ofstream outfile(filename);

  //////////////////
  // write header //
  //////////////////
  const double version = 2.2;
  const int filetype = 0; // denotes ascii encoding
  const int datasize = 8;
  outfile << "$MeshFormat\n";
  outfile << version << " " << filetype << " " << datasize << '\n';
  outfile << "$EndMeshFormat\n";

  /////////////////
  // write nodes //
  /////////////////
  outfile << "$Nodes\n";
  outfile << mesh.nodes.size() << '\n';
  for (int i = 0; i < mesh.nodes.size(); i++) {
    outfile << i+1 << " " << mesh.nodes[i] << '\n';
  }
  outfile << "$EndNodes\n";

  /////////////////
  // write elems //
  /////////////////
  outfile << "$Elements\n";
  outfile << mesh.elements.size() << std::endl;
  for (int i = 0; i < mesh.elements.size(); i++) {
    auto [type, ids, tags] = mesh.elements[i];
    outfile << i+1 << " " << gmsh::element_type(type) << " " << tags.size();
    for (auto tag : tags) { outfile << " " << tag; }
    for (auto id : ids) { outfile << " " << id+1; }
    outfile << '\n';
  }
  outfile << "$EndElements\n";

  outfile.close();

  return false;

}

io::Mesh import_gmsh_v22_ascii(std::ifstream & infile) {

  io::Mesh mesh;

  ////////////////
  // read nodes //
  ////////////////
  std::string line;
  infile >> line;
  if (line != "$Nodes") exit_with_error("invalid file format (nodes)");

  int node_id;
  int num_nodes;
  infile >> num_nodes;
  mesh.nodes.resize(num_nodes);

  for (int i = 0; i < num_nodes; i++) {
    infile >> node_id;
    mesh.nodes[node_id - 1] = ascii_read_array<double,3>(infile);
  }

  infile >> line;
  if (line != "$EndNodes") exit_with_error("invalid file format (nodes)");

  ////////////////
  // read elems //
  ////////////////

  // skip through any data sections other than $Elements
  do {
    getline(infile, line);
  } while (infile && line != "$Elements");

  int num_elems;
  infile >> num_elems;

  mesh.elements.resize(num_elems);

  for (int i = 0; i < num_elems; i++) {
    fm::vec<3,int> info = ascii_read_array<int, 3>(infile); 
    int elem_id = info[0];
    int elem_type = info[1];
    int num_tags = info[2];

    // note: gmsh uses 1-based indexing
    auto & e = mesh.elements[elem_id - 1];

    e.type = gmsh::element_type(elem_type);

    if (e.type == io::Element::Type::Unsupported) {
      std::cout << "error: encountered unsupported element type" << std::endl;
      io::print_supported_elements();
      exit(1);
    }

    e.tags.resize(num_tags);
    for (int j = 0; j < num_tags; j++) infile >> e.tags[j];

    int npe = io::nodes_per_elem(e.type);
    e.node_ids.resize(npe);
    for (int j = 0; j < npe; j++) {
      infile >> e.node_ids[j];
      e.node_ids[j]--; // one-based to zero-based indices
    }

    getline(infile, line);
  }

  infile >> line;
  if (line != "$EndElements") exit_with_error("invalid file format (elems)");

  infile.close();

  return mesh;
}

io::Mesh import_gmsh_v22_binary(std::ifstream & infile, bool swap_bytes) {

  io::Mesh mesh;

  auto read = [&](auto * ptr, std::streamsize size){
    infile.read((char*)ptr, size);
    if (swap_bytes) {
      uint8_t * byte_ptr = reinterpret_cast<uint8_t*>(ptr);
      std::reverse(byte_ptr, byte_ptr + sizeof(*ptr));
    }
  };

  ////////////////
  // read nodes //
  ////////////////
  std::string line;
  infile >> line;
  if (line != "$Nodes") exit_with_error("invalid file format (nodes)");

  int id;
  int num_nodes;
  infile >> num_nodes;
  mesh.nodes.resize(num_nodes);
  getline(infile, line); // skip the newline
  for (int i = 0; i < num_nodes; i++) {
    // note: gmsh uses 1-based indexing
    read((char*)&id, sizeof(int)); 
    read((char*)&mesh.nodes[id-1], sizeof(double) * 3);
  }

  infile >> line;
  if (line != "$EndNodes") exit_with_error("invalid file format (nodes)");

  // skip through any data sections other than $Elements
  do {
    getline(infile, line);
  } while (infile && line != "$Elements");

  int num_elems;
  infile >> num_elems;
  mesh.elements.resize(num_elems);

  getline(infile, line); // skip the newline after the number of elements

  // iterate through all the blocks of elements
  // until all of the elements are accounted for
  int element_count = 0;
  do {
    auto [gmsh_type, block_size, num_tags] = binary_read_array<int,3>(infile);

    io::Element::Type type = gmsh::element_type(gmsh_type);
    int npe = nodes_per_elem(type);

    for (int i = 0; i < block_size; i++) {
      int elem_id;
      read((char*)&elem_id, sizeof(int));

      io::Element & e = mesh.elements[elem_id - 1];
      e.type = type;
      e.tags.resize(num_tags);
      read((char*)&e.tags[0], sizeof(int) * num_tags);

      // note: gmsh uses 1-based indexing
      e.node_ids.resize(npe);
      read((char*)&e.node_ids[0], sizeof(int) * npe);
      for (auto & id : e.node_ids) { id--; }
    }

    element_count += block_size;
  } while (element_count < num_elems);

  infile >> line;

  if (line != "$EndElements") exit_with_error("invalid file format (elems)");

  infile.close();

  return mesh;

}

namespace io {

bool export_gmsh_v22(const Mesh & mesh, std::string filename, FileEncoding enc) {
  if (enc == FileEncoding::ASCII) {
    return export_gmsh_v22_ascii(mesh, filename);
  } else {
    return export_gmsh_v22_binary(mesh, filename);
  }
}

Mesh import_gmsh_v22(std::string filename) {

  std::ifstream infile(filename);

  if (!infile) {
    std::cout << "error: " << filename << " not found" << std::endl;
    exit(1);
  }

  std::string line;

  /////////////////
  // read header //
  /////////////////
  infile >> line;
  if (line != "$MeshFormat") exit_with_error("invalid file format (header)" + line);

  double version_number;
  int filetype;
  int datasize;

  infile >> version_number >> filetype >> datasize;
  if (version_number != 2.2) exit_with_error("unsupported version number");
  if (filetype > 1) exit_with_error("unsupported file type");
  if (datasize != 8) exit_with_error("invalid data size field");

  if (filetype == 0) {
    infile >> line;
    if (line != "$EndMeshFormat") {
      exit_with_error("invalid file format (end header):" + line);
    }
    return import_gmsh_v22_ascii(infile);
  } else {
    getline(infile, line); // skip the newline
    int one;
    infile.read((char*)&one, 4);
    bool swap_bytes = (one != 1);
    infile >> line;
    if (line != "$EndMeshFormat") {
      exit_with_error("invalid file format (end header):" + line);
    }
    return import_gmsh_v22_binary(infile, swap_bytes);
  }

}

} // namespace io
