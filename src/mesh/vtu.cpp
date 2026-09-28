#include "zlib.h"
#include "femto/mesh/io.hpp"
#include "util.hpp"
#include "base64.hpp"
#include "node_ordering.hpp"

#include <thread>
#include <cstring>
#include <iostream>

std::vector<uint8_t> compress(const std::vector<uint8_t>& uncompressed_data) {
  unsigned long uncompressed_bytes = uncompressed_data.size();
  unsigned long compressed_bytes = compressBound(uncompressed_bytes);
  std::vector<uint8_t> compressed_data(compressed_bytes); // allocate enough size for output buffer
  int error = compress((Bytef *)&compressed_data[0], &compressed_bytes, 
                       (Bytef *)&uncompressed_data[0], uncompressed_bytes);
  if (error) {
    std::cout << "zlib error while compressing: " << error << std::endl;
  } 
  compressed_data.resize(compressed_bytes);
  return compressed_data;
}

namespace io {

template < typename T >
void append_to_byte_array(uint8_t * & ptr, const T & data) {
  std::memcpy(ptr, &data, sizeof(T));
  ptr += sizeof(T);
}

template < typename T >
void append_to_byte_array(uint8_t * & ptr, const std::vector<T> & data) {
  std::memcpy(ptr, &data[0], sizeof(T) * data.size());
  ptr += sizeof(T) * data.size();
}

template < typename new_type, typename old_type, uint32_t n >
std::vector< std::array< new_type, n > > convert(const std::vector< fm::vec< n, old_type > > & old_data) {
  std::vector< std::array< new_type, n > > new_data(old_data.size());
  for (std::size_t i = 0; i < old_data.size(); i++) {
    for (std::size_t j = 0; j < n; j++) {
      new_data[i][j] = new_type(old_data[i][j]);
    }
  }
  return new_data;
}

//template < typename new_type, typename old_type >
//std::vector< new_type > convert(const std::vector< old_type > & old_data) {
//  std::vector< new_type > new_data(old_data.size());
//  for (std::size_t i = 0; i < old_data.size(); i++) {
//    new_data[i] = new_type(old_data[i]);
//  }
//  return new_data;
//}

// although not documented in VTK's official .vtu, spec, following <https://itk.org/Wiki/VTK_XML_Formats>,
// the binary data header is formatted as:
//     [#blocks][#u-size][#p-size][#c-size-1][#c-size-2]...[#c-size-#blocks][DATA]
// where each token is an integer value whose type is specified by the field  "header_type" at the top of the file
// (UInt32 if no type specified), and where the token values are:
//     [#blocks] = Number of blocks
//     [#u-size] = Block size before compression
//     [#p-size] = Size of last partial block (zero if it not needed)
//     [#c-size-i] = Size in bytes of block i after compression
// Once the data is compressed and the header is generated, the data can be written as
//     output << Base64::encode(header) << Base64::encode(join(compress(data_1), compress(data_2), ...))
template < typename header_int_t >
void write_compressed_data(const std::vector<uint8_t> &data_bytes,
                           std::ofstream &outfile,
                           std::size_t bytes_per_block)
{
  header_int_t total_bytes = data_bytes.size();
  header_int_t remainder = total_bytes % bytes_per_block;
  header_int_t quotient = total_bytes / bytes_per_block;
  header_int_t size_of_last_block, number_of_blocks;
  if (remainder == 0) {
    number_of_blocks = quotient;
    size_of_last_block = bytes_per_block;
  } else {
    number_of_blocks = quotient + 1;
    size_of_last_block = remainder;
  }

  std::vector<uint8_t> header_bytes((3 + number_of_blocks) * sizeof(header_int_t));
  header_int_t * header = (header_int_t *)&header_bytes[0];
  header[0] = number_of_blocks;
  header[1] = bytes_per_block;
  header[2] = size_of_last_block;

  std::vector< std::vector<uint8_t> > compressed_bytes(number_of_blocks);

  std::vector< std::thread > threads;
  int num_threads = std::thread::hardware_concurrency();
  for (int tid = 0; tid < num_threads; tid++) {
    threads.push_back(std::thread([&, tid](){
      for (header_int_t i = tid; i < number_of_blocks; i += num_threads) {
        header_int_t block_size = (i == number_of_blocks - 1) ? size_of_last_block : bytes_per_block;
        std::vector<uint8_t>::const_iterator start = data_bytes.begin() + i * bytes_per_block;
        std::vector<uint8_t>::const_iterator stop  = start + block_size;
        std::vector<uint8_t> block_vector(start, stop);
        compressed_bytes[i] = compress(block_vector);
        header[3+i] = compressed_bytes[i].size();
      }
    }));
  };

  for (auto & thread : threads) { thread.join(); }

  std::size_t total_compressed_bytes = 0;
  for (const auto & buffer : compressed_bytes) {
    total_compressed_bytes += buffer.size();
  }

  std::vector< uint8_t > compressed_bytes_joined;
  compressed_bytes_joined.reserve(total_compressed_bytes);
  for (const auto & buffer : compressed_bytes) {
    compressed_bytes_joined.insert(compressed_bytes_joined.end(), buffer.begin(), buffer.end());
  }

  outfile << Base64::Encode(header_bytes);
  outfile << Base64::Encode(compressed_bytes_joined);
  outfile << '\n';
}

std::string type_name(uint32_t) { return "UInt32"; }
std::string type_name(int32_t) { return "Int32"; }
std::string type_name(float) { return "Float32"; }
std::string type_name(double) { return "Float64"; }

template < typename float_t, typename int_t, typename header_int_t = uint32_t >
bool export_vtu_impl(const Mesh & mesh, std::string filename, std::size_t block_size = (1 << 16)) {

  int_t num_nodes = int_t(mesh.nodes.size());
  int_t num_elements = int_t(mesh.elements.size());

  std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);

  outfile << "<?xml version=\"1.0\"?>\n";
  std::string byte_order = is_big_endian ? "BigEndian" : "LittleEndian";
  outfile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"" << byte_order
          << "\" compressor=\"vtkZLibDataCompressor\" header_type=\"" << type_name(header_int_t{}) << "\">\n";
  outfile << "<UnstructuredGrid>\n";
  outfile << "<Piece NumberOfPoints=\"" << mesh.nodes.size() << "\" NumberOfCells=\"" << mesh.elements.size() << "\">\n";

  outfile << "<Points>\n";
  outfile << "<DataArray type=\"" << type_name(float_t{}) << "\" Name=\"Points\" NumberOfComponents=\"3\" format=\"binary\">\n";
  {
    uint32_t data_bytes = num_nodes * sizeof(float_t) * 3;
    std::vector<uint8_t> byte_vector(data_bytes);
    uint8_t * ptr = &byte_vector[0];
    append_to_byte_array(ptr, convert< float_t >(mesh.nodes));
    write_compressed_data<header_int_t>(byte_vector, outfile, block_size);
  }
  outfile << "</DataArray>\n";
  outfile << "</Points>\n";

  outfile << "<Cells>\n";
  outfile << "<DataArray type=\"" << type_name(int_t{}) << "\" Name=\"connectivity\" format=\"binary\">\n";
  {
    uint32_t data_bytes = 0;
    for (auto & elem : mesh.elements) {
      data_bytes += sizeof(int_t) * nodes_per_elem(elem.type);
    }
    std::vector<uint8_t> byte_vector(data_bytes);
    uint8_t * ptr = &byte_vector[0];
    for (auto & elem : mesh.elements) {
      for (int32_t i : vtk::permutation(elem.type)) {
        int_t id = elem.node_ids[i];
        append_to_byte_array(ptr, id);
      }
    }
    write_compressed_data<header_int_t>(byte_vector, outfile, block_size);
  }
  outfile << "</DataArray>\n";



  outfile << "<DataArray type=\"" << type_name(int_t{}) << "\" Name=\"offsets\" format=\"binary\">\n";
  {
    uint32_t data_bytes = num_elements * sizeof(int_t);
    std::vector<uint8_t> byte_vector(data_bytes);
    uint8_t * ptr = &byte_vector[0];
    int_t offset = 0;
    for (auto & elem : mesh.elements) {
      offset += nodes_per_elem(elem.type);
      append_to_byte_array(ptr, offset);
    }
    write_compressed_data<header_int_t>(byte_vector, outfile, block_size);
  }
  outfile << "</DataArray>\n";

  outfile << "<DataArray type=\"UInt8\" Name=\"types\" format=\"binary\">\n";
  {
    uint32_t data_bytes = num_elements;
    std::vector<uint8_t> byte_vector(data_bytes);
    uint8_t * ptr = &byte_vector[0];
    for (auto & elem : mesh.elements) {
      uint8_t vtk_id = vtk::element_type(elem.type);
      append_to_byte_array(ptr, vtk_id);
    }
    write_compressed_data<header_int_t>(byte_vector, outfile, block_size);
  }
  outfile << "</DataArray>\n";
  outfile << "</Cells>\n";

  // if every element of the mesh has integer tags
  if (mesh.elements[0].tags.size() > 0) {
    outfile << "<CellData Scalars=\"material\">\n";
    outfile << "<DataArray type=\"" << type_name(int32_t{}) << "\" Name=\"material\" format=\"binary\">\n";
    {
      int num_tags = mesh.elements[0].tags.size();
      uint32_t data_bytes = num_tags * mesh.elements.size() * sizeof(int32_t);
      std::vector<uint8_t> byte_vector(data_bytes);
      uint8_t * ptr = &byte_vector[0];
      for (auto & elem : mesh.elements) {
        for (int32_t tag : elem.tags) {
          append_to_byte_array(ptr, tag);
        }
      }
      write_compressed_data<header_int_t>(byte_vector, outfile, block_size);
    }
    outfile << "</DataArray>\n";
    outfile << "</CellData>\n";
  }

  outfile << "</Piece>\n";
  outfile << "</UnstructuredGrid>\n";
  outfile << "</VTKFile>\n";

  outfile.close();

  return false;
}

bool export_vtu(const Mesh & mesh, std::string filename) {
  return export_vtu_impl<float, int32_t, uint32_t>(mesh, filename);
}

}
